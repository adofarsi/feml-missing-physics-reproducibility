import copy
import os
import warnings
import numpy as np
import torch
from firedrake import (Constant, Function, Identity, Mesh, TestFunction, TrialFunction,
                       VectorFunctionSpace, as_vector, assemble, dot, ds, dx, grad,
                       inner, solve, sym, tr)
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, set_working_tape, stop_annotating)
from firedrake.ml.pytorch import fem_operator, to_torch

import shear_targets as T
from operators import IntegralMonotoneSigmaY, IntegralHTangent, MonotoneReLUHazard
from shear_solver import (SHEAR_CONFIG, build_hardening_ml_ops,
                          build_nonlocal_nn_damage_ml_ops, plane_stress_svk_solver,
                          plane_stress_svk_nonlocal_nn_damage_solver)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "outputs", "shear_coupon")
TENSILE_MESH = os.path.join(ROOT, "data", "tensile_coupon_2d.msh")
SHEAR_MESH = os.path.join(ROOT, "data", "shear_coupon_2d.msh")
SHEAR_CSV = os.path.join(ROOT, "data", "shear.csv")
TENSILE_CSV = os.path.join(ROOT, "data", "tensile.csv")
CORRECTED_CSV = os.path.join(OUTPUT, "corrected_shear.csv")

E_REF = 219336.0
NU = 0.30
WEAK_BC_PENALTY = 1.0e9
PROBE_DISP_MM = 0.02
TENSILE_ELASTIC_WINDOW = (0.06, 0.14)
SHEAR_ELASTIC_MAX_MM = 0.34

TENSILE_CONFIG = {
    "name": "tensile",
    "mesh_path": TENSILE_MESH,
    "thickness": 3.150,
    "left": (1, 11, 12),
    "right": (5, 6, 7),
    "load_dir": (1.0, 0.0),
    "gage_points": ((-12.7, 0.0), (12.7, 0.0)),
}
CALIB_SHEAR_CONFIG = {
    "name": "shear",
    "mesh_path": SHEAR_MESH,
    "thickness": 3.100,
    "left": (7, 8, 12),
    "right": (10, 9, 11),
    "load_dir": (0.0, 1.0),
    "gage_points": None,
}

HARD_NEPOCHS = 60
HARD_REL_LR = 0.03
LR_DECAY_ON_DIVERGENCE = 0.5
SIGMA_Y0_INIT = 750.0
HARD_HIDDEN_SIZE = 32
HARD_P_SCALE = 100.0
HARD_N_QUAD = 16
HARD_STRESS_SCALE = 5000.0

PL_SCALE = 103.15401504400896
PL_P0 = 0.11419575298286135
PL_M = 2.2704217218701723
ELL_FIXED = 1.3788880735813305
DEG_S = 2.0
D_MAX = 0.95

DMG_NEPOCHS = 60
DMG_N_TRAIN = 60
DMG_N_EVAL = 80
DMG_LR = 5e-3
DMG_HIDDEN = 8
DMG_POWER = 2
DMG_GRAD_CLIP = 1.0
DMG_WS_SCALE = 1.0
DMG_PATIENCE = 2


def load_curve(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, filling_values=np.nan)
    displacement = np.asarray(data["displacement_mm"], dtype=float)
    columns = [name for name in data.dtype.names if name != "displacement_mm"]
    forces = np.column_stack([np.asarray(data[name], dtype=float) for name in columns])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_force = np.nanmean(forces, axis=1)
    return displacement, forces, mean_force, columns


def slope_with_intercept(displacement, force, lo, hi):
    mask = np.isfinite(force) & (displacement >= lo) & (displacement <= hi)
    return float(np.polyfit(displacement[mask], force[mask], 1)[0])


def slope_through_origin(displacement, force, disp_max):
    mask = np.isfinite(force) & (displacement > 0.0) & (displacement <= disp_max)
    d, f = displacement[mask], force[mask]
    return float(np.sum(d * f) / np.sum(d * d))


def elastic_probe(E, cfg):
    mesh = Mesh(str(cfg["mesh_path"]))
    mu = Constant(E / (2.0 * (1.0 + NU)))
    lam_ps = Constant(E * NU / (1.0 - NU**2))
    penalty = Constant(WEAK_BC_PENALTY)
    I2d = Identity(2)

    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    uh = Function(V, name="displacement")

    load_dir = as_vector([Constant(cfg["load_dir"][0]), Constant(cfg["load_dir"][1])])
    u_loaded = as_vector(
        [Constant(PROBE_DISP_MM * cfg["load_dir"][0]),
         Constant(PROBE_DISP_MM * cfg["load_dir"][1])]
    )

    def sigma(w):
        eps = sym(grad(w))
        return lam_ps * tr(eps) * I2d + 2.0 * mu * eps

    a = inner(sigma(u), sym(grad(v))) * dx
    L = 0
    for label in cfg["left"]:
        a += penalty * dot(u, v) * ds(label)
    for label in cfg["right"]:
        a += penalty * dot(u, v) * ds(label)
        L += penalty * dot(u_loaded, v) * ds(label)

    solve(a == L, uh, solver_parameters={
        "ksp_type": "preonly", "pc_type": "lu",
        "mat_type": "aij", "snes_type": "ksponly",
    })

    reaction = 0.0
    for label in cfg["right"]:
        reaction += assemble(-penalty * dot(uh - u_loaded, load_dir) * ds(label))
    load_kN = abs(float(reaction)) * (cfg["thickness"] / 1000.0)

    k_grip = load_kN / PROBE_DISP_MM
    gage_span = None
    k_gage = None
    if cfg["gage_points"] is not None:
        axis = 0 if abs(cfg["load_dir"][0]) >= abs(cfg["load_dir"][1]) else 1
        pa, pb = cfg["gage_points"]
        gage_span = abs(float(uh.at(pb)[axis] - uh.at(pa)[axis]))
        k_gage = load_kN / gage_span

    return {"load_kN": load_kN, "k_grip": k_grip, "gage_span": gage_span, "k_gage": k_gage}


def calibrate_and_correct():
    t_disp, _, t_mean, _ = load_curve(TENSILE_CSV)
    k_exp_tensile = slope_with_intercept(t_disp, t_mean, *TENSILE_ELASTIC_WINDOW)
    probe_t_ref = elastic_probe(E_REF, TENSILE_CONFIG)
    k_sim_tensile_gage = probe_t_ref["k_gage"]
    E_cal = E_REF * k_exp_tensile / k_sim_tensile_gage

    s_disp, s_forces, s_mean, s_cols = load_curve(SHEAR_CSV)
    k_exp_shear = slope_through_origin(s_disp, s_mean, SHEAR_ELASTIC_MAX_MM)
    probe_s = elastic_probe(E_cal, CALIB_SHEAR_CONFIG)
    k_sim_shear = probe_s["k_grip"]
    C_par = 1.0 / k_exp_shear - 1.0 / k_sim_shear

    correction = C_par * np.where(np.isfinite(s_mean), s_mean, 0.0)
    s_disp_corr = np.clip(s_disp - correction, 0.0, None)
    header = "displacement_mm," + ",".join(s_cols)
    np.savetxt(CORRECTED_CSV, np.column_stack([s_disp_corr, s_forces]),
               delimiter=",", header=header, comments="", fmt="%.10g")
    print(f"[step 1] E = {E_cal:.1f} MPa, C_par = {C_par:.6e} mm/kN; wrote corrected_shear.csv")
    return E_cal


def build_hardening_optimizer(model_sy):
    groups = []
    for param in model_sy.parameters():
        scale = max(float(param.detach().abs().max()), 1.0)
        groups.append({"params": [param], "lr": HARD_REL_LR * scale})
    return torch.optim.Adam(groups)


def project_parameters(model):
    with torch.no_grad():
        if not model._sigma_y0_is_fixed:
            scale = model._sy0_scale
            lo = torch.log(torch.expm1(torch.tensor(100.0, dtype=torch.float64) / scale))
            hi = torch.log(torch.expm1(torch.tensor(2500.0, dtype=torch.float64) / scale))
            model._raw_sigma_y0.clamp_(float(lo), float(hi))


def train_hardening(E):
    tg = T.hardening_targets(CORRECTED_CSV)
    levels = tg["displacement_levels"]
    branches = [
        (tg["is_elastic"], T.HARD_W_ELASTIC),
        (tg["is_transition"], T.HARD_W_TRANSITION),
        (tg["is_outer"], T.HARD_W_OUTER),
    ]

    continue_annotation()
    get_working_tape().clear_tape()
    torch.manual_seed(0)
    np.random.seed(0)

    with stop_annotating():
        mesh = Mesh(str(SHEAR_CONFIG["mesh_path"]))

    model_sy = IntegralMonotoneSigmaY(
        sigma_y0_init=SIGMA_Y0_INIT, hidden_size=HARD_HIDDEN_SIZE, p_scale=HARD_P_SCALE,
        n_quad=HARD_N_QUAD, stress_scale=HARD_STRESS_SCALE,
    ).double()
    model_Hp = IntegralHTangent(model_sy).double()
    ml_ops, ml_params = build_hardening_ml_ops(mesh, model_sy, model_Hp)

    ml_parameters_sy = to_torch(ml_params["sy"], requires_grad=True)
    ml_parameters_Hp = to_torch(ml_params["Hp"], requires_grad=True)

    def compute_loss(params):
        sy_p, Hp_p = params
        _, loads, _ = plane_stress_svk_solver(
            levels, mesh, ml_ops, {"sy": sy_p, "Hp": Hp_p},
            SHEAR_CONFIG, E, NU, WEAK_BC_PENALTY,
        )
        loss = T.force_curve_loss(loads, tg["mean_force"], tg["sigma_eff"], branches)
        print(f"Loss = {loss}")
        return loss

    with set_working_tape() as tape:
        tape.clear_tape()
        controls = [ml_params["sy"], ml_params["Hp"]]
        rf = ReducedFunctional(compute_loss(controls), [Control(c) for c in controls])
        fem_op = fem_operator(rf)

    optimiser = build_hardening_optimizer(model_sy)
    losses = []
    last_good = copy.deepcopy(model_sy.state_dict())
    best_loss = float("inf")
    best_state = copy.deepcopy(model_sy.state_dict())
    for epoch in range(HARD_NEPOCHS):
        model_sy.train()
        model_sy.zero_grad()
        snapshot = copy.deepcopy(model_sy.state_dict())
        try:
            loss = fem_op(ml_parameters_sy, ml_parameters_Hp)
            total_loss = float(loss.item())
            loss.backward()
        except Exception as exc:
            print(f"Epoch {epoch + 1}/{HARD_NEPOCHS}: forward diverged "
                  f"({type(exc).__name__}: {exc}); reverting + LR x {LR_DECAY_ON_DIVERGENCE}")
            model_sy.load_state_dict(last_good)
            for group in optimiser.param_groups:
                group["lr"] *= LR_DECAY_ON_DIVERGENCE
            continue
        last_good = copy.deepcopy(model_sy.state_dict())
        if total_loss < best_loss:
            best_loss = total_loss
            best_state = snapshot
        optimiser.step()
        project_parameters(model_sy)
        losses.append(total_loss)
        print(f"Epoch {epoch + 1}/{HARD_NEPOCHS}: Loss = {total_loss:.6e}, "
              f"sigma_y0 = {model_sy.sigma_y0.item():.6g} MPa")

    model_sy.load_state_dict(best_state)
    print(f"[step 2] hardening done; best loss = {best_loss:.6e}")
    torch.save(model_sy.state_dict(), os.path.join(OUTPUT, "model_hardening.pth"))
    np.savetxt(os.path.join(OUTPUT, "hardening_losses.dat"), np.asarray(losses))
    return model_sy


def warmstart_powerlaw(model_phi, n_iter=6000, lr=1e-2, target_scale=1.0):
    pg = torch.linspace(0.0, 0.8, 400, dtype=torch.float64)
    tgt = target_scale * (PL_SCALE / (PL_M + 1.0)) * torch.clamp(pg - PL_P0, min=0.0) ** (PL_M + 1.0)
    opt = torch.optim.Adam(model_phi.parameters(), lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = ((model_phi(pg) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss.item())


def build_damage_targets(csv_path, n_steps):
    dg = T.damage_targets(csv_path)
    fd, fm, fs = dg["full_disp"], dg["full_mean"], dg["full_std"]
    d_peak, d_fit_max = dg["d_peak"], dg["d_fit_max"]
    levels = np.linspace(d_fit_max / n_steps, d_fit_max, n_steps)
    tgt_mean = np.interp(levels, fd, fm)
    tgt_std = np.interp(levels, fd, fs)
    peak = max(float(fm.max()), 1e-12)
    tgt_sigma = np.sqrt(tgt_std**2 + (T.SIGMA_REL_FLOOR * peak) ** 2)
    is_post = levels > d_peak
    branches = [(~is_post, T.DMG_W_PRE_PEAK), (is_post, T.DMG_W_POST_PEAK)]
    return dict(levels=levels, tgt_mean=tgt_mean, tgt_sigma=tgt_sigma, branches=branches,
                fd=fd, fm=fm, fs=fs, d_peak=d_peak, d_fit_max=d_fit_max)


def train_damage(model_sy, E):
    for p in model_sy.parameters():
        p.requires_grad_(False)
    model_Hp = IntegralHTangent(model_sy).double()

    torch.manual_seed(0)
    np.random.seed(0)

    model_phi = MonotoneReLUHazard(
        hidden=DMG_HIDDEN, onset_init=PL_P0, p_max=0.65, power=DMG_POWER).double()
    ws_loss = warmstart_powerlaw(model_phi, n_iter=6000, lr=1e-2, target_scale=DMG_WS_SCALE)
    print(f"[step 3] warmstart MSE = {ws_loss:.3e} ({model_phi.describe()})")

    continue_annotation()
    get_working_tape().clear_tape()
    with stop_annotating():
        mesh = Mesh(str(SHEAR_CONFIG["mesh_path"]))
        ml_ops, ml_params = build_nonlocal_nn_damage_ml_ops(mesh, model_sy, model_Hp, model_phi)

    def forward_loads(levels, ml_params_local):
        _, loads, _, D = plane_stress_svk_nonlocal_nn_damage_solver(
            levels, mesh, ml_ops, ml_params_local, SHEAR_CONFIG, E, NU, WEAK_BC_PENALTY,
            ELL_FIXED, DEG_S, D_MAX)
        return loads, D

    p_phi = to_torch(ml_params["phi"], requires_grad=True)

    tr = build_damage_targets(CORRECTED_CSV, DMG_N_TRAIN)
    get_working_tape().clear_tape()
    with set_working_tape() as tape:
        tape.clear_tape()

        def compute_loss_train(phi_ctrl):
            loads, _ = forward_loads(
                tr["levels"], {"sy": ml_params["sy"], "Hp": ml_params["Hp"], "phi": phi_ctrl})
            return T.force_curve_loss(loads, tr["tgt_mean"], tr["tgt_sigma"], tr["branches"])

        rf_tr = ReducedFunctional(compute_loss_train(ml_params["phi"]), Control(ml_params["phi"]))
        fem_op_tr = fem_operator(rf_tr)

    scalar_ps = model_phi.scalar_parameters()
    scalar_ids = {id(p) for p in scalar_ps}
    mlp_ps = [p for p in model_phi.parameters() if id(p) not in scalar_ids]
    groups = [{"params": scalar_ps, "lr": DMG_LR}]
    if mlp_ps:
        groups.append({"params": mlp_ps, "lr": DMG_LR})
    optimiser = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=DMG_PATIENCE, min_lr=1e-4)
    losses, best_loss, best_state = [], float("inf"), copy.deepcopy(model_phi.state_dict())
    last_good = copy.deepcopy(model_phi.state_dict())
    print(f"[step 3] train N={DMG_N_TRAIN}, epochs={DMG_NEPOCHS}, lr={DMG_LR}, hidden={DMG_HIDDEN}")
    for epoch in range(DMG_NEPOCHS):
        model_phi.train()
        model_phi.zero_grad()
        snapshot = copy.deepcopy(model_phi.state_dict())
        try:
            loss = fem_op_tr(p_phi)
            total = float(loss.item())
            loss.backward()
        except Exception as exc:
            print(f"  epoch {epoch+1}: diverged ({type(exc).__name__}); revert + lr/2")
            model_phi.load_state_dict(last_good)
            for grp in optimiser.param_groups:
                grp["lr"] *= 0.5
            continue
        if DMG_GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model_phi.parameters(), DMG_GRAD_CLIP)
        last_good = copy.deepcopy(model_phi.state_dict())
        if total < best_loss:
            best_loss, best_state = total, snapshot
        optimiser.step()
        prev_lr = optimiser.param_groups[0]["lr"]
        scheduler.step(total)
        if optimiser.param_groups[0]["lr"] < prev_lr - 1e-12:
            optimiser.state.clear()
        losses.append(total)
        print(f"  epoch {epoch+1}/{DMG_NEPOCHS}: loss={total:.6f} "
              f"({model_phi.describe()}, lr={optimiser.param_groups[0]['lr']:.1e})")
    model_phi.load_state_dict(best_state)
    print(f"[step 3] damage done; best loss={best_loss:.6f}")

    ev = build_damage_targets(CORRECTED_CSV, DMG_N_EVAL)
    with stop_annotating():
        loads_f, D_f = forward_loads(ev["levels"], ml_params)
    pred = np.asarray([float(v) for v in loads_f])
    fd, fm = ev["fd"], ev["fm"]
    msk = (fd >= ev["levels"][0]) & (fd <= ev["d_fit_max"])
    rmse = float(np.sqrt(np.mean((np.interp(fd[msk], ev["levels"], pred) - fm[msk]) ** 2)))
    peak = float(fm.max())
    maxD = float(max(x.dat.data_ro.max() for x in D_f))

    np.savetxt(os.path.join(OUTPUT, "damage_losses.dat"), np.asarray(losses))
    torch.save(model_phi.state_dict(), os.path.join(OUTPUT, "model_damage.pth"))
    return rmse, peak, maxD


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    np.random.seed(0)
    torch.manual_seed(0)

    with stop_annotating():
        E = calibrate_and_correct()

    model_sy = train_hardening(E)
    rmse, peak, maxD = train_damage(model_sy, E)

    print(f"[result] damage fit RMSE = {rmse:.4f} kN "
          f"({100.0 * rmse / peak:.2f}% of peak {peak:.4f} kN; max Dbar = {maxD:.3f})")


if __name__ == "__main__":
    main()
