import os
import numpy as np
import torch
from firedrake import *
from firedrake.ml.pytorch import ml_operator, to_torch
from firedrake.__future__ import interpolate

from operators import TorchModelForE, IntegralMonotoneSigmaY, IntegralHTangent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "outputs", "zero_shot_torsion")
MESH = os.path.join(ROOT, "data", "rod_3d.msh")
E_MODEL = os.path.join(ROOT, "outputs", "displacement_controlled", "model.pth")
SY_MODEL = os.path.join(ROOT, "outputs", "load_controlled", "model.pth")

STEPS = 10
DEGREE = 3
NU = 0.3
MAX_THETA = np.radians(3.5)
SIGMA_Y0, R_INF, B_VOCE = 2.354, 0.9, 25.0


def torque_solver(mesh, ml_ops=None, ml_params=None):
    x, y, z = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, "CG", DEGREE)
    Q = FunctionSpace(mesh, "DG", DEGREE - 1)
    T = TensorFunctionSpace(mesh, "DG", DEGREE - 1)
    R = FunctionSpace(mesh, "R", 0)
    top, bottom = 2, 1
    u, v = Function(V, name="Displacement"), TestFunction(V)

    eps_p_old, p_old = Function(T), Function(Q)
    eps_p_new, p_new = Function(T), Function(Q)

    I3 = Identity(3)
    tiny = Constant(1.0e-14)
    E_field, mu_field, lmbda_field = Function(Q), Function(Q), Function(Q)

    def epsilon(w):
        return 0.5 * (grad(w) + grad(w).T)

    def update_elastic():
        if ml_ops is not None:
            E_val = assemble(ml_ops["E"](interpolate(tr(epsilon(u)), Q), ml_params["E"]))
        else:
            kappa = max_value(-tr(epsilon(u)), 0.0)
            E_val = assemble(interpolate(Constant(57.3) + Constant(87.7) * exp(-kappa / Constant(0.008)), Q))
        E_field.assign(E_val)
        mu_field.interpolate(E_field / (2.0 * (1.0 + NU)))
        lmbda_field.interpolate(E_field * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)))

    def radial_return(eps_total, eps_p_n, p_n, sy_field=None, Hp_field=None):
        eps_el = eps_total - eps_p_n
        sigma_tr = lmbda_field * tr(eps_el) * I3 + 2.0 * mu_field * eps_el
        p_hyd = tr(sigma_tr) / 3.0
        s = sigma_tr - p_hyd * I3
        seq = sqrt(1.5 * inner(s, s) + tiny)
        if sy_field is None:
            pv = variable(p_n)
            sy = Constant(SIGMA_Y0) + Constant(R_INF) * (1.0 - exp(-Constant(B_VOCE) * pv))
            Hp = diff(sy, pv)
        else:
            sy, Hp = sy_field, Hp_field
        ft = seq - sy
        dg = conditional(gt(ft, 0.0), ft / (3.0 * mu_field + Hp), 0.0)
        bt = conditional(gt(ft, 0.0), 1.0 - 3.0 * mu_field * dg / seq, 1.0)
        return bt * s + p_hyd * I3, eps_p_n + dg * (1.5 * s / seq), p_n + dg

    def stress(u):
        if ml_ops is not None:
            p_in = interpolate(p_old, Q)
            sy = assemble(ml_ops["sy"](p_in, ml_params["sy"]))
            Hp = assemble(ml_ops["Hp"](p_in, ml_params["Hp"]))
            sig, _, _ = radial_return(epsilon(u), eps_p_old, p_old, sy, Hp)
        else:
            sig, _, _ = radial_return(epsilon(u), eps_p_old, p_old)
        return sig

    theta = Function(R).assign(0.0)
    half = theta / 2
    u_bot = as_vector([Constant(0), y * (cos(half) - 1) + z * sin(half), -y * sin(half) + z * (cos(half) - 1)])
    u_top = as_vector([Constant(0), y * (cos(half) - 1) - z * sin(half), y * sin(half) + z * (cos(half) - 1)])
    bcs = [DirichletBC(V, u_bot, bottom), DirichletBC(V, u_top, top)]

    residual = inner(stress(u), epsilon(v)) * dx
    parameters = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
                  "snes_rtol": 1e-8, "snes_atol": 1e-8, "snes_max_it": 200,
                  "ksp_type": "preonly", "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}

    displacements, stresses, plastic_strains = [], [], []
    for i in range(1, STEPS + 1):
        theta.assign(i * MAX_THETA / STEPS)
        update_elastic()
        solve(residual == 0, u, bcs=bcs, J=derivative(residual, u), solver_parameters=parameters,
              form_compiler_parameters={"quadrature_degree": 6})
        update_elastic()
        if ml_ops is not None:
            sy = assemble(ml_ops["sy"](interpolate(p_old, Q), ml_params["sy"]))
            Hp = assemble(ml_ops["Hp"](interpolate(p_old, Q), ml_params["Hp"]))
            _, ep, pp = radial_return(epsilon(u), eps_p_old, p_old, sy, Hp)
        else:
            _, ep, pp = radial_return(epsilon(u), eps_p_old, p_old)
        eps_p_new.interpolate(ep)
        p_new.interpolate(pp)
        eps_p_old.assign(eps_p_new)
        p_old.assign(p_new)
        displacements.append(u.copy(deepcopy=True))
        stresses.append(Function(T, name="Stress").interpolate(stress(u)))
        plastic_strains.append(p_old.copy(deepcopy=True))
    return displacements, stresses, plastic_strains


def von_mises(sigma, mesh):
    dev = sigma - tr(sigma) / 3.0 * Identity(3)
    return Function(FunctionSpace(mesh, "DG", DEGREE - 1)).interpolate(sqrt(1.5 * inner(dev, dev)))


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    mesh = Mesh(MESH)
    Q = FunctionSpace(mesh, "DG", DEGREE - 1)
    R = FunctionSpace(mesh, "R", 0)

    model_E = TorchModelForE(lower_bound_stiffness=20.0).double()
    model_E.load_state_dict(torch.load(E_MODEL, weights_only=True))
    for p in model_E.parameters():
        p.requires_grad_(False)

    model_sy = IntegralMonotoneSigmaY(sigma_y0_fixed=SIGMA_Y0, hidden_size=32,
                                      p_scale=100.0, n_quad=16, stress_scale=20.0).double()
    model_sy.load_state_dict(torch.load(SY_MODEL, weights_only=True))
    for p in model_sy.parameters():
        p.requires_grad_(False)
    model_Hp = IntegralHTangent(model_sy).double()

    ml_ops = {"E": ml_operator(model_E, function_space=Q),
              "sy": ml_operator(model_sy, function_space=Q),
              "Hp": ml_operator(model_Hp, function_space=Q)}
    ml_params = {"E": Function(R).assign(0.0), "sy": Function(R).assign(0.0), "Hp": Function(R).assign(0.0)}

    disp_gt, stress_gt, p_gt = torque_solver(mesh)
    disp_ml, stress_ml, p_ml = torque_solver(mesh, ml_ops, ml_params)

    u_err = np.linalg.norm(disp_ml[-1].dat.data - disp_gt[-1].dat.data, axis=1).max()
    vm_err = np.abs(von_mises(stress_ml[-1], mesh).dat.data - von_mises(stress_gt[-1], mesh).dat.data).max()
    p_err = np.abs(p_ml[-1].dat.data - p_gt[-1].dat.data).max()
    print(f"max abs displacement error   {u_err:.4e}")
    print(f"max abs von Mises error      {vm_err:.4e}")
    print(f"max abs plastic strain error {p_err:.4e}")

    with CheckpointFile(os.path.join(OUTPUT, "solution_fields.h5"), "w") as f:
        f.save_function(disp_gt[-1], name="displacement_gt")
        f.save_function(stress_gt[-1], name="stress_gt")
        f.save_function(p_gt[-1], name="plastic_strain_gt")
        f.save_function(disp_ml[-1], name="displacement_ml")
        f.save_function(stress_ml[-1], name="stress_ml")
        f.save_function(p_ml[-1], name="plastic_strain_ml")


if __name__ == "__main__":
    main()
