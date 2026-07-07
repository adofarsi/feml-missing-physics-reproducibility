import os

from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.ml.pytorch import ml_operator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHEAR_CONFIG = {
    "name": "shear",
    "mesh_path": os.path.join(ROOT, "data", "shear_coupon_2d.msh"),
    "thickness": 3.100,
    "left": (7, 8, 12),
    "right": (10, 9, 11),
    "load_dir": (0.0, 1.0),
}


def build_hardening_ml_ops(mesh, model_sy, model_Hp):
    Q = FunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    ml_ops = {
        "sy": ml_operator(model_sy, function_space=Q),
        "Hp": ml_operator(model_Hp, function_space=Q),
    }
    ml_params = {
        "sy": Function(R, name="ml_params_sy").assign(0.0),
        "Hp": Function(R, name="ml_params_Hp").assign(0.0),
    }
    return ml_ops, ml_params


def build_nonlocal_nn_damage_ml_ops(mesh, model_sy, model_Hp, model_phi):
    Q = FunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    ml_ops = {
        "sy": ml_operator(model_sy, function_space=Q),
        "Hp": ml_operator(model_Hp, function_space=Q),
        "phi": ml_operator(model_phi, function_space=Q),
    }
    ml_params = {
        "sy": Function(R, name="ml_params_sy").assign(0.0),
        "Hp": Function(R, name="ml_params_Hp").assign(0.0),
        "phi": Function(R, name="ml_params_phi").assign(0.0),
    }
    return ml_ops, ml_params


_SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

_FCP = {"quadrature_degree": 4}


def _kinematics(mesh, E_value, nu_value):
    I2d = Identity(2)
    tiny = Constant(1.0e-14)
    E = Constant(E_value)
    nu = Constant(nu_value)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def def_grad(w_disp):
        return I2d + grad(w_disp)

    def epsilon(w_disp):
        Fdef = def_grad(w_disp)
        return 0.5 * (Fdef.T * Fdef - I2d)

    def trial_deviatoric(eps_total, e33_expr, eps_p_n):
        eps_el_tr = eps_total - eps_p_n
        eps_el_tr_33 = e33_expr + tr(eps_p_n)
        tr_el_3d = tr(eps_total) + e33_expr
        sigma_tr = lmbda * tr_el_3d * I2d + 2.0 * mu * eps_el_tr
        sigma_tr_33 = lmbda * tr_el_3d + 2.0 * mu * eps_el_tr_33
        p_hyd = (tr(sigma_tr) + sigma_tr_33) / 3.0
        return sigma_tr - p_hyd * I2d, sigma_tr_33 - p_hyd, p_hyd

    def radial_return(eps_total, e33_expr, eps_p_n, p_n, sy, Hp):
        s_tr, s_tr_33, p_hyd = trial_deviatoric(eps_total, e33_expr, eps_p_n)
        J2_tr = 0.5 * (inner(s_tr, s_tr) + s_tr_33**2)
        seq = sqrt(3.0 * J2_tr + tiny)
        f_tr = seq - sy
        dg = conditional(gt(f_tr, 0.0), f_tr / (3.0 * mu + Hp), 0.0)
        beta = conditional(gt(f_tr, 0.0), 1.0 - 3.0 * mu * dg / seq, 1.0)
        sigma_new = beta * s_tr + p_hyd * I2d
        sigma_new_33 = beta * s_tr_33 + p_hyd
        eps_p_new_val = eps_p_n + dg * (1.5 * s_tr / seq)
        p_new_val = p_n + dg
        return sigma_new, sigma_new_33, eps_p_new_val, p_new_val

    return def_grad, epsilon, radial_return


def _bc_terms(cfg, u, v, applied_displacement, penalty):
    load_dir = as_vector([Constant(cfg["load_dir"][0]), Constant(cfg["load_dir"][1])])
    u_loaded = as_vector([
        applied_displacement * Constant(cfg["load_dir"][0]),
        applied_displacement * Constant(cfg["load_dir"][1]),
    ])
    res = 0
    for label in cfg["left"]:
        res += penalty * dot(u, v) * ds(label)
    for label in cfg["right"]:
        res += penalty * dot(u - u_loaded, v) * ds(label)
    return res, load_dir, u_loaded


def plane_stress_svk_solver(displacement_levels, mesh, ml_ops, ml_params, cfg,
                            E_value, nu_value, penalty_value):
    penalty = Constant(penalty_value)
    def_grad, epsilon, radial_return = _kinematics(mesh, E_value, nu_value)

    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    T = TensorFunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    W = V * Q
    w = Function(W, name="displacement_e33")
    u, e33 = split(w)
    v, q33 = TestFunctions(W)

    eps_p_old = Function(T, name="eps_p_old")
    p_old = Function(Q, name="p_old")
    eps_p_new = Function(T, name="eps_p_new")
    p_new = Function(Q, name="p_new")
    eps_dg = Function(T, name="eps_dg")
    u_v = Function(V, name="displacement_view")
    e33_q = Function(Q, name="e33_view")
    applied_displacement = Function(R, name="applied_displacement").assign(0.0)
    bc_res, load_dir, u_loaded = _bc_terms(cfg, u, v, applied_displacement, penalty)

    def hardening_fields(assembled_ml_op=False):
        p_input = interpolate(p_old, Q)
        if assembled_ml_op:
            return assemble(ml_ops["sy"](p_input, ml_params["sy"])), \
                   assemble(ml_ops["Hp"](p_input, ml_params["Hp"]))
        return ml_ops["sy"](p_input, ml_params["sy"]), ml_ops["Hp"](p_input, ml_params["Hp"])

    def sigma_stress(w_disp, e33_expr, assembled_ml_op=False):
        sy, Hp = hardening_fields(assembled_ml_op)
        sigma, sigma_33, _, _ = radial_return(epsilon(w_disp), e33_expr, eps_p_old, p_old, sy, Hp)
        return sigma, sigma_33

    def residual(assembled_ml_op=False):
        sigma, sigma_33 = sigma_stress(u, e33, assembled_ml_op)
        res = inner(def_grad(u) * sigma, grad(v)) * dx
        res += sigma_33 * q33 * dx
        return res + bc_res

    def reaction_force():
        reaction = 0.0
        for label in cfg["right"]:
            reaction += assemble(-penalty * dot(u - u_loaded, load_dir) * ds(label))
        return reaction * (cfg["thickness"] / 1000.0)

    displacements, loads, plastic_strains = [], [], []
    for step, level in enumerate(displacement_levels, start=1):
        print(f"[{cfg['name']}] Step {step}/{len(displacement_levels)}: u = {level:.6f} mm")
        applied_displacement.assign(float(level))
        J = derivative(residual(assembled_ml_op=True), w)
        problem = NonlinearVariationalProblem(residual(assembled_ml_op=False), w, J=J,
                                              form_compiler_parameters=_FCP)
        NonlinearVariationalSolver(problem, solver_parameters=_SOLVER_PARAMETERS).solve()

        load_value = reaction_force()
        print(f"  load = {float(load_value):.6f} kN")

        sy_a, Hp_a = hardening_fields(assembled_ml_op=True)
        u_v.assign(w.sub(0))
        e33_q.assign(w.sub(1))
        eps_dg.interpolate(epsilon(u_v))
        _, _, ep_expr, p_expr = radial_return(eps_dg, e33_q, eps_p_old, p_old, sy_a, Hp_a)
        eps_p_new.interpolate(ep_expr)
        p_new.interpolate(p_expr)
        eps_p_old.assign(eps_p_new)
        p_old.assign(p_new)
        print(f"  max accumulated plastic strain = {float(p_old.dat.data_ro.max()):.6e}")

        displacements.append(w.copy(deepcopy=True))
        loads.append(load_value)
        plastic_strains.append(p_old.copy(deepcopy=True))

    return displacements, loads, plastic_strains


def plane_stress_svk_nonlocal_nn_damage_solver(displacement_levels, mesh, ml_ops, ml_params, cfg,
                                               E_value, nu_value, penalty_value,
                                               ell, deg_s=2.0, d_max=0.95):
    penalty = Constant(penalty_value)
    deg_c, ell2 = Constant(deg_s), Constant(ell**2)
    def_grad, epsilon, radial_return = _kinematics(mesh, E_value, nu_value)

    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    T = TensorFunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    Sn = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    w = Function(W, name="displacement_e33")
    u, e33 = split(w)
    v, q33 = TestFunctions(W)

    eps_p_old = Function(T)
    p_old = Function(Q)
    eps_p_new = Function(T)
    p_new = Function(Q)
    eps_dg = Function(T)
    u_v = Function(V)
    e33_q = Function(Q)
    Dbar = Function(Sn, name="nonlocal_damage")
    applied_displacement = Function(R).assign(0.0)
    bc_res, load_dir, u_loaded = _bc_terms(cfg, u, v, applied_displacement, penalty)

    Phibar = Function(Sn)
    phi_t, phi_q = TrialFunction(Sn), TestFunction(Sn)
    a_helm = phi_t * phi_q * dx + ell2 * dot(grad(phi_t), grad(phi_q)) * dx

    def hardening_fields(assembled_ml_op=False):
        p_input = interpolate(p_old, Q)
        if assembled_ml_op:
            return assemble(ml_ops["sy"](p_input, ml_params["sy"])), \
                   assemble(ml_ops["Hp"](p_input, ml_params["Hp"]))
        return ml_ops["sy"](p_input, ml_params["sy"]), ml_ops["Hp"](p_input, ml_params["Hp"])

    def sigma_stress(w_disp, e33_expr, assembled_ml_op=False):
        sy, Hp = hardening_fields(assembled_ml_op)
        sigma, sigma_33, _, _ = radial_return(epsilon(w_disp), e33_expr, eps_p_old, p_old, sy, Hp)
        return sigma, sigma_33

    def residual(assembled_ml_op=False):
        gD = (1.0 - Dbar) ** deg_c + Constant(1.0e-3)
        sigma, sigma_33 = sigma_stress(u, e33, assembled_ml_op)
        res = inner(gD * def_grad(u) * sigma, grad(v)) * dx
        res += gD * sigma_33 * q33 * dx
        return res + bc_res

    def reaction_force():
        reaction = 0.0
        for label in cfg["right"]:
            reaction += assemble(-penalty * dot(u - u_loaded, load_dir) * ds(label))
        return reaction * (cfg["thickness"] / 1000.0)

    displacements, loads, plastic_strains, damages = [], [], [], []
    for level in displacement_levels:
        applied_displacement.assign(float(level))
        J = derivative(residual(assembled_ml_op=True), w)
        problem = NonlinearVariationalProblem(residual(assembled_ml_op=False), w, J=J,
                                              form_compiler_parameters=_FCP)
        NonlinearVariationalSolver(problem, solver_parameters=_SOLVER_PARAMETERS).solve()

        loads.append(reaction_force())

        sy_a, Hp_a = hardening_fields(assembled_ml_op=True)
        u_v.assign(w.sub(0))
        e33_q.assign(w.sub(1))
        eps_dg.interpolate(epsilon(u_v))
        _, _, ep_expr, p_expr = radial_return(eps_dg, e33_q, eps_p_old, p_old, sy_a, Hp_a)
        eps_p_new.interpolate(ep_expr)
        p_new.interpolate(p_expr)

        p_input = interpolate(p_new, Q)
        Phi_loc = assemble(ml_ops["phi"](p_input, ml_params["phi"]))
        L_helm = Phi_loc * phi_q * dx
        solve(a_helm == L_helm, Phibar,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        Dbar.interpolate(min_value(Constant(d_max),
                                   max_value(Constant(0.0), 1.0 - exp(-Phibar))))

        eps_p_old.assign(eps_p_new)
        p_old.assign(p_new)
        displacements.append(w.copy(deepcopy=True))
        plastic_strains.append(p_old.copy(deepcopy=True))
        damages.append(Dbar.copy(deepcopy=True))

    return displacements, loads, plastic_strains, damages
