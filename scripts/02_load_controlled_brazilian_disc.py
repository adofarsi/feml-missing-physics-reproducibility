import os
import numpy as np
import torch
from firedrake import *
from firedrake.ml.pytorch import ml_operator, fem_operator, to_torch
from firedrake.__future__ import interpolate
from firedrake.adjoint import (annotate_tape, continue_annotation, stop_annotating,
                               set_working_tape, get_working_tape, Control, ReducedFunctional)

from operators import TorchModelForE, IntegralMonotoneSigmaY, IntegralHTangent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "outputs", "load_controlled")
E_MODEL = os.path.join(ROOT, "outputs", "displacement_controlled", "model.pth")

NU = 0.3
E_ELASTIC = 145.0
SIGMA_Y0, R_INF, B_VOCE = 2.354, 0.9, 25.0
F_MIN, F_MAX = 0.040, 0.200
N_TRAIN, N_TEST = 15, 8
EPOCHS = 400

TRAIN_LEVELS = list(np.linspace(F_MIN, F_MAX, N_TRAIN))
_STEP = (F_MAX - F_MIN) / (N_TRAIN - 1)
TEST_LEVELS = list(np.linspace(F_MIN + _STEP / 2.0, F_MAX - _STEP / 2.0, N_TEST))


def disc_mesh(radius, divisions, bc_length):
    mesh = UnitSquareMesh(divisions, divisions)
    x, y = SpatialCoordinate(mesh)
    trace = FunctionSpace(mesh, "HDiv Trace", 0)
    top = Function(trace).interpolate(
        conditional(Or(And(x >= 1 - bc_length, y >= 1.), And(x >= 1., y >= 1 - bc_length)), 1., 0.))
    bottom = Function(trace).interpolate(
        conditional(Or(And(x <= bc_length, y <= 0.), And(x <= 0., y <= bc_length)), 1., 0.))
    left = Function(trace).interpolate(
        conditional(Or(And(x <= bc_length, y >= 1), And(x <= 0., y >= 1. - bc_length)), 1., 0.))
    right = Function(trace).interpolate(
        conditional(Or(And(x >= 1 - bc_length, y <= 0), And(x >= 1., y <= bc_length)), 1., 0.))
    mesh = RelabeledMesh(mesh, [left, right, bottom, top], [1, 2, 3, 4])
    mesh.coordinates.interpolate(as_vector([x - .5, y - .5]))
    x, y = SpatialCoordinate(mesh)
    theta = atan2(y, x)
    u_b = as_vector([radius * cos(theta) - x, radius * sin(theta) - y])
    V = VectorFunctionSpace(mesh, "CG", 1)
    uu = Function(V)
    solve(inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx
          == dot(Constant((0.0, 0.0)), TestFunction(V)) * dx,
          uu, bcs=DirichletBC(V, u_b, "on_boundary"))
    mesh.coordinates.interpolate(SpatialCoordinate(mesh) + uu)
    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x * cos(pi / 4) - y * sin(pi / 4),
                                            x * sin(pi / 4) + y * cos(pi / 4)]))
    return mesh


def disc_solver(mesh, levels, ml_op=None, ml_parameters=None, ml_op_E=None, ml_params_E=None):
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    T = TensorFunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    top, bottom = 4, 3
    u, v = Function(V, name="Displacement"), TestFunction(V)

    eps_p_old, p_old = Function(T), Function(Q)
    eps_p_new, p_new, eps_dg = Function(T), Function(Q), Function(T)

    I2 = Identity(2)
    nu = Constant(NU)
    tiny = Constant(1.0e-14)

    def epsilon(w):
        return 0.5 * (grad(w) + grad(w).T)

    E_field = Function(Q).assign(Constant(E_ELASTIC))
    lmbda = E_field * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_field / (2.0 * (1.0 + nu))

    def update_E():
        if ml_op_E is not None:
            E_field.assign(assemble(ml_op_E(interpolate(tr(epsilon(u)), Q), ml_params_E)))
        else:
            kappa = max_value(-tr(epsilon(u)), Constant(0.0))
            E_field.interpolate(Constant(57.3) + Constant(87.7) * exp(-kappa / Constant(0.008)))

    def trial_deviatoric(eps_total, eps_p_n):
        eps_el = eps_total - eps_p_n
        eps_el_33 = tr(eps_p_n)
        tr_3d = tr(eps_total)
        sigma_tr = lmbda * tr_3d * I2 + 2.0 * mu * eps_el
        sigma_tr_33 = lmbda * tr_3d + 2.0 * mu * eps_el_33
        p_hyd = (tr(sigma_tr) + sigma_tr_33) / 3.0
        s = sigma_tr - p_hyd * I2
        return s, -tr(s), p_hyd

    def radial_return(eps_total, eps_p_n, p_n, sy_field=None, Hp_field=None):
        s, s33, ph = trial_deviatoric(eps_total, eps_p_n)
        seq = sqrt(1.5 * (inner(s, s) + s33 ** 2) + tiny)
        if sy_field is None:
            pv = variable(p_n)
            sy = Constant(SIGMA_Y0) + Constant(R_INF) * (1.0 - exp(-Constant(B_VOCE) * pv))
            Hp = diff(sy, pv)
        else:
            sy, Hp = sy_field, Hp_field
        ft = seq - sy
        dg = conditional(gt(ft, 0.0), ft / (3.0 * mu + Hp), 0.0)
        bt = conditional(gt(ft, 0.0), 1.0 - 3.0 * mu * dg / seq, 1.0)
        sigma_new = bt * s + ph * I2
        return sigma_new, eps_p_n + dg * (1.5 * s / seq), p_n + dg

    def stress(assemble_ml=False, assemble_stress=False):
        if ml_op is None:
            sig, _, _ = radial_return(epsilon(u), eps_p_old, p_old)
        else:
            op_sy, op_Hp = ml_op
            par_sy, par_Hp = ml_parameters
            p_in = interpolate(p_old, Q)
            if assemble_ml:
                sy, Hp = assemble(op_sy(p_in, par_sy)), assemble(op_Hp(p_in, par_Hp))
            else:
                sy, Hp = op_sy(p_in, par_sy), op_Hp(p_in, par_Hp)
            sig, _, _ = radial_return(epsilon(u), eps_p_old, p_old, sy, Hp)
        return Function(T).interpolate(sig) if assemble_stress else sig

    f_i = Function(R).assign(0.0)

    def form(assemble_ml=False):
        sgm_t, sgm_b = .0075, .0025
        return (inner(stress(assemble_ml), epsilon(v)) * dx
                + (f_i / (sgm_t * sqrt(2 * pi))) * exp(-x ** 2 / (2 * sgm_t ** 2)) * v[1] * ds(top)
                + 1e15 * exp(-x ** 2 / (2 * sgm_b ** 2)) * u[1] * v[1] * ds(bottom)
                + 1e15 * u[0] * v[0] * ds(bottom))

    modes = VectorSpaceBasis([Function(V).interpolate(Constant([1, 0])),
                              Function(V).interpolate(Constant([0, 1])),
                              Function(V).interpolate(as_vector([-y, x]))])
    parameters = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
                  "snes_rtol": 1e-8, "snes_atol": 1e-8, "snes_max_it": 200,
                  "ksp_type": "preonly", "pc_type": "lu"}

    displacements, loads, stresses = [], [], []
    for level in levels:
        f_i.assign(level)
        update_E()
        problem = NonlinearVariationalProblem(form(False), u, J=derivative(form(True), u))
        NonlinearVariationalSolver(problem, near_nullspace=modes, solver_parameters=parameters).solve()
        loads.append(f_i.copy(deepcopy=True))
        displacements.append(u.copy(deepcopy=True))
        stresses.append(stress(assemble_ml=True, assemble_stress=True).copy(deepcopy=True))
        update_E()
        eps_dg.interpolate(epsilon(u))
        if ml_op is None:
            _, ep, pp = radial_return(eps_dg, eps_p_old, p_old)
        else:
            op_sy, op_Hp = ml_op
            par_sy, par_Hp = ml_parameters
            sy = assemble(op_sy(interpolate(p_old, Q), par_sy))
            Hp = assemble(op_Hp(interpolate(p_old, Q), par_Hp))
            _, ep, pp = radial_return(eps_dg, eps_p_old, p_old, sy, Hp)
        eps_p_new.interpolate(ep)
        p_new.interpolate(pp)
        eps_p_old.assign(eps_p_new)
        p_old.assign(p_new)
    return displacements, loads, stresses


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    get_working_tape().clear_tape()
    np.random.seed(0)
    torch.manual_seed(0)

    with stop_annotating():
        mesh = disc_mesh(0.05, 20, 0.5)

    if not annotate_tape():
        continue_annotation()

    model_sy = IntegralMonotoneSigmaY(sigma_y0_fixed=SIGMA_Y0, hidden_size=32,
                                      p_scale=100.0, n_quad=16, stress_scale=20.0).double()
    with torch.no_grad():
        model_sy.fc2.bias.fill_(-8.0)
    Q = FunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    ml_op_sy = ml_operator(model_sy, function_space=Q)
    ml_par_sy = Function(R).assign(0.0)
    ml_par_sy_t = to_torch(ml_par_sy, requires_grad=True)

    model_Hp = IntegralHTangent(model_sy).double()
    ml_op_Hp = ml_operator(model_Hp, function_space=Q)
    ml_par_Hp = Function(R).assign(0.0)
    ml_par_Hp_t = to_torch(ml_par_Hp, requires_grad=True)

    model_E = TorchModelForE(lower_bound_stiffness=20.0).double()
    model_E.load_state_dict(torch.load(E_MODEL, weights_only=True))
    model_E.eval()
    for p in model_E.parameters():
        p.requires_grad_(False)
    ml_op_E = ml_operator(model_E, function_space=Q)
    ml_par_E = Function(R).assign(0.0)

    ml_op = [ml_op_sy, ml_op_Hp]
    ml_par = [ml_par_sy, ml_par_Hp]

    with stop_annotating():
        obs, _, _ = disc_solver(mesh, TRAIN_LEVELS)
        for u in obs:
            u.dat.data[:] += u.dat.data * np.random.normal(0, 0.01, size=u.dat.data.shape)
        test, _, _ = disc_solver(mesh, TEST_LEVELS)
        for u in test:
            u.dat.data[:] += u.dat.data * np.random.normal(0, 0.01, size=u.dat.data.shape)

    def loss(par):
        pred, _, _ = disc_solver(mesh, TRAIN_LEVELS, ml_op, par, ml_op_E, ml_par_E)
        return sum(norm(pred[i] - obs[i]) / norm(obs[i]) for i in range(N_TRAIN)) / N_TRAIN

    def test_error(par):
        pred, _, _ = disc_solver(mesh, TEST_LEVELS, ml_op, par, ml_op_E, ml_par_E)
        return sum(norm(pred[i] - test[i]) / norm(test[i]) for i in range(N_TEST)) / N_TEST

    with set_working_tape():
        rf = ReducedFunctional(loss(ml_par), [Control(ml_par_sy), Control(ml_par_Hp)])
        fem_op = fem_operator(rf)

    optimiser = torch.optim.Adam(model_sy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.05, total_steps=EPOCHS, pct_start=0.3)

    losses, errors = [], []
    for epoch in range(EPOCHS):
        model_sy.train()
        model_sy.zero_grad()
        value = fem_op(ml_par_sy_t, ml_par_Hp_t)
        value.backward()
        optimiser.step()
        scheduler.step()
        with stop_annotating():
            error = test_error(ml_par)
        losses.append(value.item())
        errors.append(float(error))
        print(f"epoch {epoch + 1}  loss {value.item():.4e}  test {float(error):.4e}")

    np.savetxt(os.path.join(OUTPUT, "losses.dat"), losses)
    np.savetxt(os.path.join(OUTPUT, "errors.dat"), errors)
    torch.save(model_sy.state_dict(), os.path.join(OUTPUT, "model.pth"))


if __name__ == "__main__":
    main()
