import os
import numpy as np
import torch
from firedrake import *
from firedrake.ml.pytorch import ml_operator, fem_operator, to_torch
from firedrake.__future__ import interpolate
from firedrake.adjoint import (annotate_tape, continue_annotation, stop_annotating,
                               set_working_tape, get_working_tape, Control, ReducedFunctional)

from operators import TorchModelForE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "outputs", "displacement_controlled")

DEGREE = 3
NU = 0.3
TRAIN_LEVELS = [0.0001, 0.0006, 0.0012, 0.0018, 0.0024, 0.003]
TEST_LEVELS = [0.0007, 0.0014, 0.0021, 0.0028]
EPOCHS = 200


def uniaxial_solver(levels, mesh, ml_op=None, ml_parameters=None):
    x, y = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", DEGREE)
    R = FunctionSpace(mesh, "R", 0)
    D = FunctionSpace(mesh, "DG", DEGREE - 1)
    T = TensorFunctionSpace(mesh, "DG", DEGREE - 1)
    top, bottom = 4, 3
    u, v = Function(V, name="Displacement"), TestFunction(V)

    def epsilon(w):
        return 0.5 * (grad(w) + grad(w).T)

    def modulus(assemble_ml=False):
        if ml_op is None:
            kappa = max_value(-tr(epsilon(u)), 0.0)
            E = interpolate(Constant(57.3) + Constant(87.7) * exp(-kappa / Constant(0.008)), D)
        else:
            E = ml_op(interpolate(tr(epsilon(u)), D), ml_parameters)
        return assemble(E) if assemble_ml else E

    def stress(assemble_ml=False, assemble_stress=False):
        E = modulus(assemble_ml)
        mu = E / (2.0 * (1.0 + NU))
        lmbda = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        s = lmbda * tr(epsilon(u)) * Identity(2) + 2.0 * mu * epsilon(u)
        return Function(T).interpolate(s) if assemble_stress else s

    u_top = Function(R).assign(0.0)

    def form(assemble_ml=False):
        return (inner(stress(assemble_ml), epsilon(v)) * dx
                + 1e15 * (u[1] - u_top) * v[1] * ds(top)
                + 1e15 * u[0] * v[0] * ds(top)
                + 1e15 * u[1] * v[1] * ds(bottom)
                + 1e15 * u[0] * v[0] * ds(bottom))

    def reaction():
        s = stress(assemble_ml=True, assemble_stress=True)
        return dot(dot(s, n), n) * ds(top)

    modes = VectorSpaceBasis([Function(V).interpolate(Constant([1, 0])),
                              Function(V).interpolate(Constant([0, 1])),
                              Function(V).interpolate(as_vector([-y, x]))])

    displacements, loads, stresses = [], [], []
    for level in levels:
        u_top.assign(-level)
        problem = NonlinearVariationalProblem(form(False), u, J=derivative(form(True), u))
        NonlinearVariationalSolver(problem, near_nullspace=modes).solve()
        loads.append(Function(R).assign(assemble(reaction())))
        displacements.append(u.copy(deepcopy=True))
        stresses.append(stress(assemble_ml=True, assemble_stress=True).copy(deepcopy=True))
    return displacements, loads, stresses


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    get_working_tape().clear_tape()
    np.random.seed(0)

    mesh = RectangleMesh(6, 6, 0.05, 0.1)
    D = FunctionSpace(mesh, "DG", DEGREE - 1)
    R = FunctionSpace(mesh, "R", 0)

    if not annotate_tape():
        continue_annotation()

    model = TorchModelForE(lower_bound_stiffness=20.0).double()
    with torch.no_grad():
        model.mlp[4].weight.normal_(0.0, 0.001)
        model.mlp[4].bias.fill_(-3.0)
    ml_op = ml_operator(model, function_space=D)
    ml_parameters = Function(R).assign(0.0)
    ml_parameters_t = to_torch(ml_parameters, requires_grad=True)

    with stop_annotating():
        _, force_obs, _ = uniaxial_solver(TRAIN_LEVELS, mesh)
        noise = np.random.normal(0.0, 0.01, size=len(force_obs))
        force_obs = [f * (1.0 + e) for f, e in zip(force_obs, noise)]
        _, force_test, _ = uniaxial_solver(TEST_LEVELS, mesh)

    def loss(parameters):
        _, force, _ = uniaxial_solver(TRAIN_LEVELS, mesh, ml_op, parameters)
        return sum(norm(force[i] - force_obs[i]) / norm(force_obs[i])
                   for i in range(len(TRAIN_LEVELS))) / len(TRAIN_LEVELS)

    def test_error(parameters):
        _, force, _ = uniaxial_solver(TEST_LEVELS, mesh, ml_op, parameters)
        return sum(norm(force[i] - force_test[i]) / norm(force_test[i])
                   for i in range(len(TEST_LEVELS))) / len(TEST_LEVELS)

    with set_working_tape():
        fem_op = fem_operator(ReducedFunctional(loss(ml_parameters), [Control(ml_parameters)]))

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.03, total_steps=EPOCHS, pct_start=0.3)

    losses, errors = [], []
    for epoch in range(EPOCHS):
        model.train()
        model.zero_grad()
        value = fem_op(ml_parameters_t)
        value.backward()
        optimiser.step()
        scheduler.step()
        with stop_annotating():
            error = test_error(ml_parameters)
        losses.append(value.item())
        errors.append(float(error))
        print(f"epoch {epoch + 1}  loss {value.item():.4e}  test {float(error):.4e}")

    np.savetxt(os.path.join(OUTPUT, "losses.dat"), losses)
    np.savetxt(os.path.join(OUTPUT, "errors.dat"), errors)
    torch.save(model.state_dict(), os.path.join(OUTPUT, "model.pth"))


if __name__ == "__main__":
    main()
