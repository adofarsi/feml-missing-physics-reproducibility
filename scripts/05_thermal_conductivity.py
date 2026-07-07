import os
import numpy as np
import torch
from firedrake import *
from firedrake.ml.pytorch import ml_operator, fem_operator, to_torch
from firedrake.__future__ import interpolate
from firedrake.adjoint import (annotate_tape, continue_annotation, stop_annotating,
                               set_working_tape, get_working_tape, Control, ReducedFunctional)

from operators import TorchModelForK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "outputs", "thermal")
MESH = os.path.join(ROOT, "data", "thermal_3d.msh")

DT, T_TOTAL, T_INITIAL = 20.0, 240.0, 298.15
SOURCE_TRAIN = (600.0, 1.0)
SOURCE_TEST = (500.0, 1.1)
EPOCHS = 100


def heat_mesh():
    mesh = Mesh(MESH)
    x = SpatialCoordinate(mesh)
    S = FunctionSpace(mesh, "DG", 0)
    sample = Function(S).interpolate(conditional(x[2] >= 0.00401, 1, 0))
    base = Function(S).interpolate(conditional(x[2] < 0.00401, 1, 0))
    return RelabeledMesh(mesh, [base, sample], [1, 2])


def heat_solver(source, mesh, ml_op=None, ml_parameters=None):
    max_temperature, frequency = source
    V = FunctionSpace(mesh, "CG", 1)
    const = lambda a: Function(FunctionSpace(mesh, "R", 0)).assign(a)
    T, v = Function(V, name="Temperature"), TestFunction(V)
    x = SpatialCoordinate(mesh)

    dt, t = const(DT), const(0.0)
    rhob, cpb, kb = const(8.9e3), const(0.4e3), const(0.4e3)
    rhos, cps = const(2.7e3), const(0.8e3)
    kr, Tr, beta, delta = const(2.0), const(298.0), const(1.0), const(0.62)

    def conductivity(assemble_ml=False):
        if ml_op is None:
            k = interpolate(kr * (1 + beta * (T - Tr) / Tr) ** (-delta), V)
        else:
            k = ml_op(T, ml_parameters)
        return assemble(k) if assemble_ml else k

    T_0 = Function(V).assign(T_INITIAL)
    T.assign(T_INITIAL)

    def form(assemble_ml=False):
        source_location = conditional(x[0] <= -0.02, 1, 0)
        source_temperature = T_INITIAL + (t / T_TOTAL) * max_temperature * abs(sin(frequency * 2 * pi * t / 90))
        return ((rhob * cpb / dt) * inner(T - T_0, v) * dx(1)
                + (rhos * cps / dt) * inner(T - T_0, v) * dx(2)
                + kb * inner(grad(T), grad(v)) * dx(1)
                + conductivity(assemble_ml) * inner(grad(T), grad(v)) * dx(2)
                + source_location * 1e6 * (T - source_temperature) * v * ds)

    temperatures = []
    tt = 0.0
    while tt < T_TOTAL:
        tt += float(dt)
        t.assign(tt)
        T_0.assign(T)
        problem = NonlinearVariationalProblem(form(False), T, J=derivative(form(True), T))
        NonlinearVariationalSolver(problem).solve()
        temperatures.append(T.copy(deepcopy=True))
    return temperatures


def sample_norm(field):
    return assemble(inner(field, field) * dx(2)) ** 0.5


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    get_working_tape().clear_tape()
    np.random.seed(0)
    torch.manual_seed(0)

    with stop_annotating():
        mesh = heat_mesh()

    if not annotate_tape():
        continue_annotation()

    model = TorchModelForK().double()
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    ml_op = ml_operator(model, function_space=V)
    ml_parameters = Function(R).assign(0.0)
    ml_parameters_t = to_torch(ml_parameters, requires_grad=True)

    with stop_annotating():
        obs = heat_solver(SOURCE_TRAIN, mesh)
        for T in obs:
            T.dat.data[:] += T.dat.data * np.random.normal(0, 0.02, size=T.dat.data.shape)
        test = heat_solver(SOURCE_TEST, mesh)
        for T in test:
            T.dat.data[:] += T.dat.data * np.random.normal(0, 0.02, size=T.dat.data.shape)

    def loss(parameters):
        pred = heat_solver(SOURCE_TRAIN, mesh, ml_op, parameters)
        return sum(sample_norm(pred[i] - obs[i]) / sample_norm(obs[i]) for i in range(len(pred))) / len(pred)

    def test_error(parameters):
        pred = heat_solver(SOURCE_TEST, mesh, ml_op, parameters)
        return sum(sample_norm(pred[i] - test[i]) / sample_norm(test[i]) for i in range(len(pred))) / len(pred)

    with set_working_tape():
        fem_op = fem_operator(ReducedFunctional(loss(ml_parameters), [Control(ml_parameters)]))

    optimiser = torch.optim.Adam(model.parameters(), lr=0.025)

    losses, errors = [], []
    for epoch in range(EPOCHS):
        model.train()
        model.zero_grad()
        value = fem_op(ml_parameters_t)
        value.backward()
        optimiser.step()
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
