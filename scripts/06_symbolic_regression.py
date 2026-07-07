import os
import numpy as np
import torch
from pysr import PySRRegressor

from operators import TorchModelForK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "outputs", "thermal_symbolic")
K_MODEL = os.path.join(ROOT, "outputs", "thermal", "model.pth")


def ground_truth(T):
    return 2.0 * ((1.0 + (T - 298.0) / 298.0) ** (-0.62))


def r2(pred, true):
    return 1.0 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    model = TorchModelForK().double()
    model.load_state_dict(torch.load(K_MODEL, weights_only=True))
    model.eval()

    T_fit = np.linspace(400, 700, 500)
    with torch.no_grad():
        k_fit = model(torch.tensor(T_fit, dtype=torch.float64)).numpy()

    regressor = PySRRegressor(
        niterations=60,
        populations=30,
        population_size=50,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["square", "inv", "log", "exp", "sqrt"],
        maxsize=25,
        maxdepth=8,
        parsimony=0.003,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=100.0,
        ncycles_per_iteration=550,
        turbo=True,
        bumper=True,
        loss="loss(prediction, target) = (prediction - target)^2 + 1e6 * (prediction < 0 ? prediction^2 : 0)",
        temp_equation_file=True,
        tempdir=os.path.join(OUTPUT, "pysr_temp"),
        random_state=42,
        deterministic=True,
        procs=0,
        multithreading=False,
    )
    regressor.fit(T_fit.reshape(-1, 1), k_fit, variable_names=["T"])

    equations = regressor.equations_
    T_check = np.linspace(1, 1500, 2000).reshape(-1, 1)
    positive = []
    for idx in equations.index:
        try:
            preds = regressor.predict(T_check, index=idx)
            positive.append(bool(np.all(np.isfinite(preds)) and np.all(preds > 0)))
        except Exception:
            positive.append(False)
    positive = np.array(positive)

    if positive.any():
        best_idx = equations.iloc[np.where(positive)[0]]["score"].idxmax()
    else:
        best_idx = equations.index[equations["equation"] == regressor.get_best().equation][0]

    expression = equations.loc[best_idx, "equation"]
    complexity = int(equations.loc[best_idx, "complexity"])

    T_extra = np.linspace(700, 900, 200)
    k_sr_fit = regressor.predict(T_fit.reshape(-1, 1), index=best_idx)
    k_sr_extra = regressor.predict(T_extra.reshape(-1, 1), index=best_idx)

    r2_fit = r2(k_sr_fit, ground_truth(T_fit))
    r2_extra = r2(k_sr_extra, ground_truth(T_extra))

    print(f"selected expression (complexity {complexity}): {expression}")
    print(f"R2 vs ground truth, fitting range [400,700] K:   {r2_fit:.4f}")
    print(f"R2 vs ground truth, extrapolation [700,900] K:   {r2_extra:.4f}")

    with open(os.path.join(OUTPUT, "summary.txt"), "w") as f:
        f.write(f"expression: {expression}\n")
        f.write(f"complexity: {complexity}\n")
        f.write(f"R2_fit: {r2_fit:.6f}\n")
        f.write(f"R2_extrapolation: {r2_extra:.6f}\n")


if __name__ == "__main__":
    main()
