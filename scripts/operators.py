import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchModelForE(nn.Module):
    def __init__(self, lower_bound_stiffness=30.0, random_seed=42):
        super().__init__()
        self.lower_bound_stiffness = lower_bound_stiffness
        torch.manual_seed(random_seed)
        self.mlp = nn.Sequential(
            nn.Linear(1, 30), nn.SiLU(),
            nn.Linear(30, 30), nn.SiLU(),
            nn.Linear(30, 1), nn.Softplus(),
        )

    def forward(self, I1):
        kappa = torch.relu(-I1.view(-1, 1))
        return (self.lower_bound_stiffness * (1.0 + self.mlp(1e2 * kappa))).flatten()


class IntegralMonotoneSigmaY(nn.Module):
    def __init__(self, sigma_y0_init=250.0, hidden_size=16, p_scale=100.0, n_quad=16,
                 stress_scale=1.0, sigma_y0_fixed=None):
        super().__init__()
        self._sigma_y0_is_fixed = sigma_y0_fixed is not None
        if self._sigma_y0_is_fixed:
            self.register_buffer("_sigma_y0_const",
                                 torch.tensor(float(sigma_y0_fixed), dtype=torch.float64))
        else:
            self.register_buffer("_sy0_scale",
                                 torch.tensor(sigma_y0_init / np.log(2.0), dtype=torch.float64))
            self._raw_sigma_y0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("p_scale", torch.tensor(p_scale, dtype=torch.float64))
        self.register_buffer("stress_scale", torch.tensor(stress_scale, dtype=torch.float64))
        self.fc1 = nn.Linear(1, hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size, 1, dtype=torch.float64)
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self.register_buffer("gl_nodes", torch.tensor((nodes + 1.0) / 2.0, dtype=torch.float64))
        self.register_buffer("gl_weights", torch.tensor(weights / 2.0, dtype=torch.float64))

    @property
    def sigma_y0(self):
        if self._sigma_y0_is_fixed:
            return self._sigma_y0_const
        return self._sy0_scale * F.softplus(self._raw_sigma_y0)

    def _Hp_raw(self, p_scaled):
        x = F.softplus(self.fc1(p_scaled))
        return self.stress_scale * F.softplus(self.fc2(x))

    def forward(self, p_flat):
        p = p_flat.view(-1, 1)
        s = p * self.gl_nodes.view(1, -1)
        n, q = s.shape
        h = self._Hp_raw((s * self.p_scale).reshape(-1, 1)).view(n, q)
        integral = p * (self.gl_weights.unsqueeze(0) * h).sum(dim=1, keepdim=True)
        return (self.sigma_y0 + integral).flatten()


class IntegralHTangent(nn.Module):
    def __init__(self, sigma_y_model):
        super().__init__()
        self.sigma_y_model = sigma_y_model

    def forward(self, p_flat):
        p_scaled = p_flat.view(-1, 1) * self.sigma_y_model.p_scale
        return self.sigma_y_model._Hp_raw(p_scaled).flatten()


class TorchModelForK(nn.Module):
    def __init__(self, k_min=1.0, k_max=2.0, random_seed=42):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        torch.manual_seed(random_seed)
        self.shape_net = nn.Sequential(
            nn.Linear(1, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 1), nn.Sigmoid(),
        )

    def forward(self, T):
        return (self.k_min + self.shape_net(T.view(-1, 1) / 100) * self.k_max).flatten()


def inverse_softplus(value):
    return float(np.log(np.expm1(float(value))))


class MonotoneReLUHazard(nn.Module):
    def __init__(self, hidden=6, onset_init=0.11, p_max=0.65, slope_init=1.0, power=2):
        super().__init__()
        self.hidden = int(hidden)
        self.power = int(power)
        b0 = np.linspace(onset_init, p_max, self.hidden)
        self._raw_b = nn.Parameter(torch.tensor(
            [inverse_softplus(max(float(b), 1e-3)) for b in b0], dtype=torch.float64))
        self._raw_s = nn.Parameter(torch.tensor(
            [inverse_softplus(slope_init)] * self.hidden, dtype=torch.float64))

    @property
    def breakpoints(self):
        return F.softplus(self._raw_b)

    @property
    def slopes(self):
        return F.softplus(self._raw_s)

    def forward(self, p_flat):
        k = self.power
        p = p_flat.view(-1, 1)
        d = torch.clamp(p - self.breakpoints.view(1, -1), min=0.0)
        return (self.slopes.view(1, -1) / (k + 1.0) * d ** (k + 1)).sum(dim=1).flatten()

    def rate(self, p_flat):
        k = self.power
        p = p_flat.view(-1, 1)
        d = torch.clamp(p - self.breakpoints.view(1, -1), min=0.0)
        return (self.slopes.view(1, -1) * d ** k).sum(dim=1).flatten()

    def scalar_parameters(self):
        return [self._raw_b, self._raw_s]

    def describe(self):
        return (f"power={self.power}, onset={float(self.breakpoints.min()):.3g}, "
                f"breakpoints={np.round(self.breakpoints.detach().numpy(), 3)}, "
                f"slopes={np.round(self.slopes.detach().numpy(), 2)}")

    def model_config(self):
        return {"hidden": self.hidden, "power": self.power}
