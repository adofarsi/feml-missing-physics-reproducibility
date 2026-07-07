import numpy as np

MIN_VALID_SPECIMENS = 2
SIGMA_REL_FLOOR = 0.1

HARD_D_ELASTIC_MAX_MM = 0.30
HARD_ELASTIC_START_FRAC = 0.85
HARD_N_ELASTIC = 4
HARD_TRANSITION_BAND_MM = (0.30, 0.50)
HARD_N_TRANSITION = 10
HARD_PLASTIC_STEP_MM = 0.17
HARD_W_ELASTIC = 2.0
HARD_W_TRANSITION = 1.0
HARD_W_OUTER = 1.0

DMG_N_BASE = 20
DMG_N_TRANSITION = 6
DMG_TRANSITION_FORCE = (18.0, 25.0)
DMG_N_SOFTENING = 12
DMG_W_PRE_PEAK = 1.5
DMG_W_POST_PEAK = 2.0


def load_corrected(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, filling_values=np.nan)
    disp = np.asarray(data["displacement_mm"], dtype=float)
    cols = [n for n in data.dtype.names if n != "displacement_mm"]
    forces = np.column_stack([np.asarray(data[n], dtype=float) for n in cols])

    usable = np.isfinite(forces).sum(axis=1) >= MIN_VALID_SPECIMENS
    mean = np.full_like(disp, np.nan)
    std = np.full_like(disp, np.nan)
    with np.errstate(invalid="ignore"):
        mean[usable] = np.nanmean(forces[usable], axis=1)
        std[usable] = np.nanstd(forces[usable], axis=1)
    return disp[usable], mean[usable], std[usable]


def load_specimens(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, filling_values=np.nan)
    disp = np.asarray(data["displacement_mm"], dtype=float)
    cols = [n for n in data.dtype.names if n != "displacement_mm"]
    forces = np.column_stack([np.asarray(data[n], dtype=float) for n in cols])
    return disp, forces, cols


def _stats_at(levels, disp, mean, std):
    mean_at = np.interp(levels, disp, mean)
    std_at = np.interp(levels, disp, std)
    peak = max(float(np.max(mean)), 1.0e-12)
    sigma_eff = np.sqrt(std_at**2 + (SIGMA_REL_FLOOR * peak) ** 2)
    return mean_at, std_at, sigma_eff, peak


def hardening_targets(csv_path):
    disp, mean, std = load_corrected(csv_path)
    i_peak = int(np.argmax(mean))
    d_fit = disp[: i_peak + 1]
    d_fit_max = float(d_fit[-1])
    d_el = min(HARD_D_ELASTIC_MAX_MM, d_fit_max)

    elastic = np.linspace(HARD_ELASTIC_START_FRAC * d_el, d_el, HARD_N_ELASTIC)
    lo, hi = sorted(HARD_TRANSITION_BAND_MM)
    lo, hi = max(lo, d_el), min(hi, d_fit_max)
    transition = np.linspace(lo, hi, HARD_N_TRANSITION + 1)[1:] if hi > lo else np.array([])
    outer_lo = hi if transition.size else d_el
    if d_fit_max > outer_lo + 1e-9:
        n_outer = max(int(round((d_fit_max - outer_lo) / HARD_PLASTIC_STEP_MM)), 1)
        outer = np.linspace(outer_lo, d_fit_max, n_outer + 1)[1:]
    else:
        outer = np.array([])

    levels = np.unique(np.concatenate([elastic, transition, outer]))
    mean_at, std_at, sigma_eff, peak = _stats_at(levels, d_fit, mean[: i_peak + 1], std[: i_peak + 1])
    hi_trans = max(HARD_TRANSITION_BAND_MM)
    is_elastic = levels <= d_el * (1.0 + 1.0e-12)
    is_outer = levels > hi_trans * (1.0 + 1.0e-12)
    is_transition = ~is_elastic & ~is_outer
    return {
        "displacement_levels": levels, "mean_force": mean_at, "std_force": std_at,
        "sigma_eff": sigma_eff, "is_elastic": is_elastic, "is_transition": is_transition,
        "is_outer": is_outer, "peak_force": peak,
        "d_elastic_max": d_el, "d_peak": d_fit_max,
        "zones": {"elastic": elastic, "transition": transition, "outer": outer},
        "full_disp": disp, "full_mean": mean, "full_std": std,
        "branch_weights": (HARD_W_ELASTIC, HARD_W_TRANSITION, HARD_W_OUTER),
    }


def damage_targets(csv_path):
    disp, mean, std = load_corrected(csv_path)
    i_peak = int(np.argmax(mean))
    d_peak = float(disp[i_peak])
    d_fit_max = float(disp[-1])

    base = np.linspace(d_fit_max / DMG_N_BASE, d_fit_max, DMG_N_BASE)
    rising_d, rising_f = disp[: i_peak + 1], mean[: i_peak + 1]
    env = np.maximum.accumulate(rising_f)
    f_u, idx = np.unique(env, return_index=True)
    d_u = rising_d[idx]
    f_lo = max(min(DMG_TRANSITION_FORCE), float(f_u.min()))
    f_hi = min(max(DMG_TRANSITION_FORCE), float(f_u.max()))
    transition = np.interp(np.linspace(f_lo, f_hi, DMG_N_TRANSITION), f_u, d_u) if f_hi > f_lo else np.array([])
    softening = np.linspace(d_peak, d_fit_max, DMG_N_SOFTENING + 1)[1:] if d_fit_max > d_peak else np.array([])

    levels = np.unique(np.concatenate([base, transition, softening]))
    levels = levels[(levels > 0) & (levels <= d_fit_max * (1 + 1e-12))]
    mean_at, std_at, sigma_eff, peak = _stats_at(levels, disp, mean, std)
    is_post = levels > d_peak
    return {
        "displacement_levels": levels, "mean_force": mean_at, "std_force": std_at,
        "sigma_eff": sigma_eff, "is_post_peak": is_post, "peak_force": peak,
        "d_peak": d_peak, "d_fit_max": d_fit_max,
        "zones": {"base": base, "transition": transition, "softening": softening},
        "full_disp": disp, "full_mean": mean, "full_std": std,
        "branch_weights": (DMG_W_PRE_PEAK, DMG_W_POST_PEAK),
    }


def force_curve_loss(predicted_loads, mean_force, sigma_eff, branches):
    err2 = [((pred - float(t)) / float(s)) ** 2
            for pred, t, s in zip(predicted_loads, mean_force, sigma_eff)]
    loss = 0.0
    weight_sum = 0.0
    for mask, weight in branches:
        idx = np.flatnonzero(mask)
        if idx.size:
            branch_loss = 0.0
            for i in idx:
                branch_loss += err2[int(i)] / len(idx)
            loss += weight * branch_loss
            weight_sum += weight
    return loss / max(weight_sum, 1.0e-12)
