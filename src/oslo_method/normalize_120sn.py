import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.special import ndtr


# PRC 106, 034322, Table I; MeV except eV/meV inputs.
# Normalization operations: oslo-method-software/prog/counting.c, commit d2d1172.
SN, D0_EV, Gg_meV = 9.105, 95.0, 121.0
D0_ERROR_EV, Gg_error_meV = 14.0, 25.0
RHO_SN, BETA, T_REFERENCE = 3.66e5, 0.70, 0.72
E_DISC, SIGMA_DISC, SIGMA_SN = 2.53, 3.4, 5.82
NLD_LOW, NLD_HIGH = (1.95, 2.85), (4.35, 7.65)  # Estimated figure intervals.
GSF_LOW, GSF_HIGH = (1.30, 2.50), (7.00, 8.80)  # Provisional end-region fits.
LEVEL_FWHM = 0.10
CORRECTIONS = 0.25 + 0.001 * np.arange(1, 3752)
TEMPERATURES = np.arange(round(0.8 * T_REFERENCE, 2), 1.2 * T_REFERENCE, 0.005)


def indices(energy, limits):
    a, b = [np.abs(energy - v).argmin() for v in limits]
    return np.arange(a, b + 1)


def discrete_density(energy, levels):
    width = np.median(np.diff(energy))
    sigma = LEVEL_FWHM / np.sqrt(8 * np.log(2))
    z = (energy[:, None] - levels) / sigma
    return (ndtr(z + width/(2*sigma)) - ndtr(z - width/(2*sigma))).sum(1) / width


def initial_constants(rho, energy, discrete, anchor, temperature):
    low, high = indices(energy, NLD_LOW), indices(energy, NLD_HIGH)
    l, h = (low[0] + low[-1]) // 2, (high[0] + high[-1]) // 2
    rlow, rhigh = rho[l-1:l+2].mean(), rho[h-1:h+2].mean()
    dlow = max(discrete[l-1:l+2].mean(), 0.01)
    dhigh = anchor * np.exp((energy[h] - SN) / temperature)
    if min(rlow, rhigh) <= 0 or not np.isfinite(rlow + rhigh):
        raise ValueError("Unconstrained NLD at a fixed normalization anchor.")
    alpha = (np.log(dlow/rlow) - np.log(dhigh/rhigh)) / (energy[l] - energy[h])
    A = dlow / rlow * np.exp(-alpha * energy[l])
    return A, alpha, energy[l], energy[h]


def count_correction(values, errors, target):
    c = CORRECTIONS[:, None]
    residual = c * values - target
    variance = np.hypot((c * errors)**2, 1.0)
    chi = (residual**2 / variance).sum(axis=1)
    best = len(chi) - 1 - np.argmin(chi[::-1])
    return CORRECTIONS[best], chi[best]


def normalize_nld(rho, relative_error, energy, discrete, anchor):
    # counting.c: searchT, searchAalpha, corrL/corrH. Select the printed scan minimum.
    low, high = indices(energy, NLD_LOW), indices(energy, NLD_HIGH)
    if not np.isfinite(rho[np.r_[low, high]]).all():
        raise ValueError("Unconstrained bins inside a fixed NLD normalization interval.")
    error = np.nan_to_num(relative_error * np.abs(rho))
    best = None
    for temperature in TEMPERATURES:
        A, alpha, el, eh = initial_constants(rho, energy, discrete, anchor, temperature)
        factor = A * np.exp(alpha * energy[high])
        target = anchor * np.exp((energy[high] - SN) / temperature)
        correction, chi = count_correction(factor*rho[high], factor*error[high], target)
        if best is None or chi < best[0]:
            best = chi, temperature, A, alpha, el, eh, correction
    _, temperature, A, alpha, el, eh, chigh = best
    factor = A * np.exp(alpha * energy[low])
    clow, _ = count_correction(factor*rho[low], factor*error[low],
                               np.maximum(discrete[low], 0.01))
    change = np.log(clow/chigh) / (el-eh)
    A *= clow * np.exp(-change*el)
    alpha += change
    return A * np.exp(alpha*energy) * rho, A, alpha, temperature


def relative_spread(values):
    finite = np.isfinite(values).all(axis=0)
    result = np.zeros(values.shape[1])
    mean = values[:, finite].mean(axis=0)
    sd = values[:, finite].std(axis=0, ddof=1)
    result[finite] = np.divide(sd, np.abs(mean), out=np.zeros_like(sd), where=mean != 0)
    return result


def reference_errors(rho, transmission, energy, gamma_energy, discrete):
    # Put the arbitrary Oslo scale/slope in a common physical reference before estimating errors.
    aligned_r, aligned_t = np.empty_like(rho), np.empty_like(transmission)
    for i, (r, t) in enumerate(zip(rho, transmission)):
        A, alpha, _, _ = initial_constants(r, energy, discrete, BETA*RHO_SN, T_REFERENCE)
        aligned_r[i] = A * np.exp(alpha*energy) * r
        aligned_t[i] = np.exp(alpha*gamma_energy) * t
        aligned_t[i] /= np.nansum(aligned_t[i])
    return relative_spread(aligned_r), relative_spread(aligned_t)


def log_transmission_fit(transmission, relative_error, energy, limits):
    use = indices(energy, limits)
    use = use[np.isfinite(transmission[use]) & (transmission[use] > 0)
              & np.isfinite(relative_error[use]) & (relative_error[use] > 0)]
    if len(use) < 2:
        raise ValueError("Fewer than two weighted bins in a fixed transmission-fit interval.")
    design = np.column_stack((energy[use], np.ones(len(use))))
    weights = 1 / relative_error[use]
    # Same log-transmission chi-square as counting.c, solved without the coarse 1000x1000 grid.
    return np.linalg.lstsq(design * weights[:, None], np.log(transmission[use]) * weights,
                           rcond=None)[0]


def spin_sum(energy):
    s2 = SIGMA_DISC**2 + (energy-E_DISC)/(SN-E_DISC) * (SIGMA_SN**2-SIGMA_DISC**2)
    s2 = np.maximum(s2, 0.5)
    def g(j):
        return (2*j+1)/(2*s2) * np.exp(-(j+0.5)**2/(2*s2))
    return g(0) + 2*g(1) + g(2)


def normalize_gsf(transmission, relative_error, gamma_energy, rho, energy,
                  alpha, temperature, anchor, d0, gamma_width):
    t = np.exp(alpha*gamma_energy) * transmission
    al, bl = log_transmission_fit(t, relative_error, gamma_energy, GSF_LOW)
    ah, bh = log_transmission_fit(t, relative_error, gamma_energy, GSF_HIGH)
    ilow, ihigh = indices(gamma_energy, GSF_LOW), indices(gamma_energy, GSF_HIGH)
    e1, e2, eh = gamma_energy[ilow[0]], gamma_energy[ilow[-1]], gamma_energy[ihigh[0]]
    grid = np.linspace(0, SN, 2001)
    finite = np.isfinite(t)
    extended_t = np.interp(grid, gamma_energy[finite], t[finite])
    low, high = grid <= e2, grid >= eh
    extended_t[low] = np.exp(al*grid[low] + bl)
    extended_t[high] = np.exp(ah*grid[high] + bh)
    below = grid < e1
    extended_t[below] = grid[below]**3 * np.exp((al-3/e1)*grid[below] + bl+3-3*np.log(e1))
    final_energy = SN - grid
    finite = np.isfinite(rho) & (energy >= -1e-9)
    extended_rho = np.interp(final_energy, energy[finite], rho[finite])
    h1 = energy[indices(energy, NLD_HIGH)[0]]
    high = final_energy >= h1
    extended_rho[high] = anchor * np.exp((final_energy[high]-SN)/temperature)
    integral = trapezoid(extended_t * extended_rho * spin_sum(final_energy), grid)
    B = gamma_width * 4*np.pi / (1000*d0*integral)
    return B*t/(2*np.pi*gamma_energy**3), B


def normalize_samples(ds, levels):
    energy, gamma_energy = ds.E_nld.values/1000, ds.E_gsf.values/1000
    rho = ds.nld_unnormalized.values
    transmission = ds.gsf_unnormalized.values * (2*np.pi*ds.E_gsf.values**3)
    discrete = discrete_density(energy, levels)
    sr, st = reference_errors(rho, transmission, energy, gamma_energy, discrete)
    n = len(rho)
    r_out, f_out = np.empty_like(rho), np.empty_like(transmission)
    pars = np.empty((n, 4))
    cases = ((D0_EV-D0_ERROR_EV, Gg_meV), (D0_EV+D0_ERROR_EV, Gg_meV),
             (D0_EV, Gg_meV-Gg_error_meV), (D0_EV, Gg_meV+Gg_error_meV))
    r_cases = np.empty((n, len(cases), len(energy)))
    f_cases = np.empty((n, len(cases), len(gamma_energy)))
    for i in range(n):
        for j, (d0, width) in enumerate(((D0_EV, Gg_meV),) + cases[:2]):
            anchor = BETA * RHO_SN * D0_EV/d0
            r, A, alpha, temperature = normalize_nld(rho[i], sr, energy, discrete, anchor)
            f, B = normalize_gsf(transmission[i], st, gamma_energy, r, energy,
                                  alpha, temperature, anchor, d0, width)
            if j == 0:
                r_out[i], f_out[i], pars[i] = r, f, (A, alpha, B, temperature)
            else:
                r_cases[i, j-1], f_cases[i, j-1] = r, f
        for j in (2, 3):
            r_cases[i, j] = r_out[i]
            f_cases[i, j] = f_out[i] * cases[j][1]/Gg_meV
        print(f"Normalization {i+1}/{n}", flush=True)
    out = xr.Dataset(
        {"nld": (("sample", "E_nld"), r_out), "gsf": (("sample", "E_gsf"), f_out),
         "nld_reference_variation": (("sample", "reference_case", "E_nld"), r_cases),
         "gsf_reference_variation": (("sample", "reference_case", "E_gsf"), f_cases),
         "unfolding_member": ("sample", np.arange(n)),
         "input_chain": ("sample", ds.input_chain.values),
         "input_draw": ("sample", ds.input_draw.values)},
        coords={"sample": np.arange(n), "E_nld": energy, "E_gsf": gamma_energy,
                "reference_case": ["D0_minus_1sd", "D0_plus_1sd", "Gg_minus_1sd", "Gg_plus_1sd"]},
        attrs={"beta": BETA, "Sn_MeV": SN, "rho_Sn_reduced": BETA*RHO_SN,
               "uncertainty": "nld/gsf propagate unfolding only at central external inputs; reference variations are separate, not posterior draws",
               "normalization": "counting.c anchor corrections, CT scan minimum, log-transmission fits, radiative-width integral",
               "provisional_settings": "C/O rectangles, fit intervals, discrete-level input, and automation of manual CT selection",
               "extraction_weights": "One first-generation ensemble standard-deviation matrix; marginal-error chi-square, not full covariance",
               "transmission_fits": "Weighted log-linear objective from counting.c; direct least-squares solution instead of coarse parameter grid"},
    )
    for name, value in zip(("A", "alpha", "B", "T"), pars.T):
        out[name] = ("sample", value)
    out.E_nld.attrs["units"] = out.E_gsf.attrs["units"] = "MeV"
    out.nld.attrs["units"], out.gsf.attrs["units"] = "MeV^-1", "MeV^-3"
    return out
