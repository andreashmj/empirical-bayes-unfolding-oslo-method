"""
Synthetic Oslo-method data generation.

The generated matrices use OMpy Matrix objects with axes (Ex, Eg). For one
fixed excitation-energy row, the code follows the OMpy right-hand convention,

    eta = x_true @ G_g
    nu  = x_true @ D @ G_g

where x_true is the emitted spectrum, eta is the resolution-limited emitted
spectrum, and nu is the expected detected signal.

If excitation-energy smearing is enabled, the Ex-resolution matrix acts from
the left,

    eta = G_in @ x_true @ G_g
    nu  = G_in @ x_true @ D @ G_g.

The synthetic ON/OFF count model is

    n_off ~ Poisson(background_expected),
    n     ~ Poisson(nu + background_expected).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ompy as om
import ompy.response
from ompy.detector.detector import LambdaExDetector

from .input_checks import require_bool,  require_finite_array, require_float, require_int
from .paths import DATA_DIR, repo_path


class SyntheticDataLoader:
    """
    
    Generate synthetic Oslo-method data on an OMpy Ex-Eg grid.

    Args:
        mat: Optional emitted truth matrix. If omitted, mat_path is loaded.
        mat_path: Path to the emitted truth matrix.
        response_db: Response database name used by OMpy.
        rebin_factors: Rebinning factors '(Ex factor, Eg factor)'.
        mat_scale: Multiplicative scale factor applied to the truth matrix.
        lower_eg_cut: Lower gamma-energy cut applied before response matching.
        include_background: Whether to generate ON/OFF background data.
        bg_fraction: Background-to-signal ratio inside each active Eg range.
        bg_flat_fraction: Fraction of the generated background assigned
            uniformly across the active Eg bins. The remaining fraction follows
            the detected signal expectation in the active Eg range.
        eg_tail_mass: Expected detected-signal tail mass excluded from the
            active Eg range.
        rng_seed: Seed used for synthetic Poisson draws.
        sigma_eg: Eg-resolution normalization passed to the response.
        include_ex_smearing: Whether to include Ex-resolution smearing.
        fwhm_ex_kev: Ex-resolution FWHM used when Ex smearing is enabled.
        
        
    """

    def __init__(
        self,
        mat: om.Matrix | None = None,
        mat_path: str | Path | None = None,
        response_db: str = "OSCAR2020",
        rebin_factors: tuple[int, int] = (1, 1),
        mat_scale: float = 1.0,
        lower_eg_cut: str = "60keV",
        include_background: bool = True,
        bg_fraction: float = 0.15,
        bg_flat_fraction: float = 0.10,
        eg_tail_mass: float = 1.0e-6,
        rng_seed: int = 123,
        sigma_eg: str = "30keV",
        include_ex_smearing: bool = False,
        fwhm_ex_kev: float = 150.0,
    ) -> None:
        if mat is not None and mat_path is not None:
            raise ValueError("Pass either mat or mat_path, not both.")

        rebin_values = tuple(rebin_factors)
        if len(rebin_values) != 2:
            raise ValueError("rebin_factors must contain exactly two integers.")

        self.ex_rebin_factor = require_int(
            rebin_values[0],
            "rebin_factors[0]",
            minimum=1,
        )
        self.eg_rebin_factor = require_int(
            rebin_values[1],
            "rebin_factors[1]",
            minimum=1,
        )

        self.rng_seed = require_int(rng_seed, "rng_seed", minimum=0)

        self.mat_scale = require_float(
            mat_scale,
            "mat_scale",
            minimum=0.0,
            minimum_inclusive=False,
        )

        self.include_background = require_bool(
            include_background,
            "include_background",
        )
        self.include_ex_smearing = require_bool(
            include_ex_smearing,
            "include_ex_smearing",
        )

        self.eg_tail_mass = require_float(
            eg_tail_mass,
            "eg_tail_mass",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )

        self.response_db = str(response_db)
        self.lower_eg_cut = str(lower_eg_cut)
        self.sigma_eg = str(sigma_eg)

        self.resp_norm_anchor = "1330keV"
        self.resp_norm_sigma = self.sigma_eg

        if self.include_ex_smearing:
            self.fwhm_ex_kev = require_float(
                fwhm_ex_kev,
                "fwhm_ex_kev",
                minimum=0.0,
                minimum_inclusive=False,
            )
        else:
            self.fwhm_ex_kev = None

        if self.include_background:
            self.bg_fraction = require_float(
                bg_fraction,
                "bg_fraction",
                minimum=0.0,
            )
            self.bg_flat_fraction = require_float(
                bg_flat_fraction,
                "bg_flat_fraction",
                minimum=0.0,
                maximum=1.0,
            )
        else:
            self.bg_fraction = 0.0
            self.bg_flat_fraction = 0.0

        self.mat_path: str | None
        self.mat = self._load_truth_matrix(mat, mat_path)
        self.mat = self._prepare_truth_matrix(self.mat)

        require_finite_array(
            self.mat.values,
            "truth matrix",
            ndim=2,
            nonnegative=True,
        )

        self.resp = (
            om.response.Response.from_db(self.response_db)
            .normalize_sigma(self.resp_norm_anchor, self.resp_norm_sigma)
        )

        self._G_in = self._build_ex_resolution_operator()
        self._D, self._G_g = self.resp.specialize_like(self.mat)

        eta_matrix, signal_expected_matrix = self._expected_signal_matrices()

        axes = dict(X=self.mat.Ex, Y=self.mat.Eg, copy=False)

        self._eta = eta_matrix.set_xalias("Ex").set_yalias("Eg")

        signal_expected = require_finite_array(
            signal_expected_matrix.values,
            "signal_expected",
            ndim=2,
            nonnegative=True,
        )

        self._nu = (
            om.Matrix(values=signal_expected, **axes)
            .set_xalias("Ex")
            .set_yalias("Eg")
        )

        self._active_eg_lengths = self._compute_active_eg_lengths(
            signal_expected,
        )

        background_expected = self._background_expectation(signal_expected)
        background_expected = require_finite_array(
            background_expected,
            "background_expected",
            ndim=2,
            nonnegative=True,
        )

        rng = np.random.default_rng(self.rng_seed)

        n_off_observed = rng.poisson(background_expected)

        on_expected = require_finite_array(
            signal_expected + background_expected,
            "on_expected",
            ndim=2,
            nonnegative=True,
        )
        n_observed = rng.poisson(on_expected)

        self._background_expected = (
            om.Matrix(values=background_expected, **axes)
            .set_xalias("Ex")
            .set_yalias("Eg")
        )
        self._n_off = (
            om.Matrix(values=n_off_observed, **axes)
            .set_xalias("Ex")
            .set_yalias("Eg")
        )
        self._n = (
            om.Matrix(values=n_observed, **axes)
            .set_xalias("Ex")
            .set_yalias("Eg")
        )

    def _load_truth_matrix(
        self,
        mat: om.Matrix | None,
        mat_path: str | Path | None,
    ) -> om.Matrix:
        """Load or copy the emitted truth matrix."""

        if mat is not None:
            self.mat_path = None
            return mat.copy()

        if mat_path is None:
            path = DATA_DIR / "ExEg_1e8.npz"
        else:
            path = repo_path(mat_path)

        self.mat_path = str(path)
        return om.Matrix.from_path(str(path))

    def _prepare_truth_matrix(self, matrix: om.Matrix) -> om.Matrix:
        """Apply scale, lower-Eg cut, and rebinning to the truth matrix."""

        matrix = matrix * self.mat_scale
        matrix = matrix.loc[:, self.lower_eg_cut:]

        if self.ex_rebin_factor > 1:
            matrix = matrix.rebin("Ex", factor=self.ex_rebin_factor)

        if self.eg_rebin_factor > 1:
            matrix = matrix.rebin("Eg", factor=self.eg_rebin_factor)

        return matrix

    def _build_ex_resolution_operator(self) -> om.Matrix | None:
        """Build the Ex-resolution operator, or None when Ex smearing is disabled."""

        if not self.include_ex_smearing:
            return None

        detector = LambdaExDetector(lambda excitation_energy: self.fwhm_ex_kev)

        return (
            detector.resolution_matrix(self.mat)
            .set_xalias("Ex")
            .set_yalias("Ex")
        )

    def _expected_signal_matrices(self) -> tuple[om.Matrix, om.Matrix]:
        """Return eta and detected-signal expectation before Poisson sampling."""

        if self._G_in is None:
            eta_matrix = self.mat @ self._G_g
            signal_expected_matrix = self.mat @ self._D @ self._G_g
        else:
            eta_matrix = self._G_in @ self.mat @ self._G_g
            signal_expected_matrix = self._G_in @ self.mat @ self._D @ self._G_g

        return eta_matrix, signal_expected_matrix

    def _background_expectation(self, signal_expected: np.ndarray) -> np.ndarray:
        """Return expected OFF-background counts.

        For each Ex row, the background is defined in the active Eg range used
        for the unfolding. Its total is a fixed fraction of the detected-signal
        total in that range. The binwise shape is a mixture of a
        signal-proportional component and a flat component.
        """

        background_expected = np.zeros_like(signal_expected, dtype=float)

        if not self.include_background or self.bg_fraction == 0.0:
            return background_expected

        for ex_index, active_eg_length in enumerate(self._active_eg_lengths):
            signal_active = signal_expected[ex_index, :active_eg_length]
            signal_total = float(np.sum(signal_active))

            if signal_total <= 0.0:
                continue

            background_total = self.bg_fraction * signal_total

            signal_shape = signal_active / signal_total
            flat_shape = np.full(
                active_eg_length,
                1.0 / float(active_eg_length),
                dtype=float,
            )

            background_shape = (
                (1.0 - self.bg_flat_fraction) * signal_shape
                + self.bg_flat_fraction * flat_shape
            )

            background_expected[ex_index, :active_eg_length] = (
                background_total * background_shape
            )

        return background_expected

    def _compute_active_eg_lengths(self, signal_expected: np.ndarray) -> np.ndarray:
        """Return the number of active Eg bins for each Ex row."""

        n_ex, n_eg = signal_expected.shape
        active_eg_lengths = np.ones(n_ex, dtype=np.int64)

        for ex_index in range(n_ex):
            row_signal = signal_expected[ex_index, :]
            row_total = float(np.sum(row_signal))

            if row_total <= 0.0:
                active_eg_lengths[ex_index] = 1
                continue

            retained_mass = (1.0 - self.eg_tail_mass) * row_total
            cumulative_mass = np.cumsum(row_signal)

            last_index = int(
                np.searchsorted(cumulative_mass, retained_mass, side="left")
            )

            active_eg_lengths[ex_index] = min(last_index + 1, n_eg)

        return active_eg_lengths

    def eg_slice_for_index(self, ex_index: int) -> slice:
        """Return the active Eg slice for one Ex-row index."""

        index = require_int(ex_index, "ex_index", minimum=0)

        if index >= self._active_eg_lengths.size:
            raise IndexError(
                f"ex_index={index} is outside the Ex axis with "
                f"{self._active_eg_lengths.size} rows."
            )

        return slice(0, int(self._active_eg_lengths[index]))

    @property
    def x_true(self) -> om.Matrix:
        """Emitted truth matrix."""

        return self.mat.copy()

    @property
    def eta(self) -> om.Matrix:
        """Resolution-limited emitted-spectrum expectation."""

        return self._eta.copy()

    @property
    def nu(self) -> om.Matrix:
        """Expected detected signal counts."""

        return self._nu.copy()

    @property
    def n(self) -> om.Matrix:
        """Observed ON counts."""

        return self._n.copy()

    @property
    def n_off(self) -> om.Matrix:
        """Observed OFF counts."""

        return self._n_off.copy()

    @property
    def background_expected(self) -> om.Matrix:
        """Expected OFF-background counts."""

        return self._background_expected.copy()

    @property
    def D(self) -> om.Matrix:
        """Discrete redistribution response operator."""

        return self._D.copy()

    @property
    def G_g(self) -> om.Matrix:
        """Gamma-energy resolution operator."""

        return self._G_g.copy()

    @property
    def G_in(self) -> om.Matrix | None:
        """Excitation-energy resolution operator, if enabled."""

        if self._G_in is None:
            return None

        return self._G_in.copy()

    @property
    def Ex(self) -> om.Vector:
        """Excitation-energy axis."""

        return self.mat.Ex.copy()

    @property
    def Eg(self) -> om.Vector:
        """Gamma-energy axis."""

        return self.mat.Eg.copy()

    @property
    def active_eg_lengths(self) -> np.ndarray:
        """Number of active Eg bins for each Ex row."""

        return np.asarray(self._active_eg_lengths, dtype=np.int64).copy()

    @property
    def metadata(self) -> dict:
        """Return settings used to construct the synthetic data."""

        return {
            "mat_path": self.mat_path,
            "rng_seed": self.rng_seed,
            "response_db": self.response_db,
            "resp_norm_anchor": self.resp_norm_anchor,
            "resp_norm_sigma": self.resp_norm_sigma,
            "rebin_factors": (self.ex_rebin_factor, self.eg_rebin_factor),
            "mat_scale": self.mat_scale,
            "lower_eg_cut": self.lower_eg_cut,
            "sigma_eg": self.sigma_eg,
            "include_ex_smearing": self.include_ex_smearing,
            "fwhm_ex_kev": self.fwhm_ex_kev,
            "include_background": self.include_background,
            "bg_fraction": self.bg_fraction,
            "bg_flat_fraction": self.bg_flat_fraction,
            "eg_tail_mass": self.eg_tail_mass,
        }
