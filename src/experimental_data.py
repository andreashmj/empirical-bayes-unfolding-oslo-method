"""
Script that loads experimental data for unfolding runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ompy as om
import ompy.response
from ompy.units import Quantity

from .input_checks import require_bool, require_finite_array, require_float, require_int
from .paths import repo_path


class ExperimentalDataLoader:
    has_truth = False

    def __init__(
        self,
        prompt_path: str | Path,
        background_path: str | Path | None = None,
        response_db: str = "OSCAR2020",
        rebin_factors: tuple[int, int] = (1, 1),
        lower_eg_cut: str = "550keV",
        include_background: bool = True,
        fwhm_eg_unfold: str = "30keV",
        eg_margin: str = "1000keV",
    ) -> None:
        rebin_values = tuple(rebin_factors)
        if len(rebin_values) != 2:
            raise ValueError("rebin_factors must contain exactly two integers.")

        self.ex_rebin_factor = require_int(rebin_values[0], "rebin_factors[0]", minimum=1)
        self.eg_rebin_factor = require_int(rebin_values[1], "rebin_factors[1]", minimum=1)
        self.include_background = require_bool(include_background, "include_background")
        self.eg_margin = str(eg_margin)

        self.response_db = str(response_db)
        self.lower_eg_cut = str(lower_eg_cut)
        self.fwhm_eg_unfold = str(fwhm_eg_unfold)

        self.prompt_path = str(repo_path(prompt_path))
        self.background_path = None
        if self.include_background:
            if background_path is None:
                raise ValueError("background_path is required when include_background=True.")
            self.background_path = str(repo_path(background_path))

        self._n = self._prepare_matrix(self._load_matrix(self.prompt_path))
        require_finite_array(self._n.values, "prompt matrix", ndim=2, nonnegative=True)

        if self.include_background:
            self._n_off = self._prepare_matrix(self._load_matrix(self.background_path))
            require_finite_array(self._n_off.values, "background matrix", ndim=2, nonnegative=True)
        else:
            self._n_off = (om.Matrix( values=np.zeros_like(self._n.values, dtype=float), X=self._n.Ex, Y=self._n.Eg, copy=False).set_xalias("Ex").set_yalias("Eg"))

        self.resp = (om.response.Response.from_db(self.response_db).normalize_FWHM("1330keV", self.fwhm_eg_unfold))
        self._D, self._G_g = self.resp.specialize_like(self._n)
        self._active_eg_lengths = self._compute_active_eg_lengths()

    def _load_matrix(self, path: str | Path) -> om.Matrix:
        return om.Matrix.from_path(str(path)).set_xalias("Ex").set_yalias("Eg")

    def _prepare_matrix(self, matrix: om.Matrix) -> om.Matrix:
        matrix = matrix.loc[:, self.lower_eg_cut:]

        if self.ex_rebin_factor > 1:
            matrix = matrix.rebin("Ex", factor=self.ex_rebin_factor)
        if self.eg_rebin_factor > 1:
            matrix = matrix.rebin("Eg", factor=self.eg_rebin_factor)

        return matrix

    def _compute_active_eg_lengths(self) -> np.ndarray:
        values = require_finite_array(self._n.values, "experimental prompt matrix", ndim=2, nonnegative=True)
        ex_axis = np.asarray(self._n.Ex, dtype=float)
        eg_axis = np.asarray(self._n.Eg, dtype=float)

        margin = require_float( Quantity(self.eg_margin).to("keV").magnitude, "eg_margin", minimum=0.0 )

        active_eg_lengths = np.ones(ex_axis.size, dtype=np.int64)
        for ex_index, ex_value in enumerate(ex_axis):
            row_upper_kev = max(float(ex_value) + margin, float(eg_axis[0]))
            active = np.searchsorted(eg_axis, row_upper_kev, side="right")
            active_eg_lengths[ex_index] = max(2, min(int(active), eg_axis.size))

        return active_eg_lengths

    def eg_slice_for_index(self, ex_index: int) -> slice:
        index = require_int(ex_index, "ex_index", minimum=0)
        return slice(0, int(self._active_eg_lengths[index]))

    @property
    def n(self) -> om.Matrix:
        return self._n.copy()

    @property
    def n_off(self) -> om.Matrix:
        return self._n_off.copy()

    @property
    def D(self) -> om.Matrix:
        return self._D.copy()

    @property
    def G_g(self) -> om.Matrix:
        return self._G_g.copy()

    @property
    def Ex(self) -> om.Vector:
        return self._n.Ex.copy()

    @property
    def Eg(self) -> om.Vector:
        return self._n.Eg.copy()

    @property
    def active_eg_lengths(self) -> np.ndarray:
        return np.asarray(self._active_eg_lengths, dtype=np.int64).copy()

    @property
    def background_expected(self) -> om.Matrix:
        return self._n_off.copy()

    @property
    def run_settings(self) -> dict:
        return {
            "dataset_kind": "experimental",
            "prompt_path": self.prompt_path,
            "background_path": self.background_path,
            "response_db": self.response_db,
            "rebin_factors": (self.ex_rebin_factor, self.eg_rebin_factor),
            "lower_eg_cut": self.lower_eg_cut,
            "fwhm_eg_unfold": self.fwhm_eg_unfold,
            "include_background": self.include_background,
            "eg_margin": self.eg_margin,
        }
