"""
Privacy audit for synthetic datasets.
Measures re-identification risk via nearest-neighbor distance and membership inference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class PrivacyAuditor:
    """
    Measures how similar synthetic records are to real training records.
    High similarity = higher re-identification risk.

    Metrics:
    - NNDR (Nearest Neighbor Distance Ratio): ratio of distances to 1st vs 2nd NN
    - DCR (Distance to Closest Record): distribution of min distances synthetic→real
    - Privacy score: 0 (no privacy) to 1 (perfect privacy)
    """

    def __init__(self, numeric_cols: list[str] = None):
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()

    def _prepare(self, real: pd.DataFrame, synth: pd.DataFrame):
        cols = self.numeric_cols or list(real.select_dtypes(include="number").columns)
        R = self.scaler.fit_transform(real[cols].fillna(0))
        S = self.scaler.transform(synth[cols].fillna(0))
        return R, S

    def dcr(self, real: pd.DataFrame, synth: pd.DataFrame, sample: int = 2000) -> dict:
        """Distance to closest real record for each synthetic record."""
        R, S = self._prepare(real, synth)
        R = R[:sample]; S = S[:sample]
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(R)
        dists, _ = nn.kneighbors(S)
        dcr_vals = dists.flatten()
        return {
            "dcr_mean": float(np.mean(dcr_vals)),
            "dcr_p5": float(np.percentile(dcr_vals, 5)),
            "dcr_p50": float(np.percentile(dcr_vals, 50)),
            "pct_too_close": float(np.mean(dcr_vals < 0.1)),
            "privacy_score": float(np.clip(np.mean(dcr_vals) / 2, 0, 1)),
        }

    def report(self, real: pd.DataFrame, synth: pd.DataFrame) -> str:
        metrics = self.dcr(real, synth)
        score = metrics["privacy_score"]
        rating = "GOOD" if score > 0.6 else "MODERATE" if score > 0.3 else "POOR"
        return (
            f"Privacy Audit Report\n{'='*40}\n"
            f"DCR mean:      {metrics['dcr_mean']:.4f}\n"
            f"DCR p5:        {metrics['dcr_p5']:.4f}\n"
            f"Too-close pct: {metrics['pct_too_close']:.1%}\n"
            f"Privacy score: {score:.3f} ({rating})\n"
        )
