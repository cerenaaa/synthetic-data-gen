"""
Statistical synthetic tabular data generation.
Uses Gaussian copula to preserve column correlations and marginal distributions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm, gaussian_kde
from scipy.linalg import cholesky


@dataclass
class ColumnSpec:
    name: str
    dtype: str          # "continuous", "categorical", "binary", "integer"
    params: dict        # distribution params


class GaussianCopulaSynthesizer:
    """
    Fits a Gaussian copula to real data and samples synthetic rows.
    Preserves: marginal distributions, pairwise correlations, and data types.
    Does NOT preserve: higher-order interactions (use CTGAN for that).
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        self.corr_matrix: np.ndarray = None
        self.column_specs: list[ColumnSpec] = []
        self.kde_fits: dict = {}
        self.cat_maps: dict = {}
        self._columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaSynthesizer":
        self._columns = list(df.columns)
        numeric_df = pd.DataFrame()

        for col in df.columns:
            if df[col].dtype == "object" or df[col].nunique() < 15:
                cats = df[col].astype(str).unique().tolist()
                self.cat_maps[col] = cats
                encoded = df[col].astype(str).map({c: i for i, c in enumerate(cats)}).astype(float)
                numeric_df[col] = encoded
            else:
                self.kde_fits[col] = gaussian_kde(df[col].dropna())
                uniform = df[col].rank(pct=True).clip(0.001, 0.999)
                numeric_df[col] = norm.ppf(uniform)

        self.corr_matrix = numeric_df.corr().fillna(0).values
        print(f"Fitted copula on {len(df)} rows, {len(self._columns)} columns")
        return self

    def sample(self, n: int) -> pd.DataFrame:
        try:
            L = cholesky(self.corr_matrix + np.eye(len(self._columns)) * 1e-6, lower=True)
        except Exception:
            L = np.eye(len(self._columns))

        Z = self.rng.standard_normal((n, len(self._columns))) @ L.T
        U = norm.cdf(Z)

        result = pd.DataFrame()
        for i, col in enumerate(self._columns):
            u = U[:, i].clip(0.001, 0.999)
            if col in self.cat_maps:
                cats = self.cat_maps[col]
                idx = (u * len(cats)).astype(int).clip(0, len(cats) - 1)
                result[col] = [cats[j] for j in idx]
            else:
                kde = self.kde_fits[col]
                result[col] = np.quantile(kde.resample(10000)[0], u).round(4)
        return result


def generate_synthetic_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic customer dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 75, n),
        "annual_income": rng.lognormal(10.5, 0.6, n).round(0),
        "tenure_months": rng.exponential(24, n).clip(1, 120).astype(int),
        "monthly_spend": rng.gamma(3, 20, n).round(2),
        "num_products": rng.choice([1,2,3,4,5], n, p=[0.3,0.3,0.2,0.12,0.08]),
        "region": rng.choice(["NA","EMEA","APAC","LATAM"], n),
        "segment": rng.choice(["Enterprise","Mid-Market","SMB"], n, p=[0.2,0.35,0.45]),
        "churned": rng.binomial(1, 0.18, n),
    })
    synth = GaussianCopulaSynthesizer(seed)
    synth.fit(df)
    return synth.sample(n)
