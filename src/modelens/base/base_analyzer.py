import pandas as pd

from .correlation import correlation as get_correlation
from .info import info as get_info
from .vif import vif as get_vif
from .vif_with_linear_regression import (
    vif_with_linear_regression as get_vif_with_linear_regression,
)


class BaseAnalyzer:
    def __init__(self, df: pd.DataFrame, target: str):
        self._set_data(df, target)

    def _set_data(self, df: pd.DataFrame, target: str):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame.")

        self.df = df.copy()
        self.target = target

    def reinit(self, df: pd.DataFrame, target: str):
        self._set_data(df, target)

    def info(self):

        return get_info(self.df)

    def correlation(self, features=None, figsize=(12, 6), threshold=0.7):
        return get_correlation(
            self.df, features=features, threshold=threshold, figsize=figsize
        )

    def vif(self, features=None):
        return get_vif(df=self.df, features=features)

    def vif_with_linear_regression(self, features=None):
        return get_vif_with_linear_regression(df=self.df, features=features)
