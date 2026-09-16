import pandas as pd

from .check_overfitting import check_overfitting as get_check_overfitting
from .compare_models import compare_models as get_compare_models
from .compare_regularization import compare_regularization as get_compare_regularization
from .correlation import correlation as get_correlation
from .evaluate_feature_removal_combinations import (
    evaluate_feature_removal_combinations as get_evaluate_feature_removal_combinations,
)
from .evaluate_single_feature_removal import (
    evaluate_single_feature_removal as get_evaluate_single_feature_removal,
)
from .info import info as get_info
from .learning_curve import plot_learning_curve
from .permutation_importance import (
    calculate_permutation_importance as get_permutation_importance,
)
from .residual_analysis import residual_analysis as get_residual_analysis
from .tune_model import tune_model as get_tune_model
from .vif import vif as get_vif
from .vif_with_linear_regression import (
    vif_with_linear_regression as get_vif_with_linear_regression,
)


class RegressionAnalyzer:
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

    def evaluate_single_feature_removal(
        self,
        model=None,
        features=None,
        export_html=False,
        file_name=None,
        random_state=42,
    ):
        return get_evaluate_single_feature_removal(
            df=self.df,
            model=model,
            features=features,
            target=self.target,
            random_state=random_state,
            export_html=export_html,
            file_name=file_name,
        )

    def evaluate_feature_removal_combinations(
        self,
        candidates,
        features,
        model=None,
        export_html=False,
        file_name=None,
        random_state=42,
    ):
        return get_evaluate_feature_removal_combinations(
            df=self.df,
            model=model,
            features=features,
            candidates=candidates,
            target=self.target,
            random_state=random_state,
            export_html=export_html,
            file_name=file_name,
        )

    def compare_models(
        self,
        features,
        models,
        export_html=False,
        file_name=None,
        random_state=42,
    ):
        return get_compare_models(
            df=self.df,
            features=features,
            target=self.target,
            models=models,
            export_html=export_html,
            file_name=file_name,
            random_state=random_state,
        )

    def permutation_importance(
        self,
        random_state=42,
        export_html=False,
        html_path="html_reports/permutation_importance.html",
        model=None,
    ):

        return get_permutation_importance(
            df=self.df,
            model=model,
            target=self.target,
            random_state=random_state,
            export_html=export_html,
            html_path=html_path,
        )

    def check_overfitting(self, features=None, model=None, random_state=42):
        return get_check_overfitting(
            df=self.df,
            features=features,
            target=self.target,
            random_state=random_state,
            model=model,
        )

    def residual_analysis(
        self,
        features=None,
        model=None,
        test_size=0.2,
        random_state=42,
        export_html=False,
        html_path="html_reports/residual_analysis.html",
        residual_threshold=4,
    ):
        return get_residual_analysis(
            df=self.df,
            features=features,
            target=self.target,
            model=model,
            test_size=test_size,
            random_state=random_state,
            export_html=export_html,
            html_path=html_path,
            residual_threshold=residual_threshold,
        )

    def compare_regularization(
        self,
        features=None,
        alphas=None,
        l1_ratio=0.5,
        random_state=42,
    ):
        return get_compare_regularization(
            df=self.df,
            target=self.target,
            features=features,
            alphas=alphas,
            l1_ratio=l1_ratio,
            random_state=random_state,
        )

    def tune_model(
        self,
        model,
        param_grid: dict,
        features=None,
        scoring=None,
        cv=5,
        export_html=False,
        file_name=None,
        random_state=42,
    ):
        return get_tune_model(
            df=self.df,
            model=model,
            target=self.target,
            param_grid=param_grid,
            features=features,
            scoring=scoring,
            cv=cv,
            export_html=export_html,
            file_name=file_name,
            random_state=random_state,
        )

    def learning_curve(
        self,
        features=None,
        model=None,
        cv=5,
        random_state=42,
        train_sizes=None,
    ):
        return plot_learning_curve(
            df=self.df,
            target=self.target,
            features=features,
            model=model,
            cv=cv,
            random_state=random_state,
            train_sizes=train_sizes,
        )
