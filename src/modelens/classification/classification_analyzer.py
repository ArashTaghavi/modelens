import pandas as pd

from modelens.base import BaseAnalyzer

from .analyze_prediction_group import (
    analyze_prediction_group as get_analyze_prediction_group,
)
from .compare_class_weight import (
    compare_class_weight as get_compare_class_weight,
)
from .compare_feature_sets import compare_feature_sets as get_compare_feature_sets
from .compare_models import compare_models as get_compare_models
from .evaluate_feature_removal_combinations import (
    evaluate_feature_removal_combinations as get_evaluate_feature_removal_combinations,
)
from .evaluate_single_feature_removal import (
    evaluate_single_feature_removal as get_evaluate_single_feature_removal,
)
from .evaluate_thresholds import evaluate_thresholds as get_evaluate_thresholds
from .feature_extraction import feature_extraction as get_feature_extraction
from .learning_curve import plot_learning_curve
from .permutation_importance import (
    calculate_permutation_importance as get_permutation_importance,
)
from .prediction_error_analysis import (
    prediction_error_analysis as get_prediction_error_analysis,
)
from .tune_model import (
    tune_model as get_tune_model,
)


class ClassificationAnalyzer(BaseAnalyzer):
    def __init__(self, df: pd.DataFrame, target: str):
        self._set_data(df, target)

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

    def compare_feature_sets(
        self,
        model,
        feature_sets: dict,
        cv=5,
        export_html=False,
        file_name="feature_sets_comparison",
        random_state=42,
    ):

        return get_compare_feature_sets(
            df=self.df,
            target=self.target,
            model=model,
            feature_sets=feature_sets,
            cv=cv,
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

    def prediction_error_analysis(
        self,
        features=None,
        model=None,
        test_size=0.2,
        random_state=42,
        export_html=False,
        html_path="html_reports/residual_analysis.html",
        residual_threshold=4,
    ):
        return get_prediction_error_analysis(
            df=self.df,
            features=features,
            target=self.target,
            model=model,
            test_size=test_size,
            random_state=random_state,
            export_html=export_html,
            html_path=html_path,
            threshold=residual_threshold,
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

    def evaluate_thresholds(
        self,
        model,
        features,
        thresholds=None,
        test_size=0.2,
        random_state=42,
    ):
        return get_evaluate_thresholds(
            df=self.df,
            model=model,
            features=features,
            target=self.target,
            thresholds=thresholds,
            test_size=test_size,
            random_state=random_state,
        )

    def compare_class_weight(
        self,
        model,
        features,
        cv=5,
        random_state=42,
    ):
        return get_compare_class_weight(
            df=self.df,
            model=model,
            features=features,
            target=self.target,
            cv=cv,
            random_state=random_state,
        )

    def analyze_prediction_group(
        self,
        model,
        features,
        threshold=0.5,
        group="FN",
        test_size=0.2,
        random_state=42,
    ):
        return get_analyze_prediction_group(
            df=self.df,
            model=model,
            features=features,
            target=self.target,
            threshold=threshold,
            group=group,
            test_size=test_size,
            random_state=random_state,
        )

    def feature_extraction(
        self,
        features=None,
        n_components=None,
    ):
        return get_feature_extraction(
            df=self.df,
            target=self.target,
            features=features,
            n_components=n_components,
        )
