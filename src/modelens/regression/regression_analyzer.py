import pandas as pd

from modelens.utils.html_reporter import export_dataframe_to_html

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


    def compare_feature_sets(
        self,
        model,
        feature_sets: dict,
        cv=5,
        export_html=False,
        file_name="feature_sets_comparison",
        random_state=42,
    ):
        import time
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from sklearn.base import clone
        from sklearn.model_selection import KFold, cross_validate

        kfold = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        results = {}

        # -----------------------------------
        # Evaluate Feature Sets
        # -----------------------------------

        for name, features in feature_sets.items():

            X = self.df[features]
            y = self.df[self.target]

            scores = cross_validate(
                clone(model),
                X,
                y,
                cv=kfold,
                scoring={
                    "r2": "r2",
                    "rmse": "neg_root_mean_squared_error",
                },
                return_train_score=True,
            )

            # -----------------------------------
            # Mean Values
            # -----------------------------------

            train_r2_mean = scores["train_r2"].mean()
            test_r2_mean = scores["test_r2"].mean()

            train_rmse_mean = -scores["train_rmse"].mean()
            test_rmse_mean = -scores["test_rmse"].mean()

            fit_time_mean = scores["fit_time"].mean()
            fit_time_std = scores["fit_time"].std()

            # -----------------------------------
            # Prediction Time
            # -----------------------------------

            predict_times = []

            for train_index, test_index in kfold.split(X):

                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_test = X.iloc[test_index]

                fold_model = clone(model)

                fold_model.fit(
                    X_train,
                    y_train,
                )

                # Warm-up
                fold_model.predict(X_test.iloc[:1])

                start = time.perf_counter()

                fold_model.predict(X_test)

                elapsed = time.perf_counter() - start

                predict_times.append(elapsed)

            predict_time_mean = float(
                np.mean(predict_times)
            )

            predict_time_std = float(
                np.std(predict_times)
            )

            # -----------------------------------
            # Gaps
            # -----------------------------------

            r2_gap = (
                train_r2_mean
                - test_r2_mean
            )

            rmse_gap = (
                test_rmse_mean
                - train_rmse_mean
            )

            # -----------------------------------
            # Result
            # -----------------------------------

            results[name] = {
                "Features": len(features),

                "R2": (
                    f"Train: {train_r2_mean:.3f} ± "
                    f"{scores['train_r2'].std():.3f}\n"
                    f"Test: {test_r2_mean:.3f} ± "
                    f"{scores['test_r2'].std():.3f}\n"
                    f"Gap: {r2_gap:.3f}"
                ),

                "RMSE": (
                    f"Train: {train_rmse_mean:.3f} ± "
                    f"{scores['train_rmse'].std():.3f}\n"
                    f"Test: {test_rmse_mean:.3f} ± "
                    f"{scores['test_rmse'].std():.3f}\n"
                    f"Gap: {rmse_gap:.3f}"
                ),

                # Internal Values
                "_Test R2 Mean": test_r2_mean,
                "_Test RMSE Mean": test_rmse_mean,

                "_Fit Time Mean": fit_time_mean,
                "_Fit Time Std": fit_time_std,

                "_Predict Time Mean": predict_time_mean,
                "_Predict Time Std": predict_time_std,
            }

        result = pd.DataFrame(results).T

        result.index.name = "Feature Set"

        # -----------------------------------
        # Sort by Test R2
        # -----------------------------------

        result = result.sort_values(
            by="_Test R2 Mean",
            ascending=False,
        )

        # -----------------------------------
        # Best Row
        # -----------------------------------

        best_test_r2 = float(
            result.iloc[0]["_Test R2 Mean"]
        )

        best_test_rmse = float(
            result.iloc[0]["_Test RMSE Mean"]
        )

        best_fit_time = float(
            result.iloc[0]["_Fit Time Mean"]
        )

        best_predict_time = float(
            result.iloc[0]["_Predict Time Mean"]
        )

        # -----------------------------------
        # Test R2 Loss
        # -----------------------------------

        r2_values = (
            result["_Test R2 Mean"]
            .astype(float)
        )

        r2_loss = (
            best_test_r2
            - r2_values
        )

        r2_loss_percent = (
            r2_loss
            / best_test_r2
            * 100
        )

        result["Test R2 Loss"] = [
            f"{percent:.2f}% ({loss:.3f})"
            for percent, loss in zip(
                r2_loss_percent,
                r2_loss,
            )
        ]

        # -----------------------------------
        # Test RMSE Loss
        # -----------------------------------

        rmse_values = (
            result["_Test RMSE Mean"]
            .astype(float)
        )

        rmse_loss = (
            rmse_values
            - best_test_rmse
        )

        rmse_loss_percent = (
            rmse_loss
            / best_test_rmse
            * 100
        )

        result["Test RMSE Loss"] = [
            f"{percent:.2f}% ({loss:.3f})"
            for percent, loss in zip(
                rmse_loss_percent,
                rmse_loss,
            )
        ]

        # -----------------------------------
        # Fit / Predict Time
        # -----------------------------------

        result["Fit / Predict Time (s)"] = [
            (
                f"Fit: {fit_mean:.3f} ± "
                f"{fit_std:.3f}\n"
                f"Predict: {predict_mean:.3f} ± "
                f"{predict_std:.3f}"
            )
            for (
                fit_mean,
                fit_std,
                predict_mean,
                predict_std,
            ) in zip(
                result["_Fit Time Mean"],
                result["_Fit Time Std"],
                result["_Predict Time Mean"],
                result["_Predict Time Std"],
            )
        ]

        # -----------------------------------
        # Speed vs Best
        # -----------------------------------

        speed_values = []

        for index in result.index:

            fit_time = float(
                result.loc[
                    index,
                    "_Fit Time Mean",
                ]
            )

            predict_time = float(
                result.loc[
                    index,
                    "_Predict Time Mean",
                ]
            )

            if fit_time <= best_fit_time:
                fit_speed = (
                    best_fit_time
                    / fit_time
                )
            else:
                fit_speed = -(
                    fit_time
                    / best_fit_time
                )

            if predict_time <= best_predict_time:
                predict_speed = (
                    best_predict_time
                    / predict_time
                )
            else:
                predict_speed = -(
                    predict_time
                    / best_predict_time
                )

            speed_values.append(
                f"Predict: {predict_speed:.2f}x\n"
                f"Fit: {fit_speed:.2f}x"
            )

        result["Speed vs Best"] = speed_values

        # -----------------------------------
        # Final Columns
        # -----------------------------------

        result = result[
            [
                "Features",
                "R2",
                "RMSE",
                "Test R2 Loss",
                "Test RMSE Loss",
                "Fit / Predict Time (s)",
                "Speed vs Best",
            ]
        ]

        # -----------------------------------
        # HTML
        # -----------------------------------

        if export_html:

            html_path = (
                "html_reports/"
                "compare_feature_sets/"
                f"{file_name}.html"
            )

            export_dataframe_to_html(
                df=result,
                path=Path(html_path),
                title="Feature Sets Comparison",
            )

            return (
                f"✓ Report generated successfully: "
                f"{html_path}"
            )

        return result
