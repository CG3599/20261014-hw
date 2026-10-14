
# ============================================================
# 50 STARTUPS - MULTIPLE LINEAR REGRESSION
# CRISP-DM + 5 FEATURE SELECTION METHODS
# Optimized Version
# ============================================================

import itertools
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, cross_val_predict
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
INNER_CV = KFold(n_splits=4, shuffle=True, random_state=42)
OUTER_CV = KFold(n_splits=5, shuffle=True, random_state=123)

NUMERIC = ["R&D Spend", "Administration", "Marketing Spend"]
FEATURES = ["State_New York", "State_California"] + NUMERIC


# ============================================================
# 1. BUSINESS UNDERSTANDING
# ============================================================

# Goal: Predict startup profit and identify useful features.


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

df = pd.read_csv("50_Startups.csv")

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())
print("\nStatistics:\n", df.describe())


# ============================================================
# 3. DATA PREPARATION
# ============================================================

df = df.dropna().copy()

X = df.drop(columns=["Profit"])
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "state",
                OneHotEncoder(
                    categories=[["Florida", "New York", "California"]],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                ["State"]
            ),
            ("numeric", "passthrough", NUMERIC)
        ],
        verbose_feature_names_out=False
    )


# ============================================================
# 4. MODELING - CUSTOM FEATURE SELECTOR
# ============================================================

class FeatureSelectionRegressor(
    RegressorMixin, BaseEstimator
):
    def __init__(self, method="EFS", cv=4, random_state=42,
                 n_features=None):
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.n_features = n_features

    def _cv(self):
        return KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

    def _score(self, X, y, indices):
        scores = cross_val_score(
            LinearRegression(),
            X[:, indices],
            y,
            cv=self._cv(),
            scoring="neg_root_mean_squared_error"
        )
        return -scores.mean()

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p = X.shape[1]
        all_idx = list(range(p))
        if self.n_features is not None and not 1 <= self.n_features <= p:
            raise ValueError("n_features must be between 1 and the feature count")

        if self.n_features == p or self.method == "Baseline":
            chosen = all_idx

        elif self.method == "EFS":
            sizes = (range(1, p + 1) if self.n_features is None
                     else [self.n_features])
            candidates = (
                combo
                for k in sizes
                for combo in itertools.combinations(all_idx, k)
            )
            chosen = list(min(
                candidates,
                key=lambda idx: self._score(X, y, list(idx))
            ))

        elif self.method == "RFECV":
            selector = (RFE(
                LinearRegression(), n_features_to_select=self.n_features
            ) if self.n_features is not None else RFECV(
                LinearRegression(),
                cv=self._cv(),
                scoring="neg_root_mean_squared_error",
                min_features_to_select=1
            ))
            selector.fit(X, y)
            chosen = list(np.flatnonzero(selector.support_))

        elif self.method == "LassoCV":
            selector = Pipeline([
                ("scale", StandardScaler()),
                ("lasso", LassoCV(
                    cv=self._cv(),
                    max_iter=100000,
                    random_state=self.random_state
                ))
            ])
            selector.fit(X, y)
            coef = selector.named_steps["lasso"].coef_
            chosen = list(np.flatnonzero(np.abs(coef) > 1e-8))

            if self.n_features is not None:
                # Fixed-count diagnostic: rank tuned, standardized Lasso weights.
                # Break zero/tied weights using standardized OLS weights.
                scaled = selector.named_steps["scale"].transform(X)
                tie_weights = np.abs(LinearRegression().fit(scaled, y).coef_)
                ranking = sorted(all_idx, key=lambda i: (
                    -abs(coef[i]), -tie_weights[i], i
                ))
                chosen = sorted(ranking[:self.n_features])

            if not chosen:
                chosen = [int(np.argmax(np.abs(coef)))]

        elif self.method == "Backward":
            chosen = all_idx.copy()

            target = self.n_features if self.n_features is not None else 1
            while len(chosen) > target:
                model = sm.OLS(
                    y,
                    sm.add_constant(X[:, chosen], has_constant="add")
                ).fit()

                pvalues = np.nan_to_num(
                    model.pvalues[1:],
                    nan=1.0
                )

                if self.n_features is None and pvalues.max() <= 0.05:
                    break

                chosen.pop(int(np.argmax(pvalues)))

        elif self.method == "SFS":
            best_score = float("inf")
            chosen = all_idx

            sizes = (range(1, p) if self.n_features is None
                     else [self.n_features])
            for k in sizes:
                selector = SequentialFeatureSelector(
                    LinearRegression(),
                    n_features_to_select=k,
                    direction="forward",
                    cv=self._cv(),
                    scoring="neg_root_mean_squared_error"
                )
                selector.fit(X, y)

                indices = list(np.flatnonzero(
                    selector.get_support()
                ))

                score = self._score(X, y, indices)

                if score < best_score:
                    best_score = score
                    chosen = indices

            if self.n_features is None:
                full_score = self._score(X, y, all_idx)
                if full_score < best_score:
                    chosen = all_idx

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.selected_indices_ = np.array(chosen, dtype=int)
        self.model_ = LinearRegression()
        self.model_.fit(X[:, self.selected_indices_], y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.model_.predict(
            X[:, self.selected_indices_]
        )


def build_pipeline(method):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("selector", FeatureSelectionRegressor(
            method=method,
            cv=4,
            random_state=RANDOM_STATE
        ))
    ])


# ============================================================
# 5. EVALUATION - NESTED CROSS VALIDATION
# ============================================================

methods = [
    "Baseline", "EFS", "RFECV",
    "LassoCV", "Backward", "SFS"
]

results = []
fitted_models = {}

for method in methods:
    print(f"\nEvaluating: {method}")

    pipeline = build_pipeline(method)

    # Outer CV: evaluates selection + regression together
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=OUTER_CV,
        scoring="neg_root_mean_squared_error"
    )

    # Fit using all training data
    pipeline.fit(X_train, y_train)

    selector = pipeline.named_steps["selector"]
    names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    selected_features = names[selector.selected_indices_]

    predictions = pipeline.predict(X_test)

    n = len(y_test)
    p = len(selected_features)
    r2 = r2_score(y_test, predictions)

    adjusted_r2 = (
        1 - (1 - r2) * (n - 1) / (n - p - 1)
        if n > p + 1 else np.nan
    )

    results.append({
        "Method": method,
        "Features": p,
        "Selected Features": ", ".join(selected_features),
        "CV_RMSE": -cv_scores.mean(),
        "CV_STD": cv_scores.std(),
        "Test_MAE": mean_absolute_error(y_test, predictions),
        "Test_RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "Test_R2": r2,
        "Adjusted_R2": adjusted_r2
    })

    fitted_models[method] = pipeline

    print("Selected:", list(selected_features))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV_RMSE")

print("\n========== COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# Export all five selection schemes, keeping the baseline as a reference.
results_df.to_csv("all_method_results.csv", index=False)
scheme_results = results_df[results_df["Method"] != "Baseline"].copy()
scheme_results.to_csv("scheme_performance.csv", index=False)
with open("scheme_performance.html", "w", encoding="utf-8") as output:
    output.write("""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Feature Selection Scheme Performance</title><style>
body { font: 16px Arial, sans-serif; margin: 28px; color: #171717; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 12px; text-align: left; }
th { border-bottom: 1px solid #222; white-space: nowrap; }
tbody tr:nth-child(odd) { background: #f6f6f6; }
td { vertical-align: top; }
p { color: #555; }
</style><h2>Performance of All Five Feature Selection Schemes</h2>
<p>Same 40 training rows and 10 test rows for every method.
Sorted by nested training CV RMSE (lower is better).
Test metrics use the held-out test set; higher R-squared is better.</p>
<div class="table-wrap">""" + scheme_results.to_html(
        index=False, border=0, float_format=lambda value: f"{value:.4f}"
    ) + "</div><h3>Baseline: All Five Features</h3>" + results_df[
        results_df["Method"] == "Baseline"
    ].to_html(index=False, border=0,
              float_format=lambda value: f"{value:.4f}") + "</html>")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for axis, metric, title in zip(
    axes, ["CV_RMSE", "Test_RMSE", "Test_R2"],
    ["Nested CV RMSE", "Test RMSE", "Test R-squared"]
):
    positions = np.arange(len(scheme_results))
    axis.bar(positions, scheme_results[metric])
    axis.set_xticks(positions, scheme_results["Method"], rotation=30)
    axis.set_title(title)
    axis.set_ylabel(title)
    axis.set_xlabel("Feature Selection Scheme")
    baseline = results_df.loc[results_df["Method"] == "Baseline", metric].iloc[0]
    axis.axhline(baseline, color="tab:red", linestyle="--",
                 label="Baseline (all features)")
    axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig("scheme_performance.png", dpi=180, bbox_inches="tight")
plt.show()


# ============================================================
# SELECT BEST METHOD USING TRAINING CV ONLY
# ============================================================

best_method = results_df.iloc[0]["Method"]
best_pipeline = fitted_models[best_method]

preprocessor = best_pipeline.named_steps["preprocessor"]
selector = best_pipeline.named_steps["selector"]

selected_names = preprocessor.get_feature_names_out()[
    selector.selected_indices_
]

regression = selector.model_

print("\nBest Method:", best_method)
print("Selected Features:", list(selected_names))

print("\nRegression Equation:")
equation = f"Profit = {regression.intercept_:.4f}"

for name, coef in zip(selected_names, regression.coef_):
    equation += f" + ({coef:.6f} * {name})"

print(equation)


# ============================================================
# VISUALIZATION
# ============================================================

# For each feature count, choose the subset using training CV only.
# Report held-out test metrics; the test set never chooses the subset.
feature_count_results = []
for count in range(1, len(FEATURES) + 1):
    count_pipeline = build_pipeline("EFS")
    count_pipeline.set_params(selector__n_features=count)
    count_pipeline.fit(X_train, y_train)
    count_selector = count_pipeline.named_steps["selector"]
    count_names = count_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()[count_selector.selected_indices_]
    count_predictions = count_pipeline.predict(X_test)
    feature_count_results.append({
        "Number of Features": count,
        "Selected Features": "[" + ", ".join(count_names) + "]",
        "RMSE": np.sqrt(mean_squared_error(y_test, count_predictions)),
        "R-squared": r2_score(y_test, count_predictions)
    })

feature_count_df = pd.DataFrame(feature_count_results)
print("\n========== RESULTS BY NUMBER OF FEATURES ==========")
print("Subsets selected by training CV; RMSE and R-squared on test data.")
print(feature_count_df.to_string(index=False, float_format="%.6f"))
feature_count_df.to_csv("feature_count_results.csv", index=False)

table_html = feature_count_df.to_html(
    index=False, border=0, float_format=lambda value: f"{value:.6f}"
)
with open("feature_count_results.html", "w", encoding="utf-8") as output:
    output.write("""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Results by Number of Features</title><style>
body { font: 16px Arial, sans-serif; margin: 28px; color: #171717; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 12px; text-align: left; }
th { border-bottom: 1px solid #222; }
tbody tr:nth-child(odd) { background: #f6f6f6; }
td:nth-child(3), td:nth-child(4) { text-align: right; }
p { color: #555; }
</style><h2>Results by Number of Features</h2>
<p>Subsets selected using training cross-validation.
RMSE and R-squared evaluated on the held-out test set.</p>"""
                 + table_html + "</html>")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for axis, metric, title in zip(
    axes, ["RMSE", "R-squared"],
    ["RMSE by Number of Features", "R-squared by Number of Features"]
):
    axis.plot(feature_count_df["Number of Features"],
              feature_count_df[metric], marker="o")
    axis.set_xlabel("Number of Features")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.set_xticks(feature_count_df["Number of Features"])
fig.tight_layout()
fig.savefig("feature_count_comparison.png", dpi=180, bbox_inches="tight")
plt.show()

# Fixed-count performance curves for every selection scheme.
curve_notes = {
    "EFS": "Exhaustive search: best training-CV subset at each feature count.",
    "RFECV": "Fixed-count RFE elimination path underlying RFECV; the original "
             "RFECV comparison still chooses its count automatically.",
    "LassoCV": "Fixed-count ranking by absolute standardized LassoCV coefficients; "
               "tied/zero coefficients ranked by absolute standardized OLS weights. "
               "Each selected subset is refitted with ordinary linear regression.",
    "Backward": "Remove the largest OLS p-value until the requested count remains; "
                "the 0.05 stopping rule applies only to the automatic comparison.",
    "SFS": "Forward sequential selection using training CV at each requested count."
}
all_curves = []
report_sections = []
for method, note in curve_notes.items():
    if method == "EFS":
        count_df = feature_count_df.copy()
    else:
        rows = []
        for count in range(1, len(FEATURES) + 1):
            pipeline = build_pipeline(method)
            pipeline.set_params(selector__n_features=count)
            pipeline.fit(X_train, y_train)
            selected = pipeline.named_steps["selector"].selected_indices_
            assert len(selected) == count
            names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            predictions = pipeline.predict(X_test)
            rows.append({
                "Number of Features": count,
                "Selected Features": "[" + ", ".join(names[selected]) + "]",
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R-squared": r2_score(y_test, predictions)
            })
        count_df = pd.DataFrame(rows)

    count_df.to_csv(f"{method}_feature_count_results.csv", index=False)
    all_curves.append(count_df.assign(Method=method))
    print(f"\n========== {method}: 1-5 FEATURES ==========")
    print(note)
    print(count_df.to_string(index=False, float_format="%.6f"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, metric, title in zip(
        axes, ["RMSE", "R-squared"],
        ["RMSE by Number of Features", "R-squared by Number of Features"]
    ):
        axis.plot(count_df["Number of Features"], count_df[metric], marker="o")
        axis.set_xlabel("Number of Features")
        axis.set_ylabel(metric)
        axis.set_title(title)
        axis.set_xticks(range(1, len(FEATURES) + 1))
    fig.suptitle(method + " - Fixed Feature Count Comparison")
    fig.tight_layout()
    image_name = f"{method}_feature_count_comparison.png"
    fig.savefig(image_name, dpi=180, bbox_inches="tight")
    plt.show()
    report_sections.append(
        f'<section id="{method}"><h2>{method}</h2><p>{note}</p>'
        + count_df.to_html(index=False, border=0,
                           float_format=lambda value: f"{value:.6f}")
        + f'<img src="{image_name}" alt="{method} RMSE and R-squared curves">'
        + '</section>'
    )

all_curves_df = pd.concat(all_curves, ignore_index=True)
assert len(all_curves_df) == 25
full_rows = all_curves_df[all_curves_df["Number of Features"] == len(FEATURES)]
assert np.allclose(full_rows["RMSE"], full_rows["RMSE"].iloc[0])
assert np.allclose(full_rows["R-squared"], full_rows["R-squared"].iloc[0])
all_curves_df.to_csv("all_schemes_feature_count_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for method, group in all_curves_df.groupby("Method", sort=False):
    for axis, metric in zip(axes, ["RMSE", "R-squared"]):
        axis.plot(group["Number of Features"], group[metric], marker="o", label=method)
for axis, metric in zip(axes, ["RMSE", "R-squared"]):
    axis.set_title(f"{metric} by Number of Features - All Schemes")
    axis.set_xlabel("Number of Features")
    axis.set_ylabel(metric)
    axis.set_xticks(range(1, len(FEATURES) + 1))
    axis.legend()
fig.tight_layout()
fig.savefig("all_schemes_feature_count_comparison.png", dpi=180,
            bbox_inches="tight")
plt.show()

with open("all_schemes_feature_count_results.html", "w", encoding="utf-8") as output:
    output.write("""<!doctype html><html lang="en"><meta charset="utf-8">
<title>All Five Schemes: 1-5 Features</title><style>
body { font: 16px Arial, sans-serif; margin: 28px; color: #171717; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 12px; text-align: left; }
th { border-bottom: 1px solid #222; }
tbody tr:nth-child(odd) { background: #f6f6f6; }
td:nth-child(3), td:nth-child(4) { text-align: right; }
img { width: 100%; max-width: 1200px; display: block; margin: 20px 0; }
p { color: #555; line-height: 1.5; }
section { margin: 40px 0; }
</style><h1>All Five Schemes: Performance by Number of Features</h1>
<p>Each method has five rows and two curves. Feature selection uses training
data only. RMSE and R-squared use the same 10-row test set (80/20 split,
random_state=42). These curves are diagnostics; they do not select the saved model.
Florida is the reference state, leaving five encoded predictors.</p>
<nav><a href="#EFS">EFS</a> | <a href="#RFECV">RFECV</a> |
<a href="#LassoCV">LassoCV</a> | <a href="#Backward">Backward</a> |
<a href="#SFS">SFS</a></nav>
<img src="all_schemes_feature_count_comparison.png" alt="All schemes compared">
""" + "".join(report_sections) + "</html>")

plt.figure(figsize=(9, 5))
plt.bar(results_df["Method"], results_df["CV_RMSE"])
plt.ylabel("Nested CV RMSE")
plt.xlabel("Feature Selection Method")
plt.title("50 Startups - Feature Selection Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

predictions = best_pipeline.predict(X_test)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions)

low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title(f"Actual vs Predicted - {best_method}")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 6. DEPLOYMENT
# ============================================================

joblib.dump(best_pipeline, "startup_best_model.pkl")

new_startup = pd.DataFrame({
    "R&D Spend": [150000],
    "Administration": [120000],
    "Marketing Spend": [200000],
    "State": ["New York"]
})

predicted_profit = best_pipeline.predict(new_startup)[0]

print("\nPredicted Profit:", round(predicted_profit, 2))
print("Saved: startup_best_model.pkl")
