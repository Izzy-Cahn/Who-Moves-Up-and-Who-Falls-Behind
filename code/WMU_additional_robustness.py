"""
WHO MOVES UP AND WHO FALLS BEHIND
Additional robustness and deeper wealth analyses
=================================================

This script is designed to be run AFTER the main replication script, but it is
standalone with respect to the raw PSID .dta files.  

ANALYSIS 1. Flexible econometric benchmarks for the main benchmark outcome
--------------------------------------------------------------------------
We compare:
    1. Income-only linear logit
    2. Income-only spline logit
    3. Income + wealth additive spline logit
    4. Full-background linear logit
    5. Full-background additive spline logit
    6. Full-background spline logit with an income x wealth spline interaction

All models use five-fold FAMILY-GROUPED cross-fitting by ID68.  Continuous
nonlinearities are estimated with cubic spline bases fit on training data only.
The full additive spline specification allows flexible nonlinearities in income,
wealth, and age but no interactions; the final specification adds only an
income x wealth tensor-product spline interaction.  This helps distinguish:
    - gains from adding wealth,
    - gains from nonlinearities,
    - gains from an income-wealth interaction,
    - gains that remain specific to tree-based models.

The script also validates the richer spline-logit predictions against realized
mobility while holding the income-only spline prediction approximately fixed.

ANALYSIS 2. Efficient uncertainty for income/wealth importance
---------------------------------------------------------------
For relative UPWARD mobility (+10, +20, +30, +40 percentile points), the
script estimates full-background Random Forests on a COMMON-SUPPORT SAMPLE:
parents must begin at or below the 60th percentile, so every observation is
mechanically capable of every one of the four upward movements.  The exact
same rows and outer folds are used at every threshold.

We focus on grouped permutation importance for INCOME and WEALTH.  Rather than
refitting the entire forest hundreds or thousands of times, we:
    1. fit/tune the cross-fitted models once;
    2. save held-out baseline predictions;
    3. save several held-out predictions after permuting income or wealth;
    4. resample PSID family clusters from those held-out prediction objects.

This "fixed-model held-out cluster bootstrap" is fast.  It measures uncertainty
in held-out predictive importance conditional on the fitted cross-fitted models;
it is NOT a full model-refitting bootstrap.  

ANALYSIS 3. Deeper look at wealth and increasingly large upward movements
-------------------------------------------------------------------------
Using the same common-support sample, the script reports:
    - income and wealth permutation importance at +10/+20/+30/+40;
    - the wealth share of combined income+wealth importance;
    - paired bootstrap changes from +10 to +40;
    - rank-standardized realized mobility for low- vs high-wealth families;
    - wealth gradients separately by parental starting-rank band.

The direct realized-outcome standardization is deliberately non-ML: it asks
whether the relationship between wealth and increasingly demanding upward
movements is also visible directly in realized outcomes after holding the
parental starting-rank distribution approximately fixed.


Main outputs
------------
robustness_flexible_econometric_models.csv
robustness_flexible_econometric_metric_differences.csv
robustness_flexible_logit_revision_validation.csv
figure_robustness_flexible_econometric_models.pdf
figure_robustness_flexible_logit_revision_validation.pdf

robustness_relative_common_support_permutation_importance.csv
robustness_relative_common_support_trend_tests.csv
robustness_relative_common_support_tuning.csv
figure_robustness_income_wealth_importance_thresholds.pdf

robustness_standardized_wealth_gradient.csv
robustness_wealth_by_starting_rank.csv
figure_robustness_standardized_wealth_gradient.pdf
figure_robustness_wealth_gradient_by_starting_rank.pdf

Runtime
-------
The expensive part is Analysis 2 because each of four mobility thresholds uses
nested Random-Forest tuning.  The bootstrap itself is cheap because forests are
NOT re-estimated inside bootstrap replications.
"""

# ============================================================
# 0. IMPORTS AND SETTINGS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 12345
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

# Number of held-out permutations for income and wealth.
# 10 is usually enough for a stable mean and keeps runtime modest.
N_IMPORTANCE_PERMUTATIONS = 10

# Fast bootstrap: NO model refitting.  Increase to 500 if desired after the
# results are stable; 250 is a good paper-development compromise.
EVAL_BOOT_REPS = 250

# Direct realized-outcome standardization is very cheap, so we can use more.
STANDARDIZATION_BOOT_REPS = 500

# Flexible-logit metric and revision-validation bootstrap.
ECON_BOOT_REPS = 500

# Spline settings.  Cubic splines with 5 knots are deliberately moderate for
# a sample of roughly 2,700 observations.
SPLINE_KNOTS = 5
SPLINE_DEGREE = 3

DATA_DIR = (
    "/Users/yisroelcahn/Library/Mobile Documents/"
    "com~apple~CloudDocs/Documents/Who Moves Up/Data"
)

RESULTS_DIR = (
    "/Users/yisroelcahn/Library/Mobile Documents/"
    "com~apple~CloudDocs/Documents/Who Moves Up/Results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

BENCHMARK_FILE = "psidcleaned_data3.dta"
RELATIVE_FILE = "psidcleaned_data1.dta"

# The same moderate grid used in the cleaned main analysis.
RF_GRID = {
    "n_estimators": [250, 400],
    "max_depth": [5, 10],
    "max_features": ["sqrt", 0.50],
    "min_samples_leaf": [5, 15],
}


# ============================================================
# 1. VARIABLE NAMES AND FEATURE SETS
# ============================================================

RENAME_PARENT = {
    "page1991": "Age",
    "pselfemployed91": "Self-employed",
    "pfemale91": "Female",

    "poccTechnical": "Occupation = Technical",
    "poccManager": "Occupation = Manager",
    "poccSales": "Occupation = Sales",
    "poccClerical": "Occupation = Clerical",
    "poccCraftsman": "Occupation = Craftsman",
    "poccOperatives": "Occupation = Operatives",
    "poccTransport": "Occupation = Transport",
    "poccLaborers": "Occupation = Laborers",
    "poccFarmers": "Occupation = Farmers",
    "poccFarmLaborers": "Occupation = Farm Laborers",
    "poccService": "Occupation = Service",
    "poccPrivate": "Occupation = Private",

    "pindAgriculture": "Industry = Agriculture",
    "pindMining": "Industry = Mining",
    "pindConstruction": "Industry = Construction",
    "pindManufacturing": "Industry = Manufacturing",
    "pindTransportation": "Industry = Transportation",
    "pindRetailTrade": "Industry = Retail Trade",
    "pindFinance": "Industry = Finance",
    "pindBusiness": "Industry = Business",
    "pindPersonal": "Industry = Personal",
    "pindEntertainment": "Industry = Entertainment",
    "pindProfessional": "Industry = Professional",
    "pindPublic": "Industry = Public Admin",

    "pBlack": "African-American",
    "pWhite": "White",

    "pNortheast": "Northeast",
    "pNorthCentral": "Northcentral",
    "pSouth": "South",
    "pWest": "West",

    "psomeHS": "Some High School",
    "pHS": "High School",
    "psomeCollege": "Some College",
    "pCollege": "College",

    "pmarried": "Married",

    "wealth89": "Wealth",
    "wage8594_p": "Wage Percentile",
    "wageL8594": "Wage",
    "wageL0919": "Child Wage",
}

INCOME_COLS = ["Wage"]
WEALTH_COLS = ["Wealth"]

EDUCATION_COLS = [
    "Some High School",
    "High School",
    "Some College",
    "College",
]

DEMOGRAPHIC_COLS = [
    "Age",
    "Female",
    "African-American",
    "Married",
]

EMPLOYMENT_COLS = ["Self-employed"]

REGION_COLS = [
    "Northeast",
    "Northcentral",
    "South",
    "West",
]

OCCUPATION_COLS = [
    "Occupation = Technical",
    "Occupation = Manager",
    "Occupation = Sales",
    "Occupation = Clerical",
    "Occupation = Craftsman",
    "Occupation = Operatives",
    "Occupation = Transport",
    "Occupation = Laborers",
    "Occupation = Farmers",
    "Occupation = Farm Laborers",
    "Occupation = Service",
    "Occupation = Private",
]

INDUSTRY_COLS = [
    "Industry = Agriculture",
    "Industry = Mining",
    "Industry = Construction",
    "Industry = Manufacturing",
    "Industry = Transportation",
    "Industry = Retail Trade",
    "Industry = Finance",
    "Industry = Business",
    "Industry = Personal",
    "Industry = Entertainment",
    "Industry = Professional",
    "Industry = Public Admin",
]

FULL_FEATURES = (
    INCOME_COLS
    + WEALTH_COLS
    + EDUCATION_COLS
    + DEMOGRAPHIC_COLS
    + EMPLOYMENT_COLS
    + REGION_COLS
    + OCCUPATION_COLS
    + INDUSTRY_COLS
)

# All dummy/binary features other than the three continuous variables treated
# flexibly in the spline-logit specifications.
FULL_DISCRETE_FEATURES = [
    x for x in FULL_FEATURES
    if x not in ["Wage", "Wealth", "Age"]
]


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def load_psid(filename):
    df = pd.read_stata(
        os.path.join(DATA_DIR, filename),
        convert_categoricals=False,
    )

    available = {
        old: new for old, new in RENAME_PARENT.items()
        if old in df.columns
    }
    df = df.rename(columns=available)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def check_columns(df, columns, label):
    missing = [x for x in columns if x not in df.columns]
    if missing:
        raise KeyError(
            f"Missing variables needed for {label}:\n" + "\n".join(missing)
        )


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return np.average(values[ok], weights=weights[ok])


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.atleast_1d(quantiles).astype(float)

    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]

    if len(values) == 0:
        out = np.full(len(quantiles), np.nan)
        return out[0] if len(out) == 1 else out

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative = cumulative / weights.sum()
    out = np.interp(quantiles, cumulative, values)

    if len(out) == 1:
        return float(out[0])
    return out


def assign_weighted_quantile_number(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    cuts = weighted_quantile(values, weights, np.arange(1, q) / q)
    cuts = np.asarray(cuts, dtype=float)
    cuts = np.unique(cuts[np.isfinite(cuts)])

    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    out[ok] = np.digitize(values[ok], cuts, right=True) + 1
    return out


def weighted_brier(y, p, w):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    return weighted_mean((y - p) ** 2, w)


def weighted_accuracy(y, p, w, threshold=0.50):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return weighted_mean((p >= threshold).astype(int) == y, w)


def weighted_metrics(y, p, w):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)

    p = np.clip(p, 1e-8, 1 - 1e-8)

    return {
        "AUC": roc_auc_score(y, p, sample_weight=w),
        "Brier": weighted_brier(y, p, w),
        "LogLoss": log_loss(y, p, sample_weight=w, labels=[0, 1]),
        "Accuracy": weighted_accuracy(y, p, w),
    }


def cluster_bootstrap_indices(groups, rng):
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)

    group_to_rows = {
        g: np.flatnonzero(groups == g)
        for g in unique_groups
    }

    return np.concatenate([group_to_rows[g] for g in sampled])


def make_cluster_bootstrap_indices(groups, reps, seed):
    rng = np.random.default_rng(seed)
    return [cluster_bootstrap_indices(groups, rng) for _ in range(reps)]


def percentile_ci(draws, lower=2.5, upper=97.5):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 20:
        return np.nan, np.nan
    return tuple(np.percentile(draws, [lower, upper]))


def safe_weighted_auc(y, p, w):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


# ============================================================
# 3. FAMILY-GROUPED FOLD HELPERS
# ============================================================

def make_outer_folds(X, y, groups, seed=RANDOM_STATE):
    cv = StratifiedGroupKFold(
        n_splits=N_OUTER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    return list(cv.split(X, y, groups=groups))


def make_rf(params, seed):
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        criterion="gini",
        **params,
    )


def weighted_inner_rf_search(X, y, w, groups, seed):
    """
    Weighted, family-grouped inner CV.  This is used only for Analysis 2.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    groups = np.asarray(groups)

    cv = StratifiedGroupKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    rows = []
    for config_number, params in enumerate(ParameterGrid(RF_GRID), start=1):
        scores = []

        for inner_fold, (fit_idx, valid_idx) in enumerate(
            cv.split(X, y, groups=groups), start=1
        ):
            imputer = SimpleImputer(strategy="median")
            X_fit = imputer.fit_transform(X.iloc[fit_idx])
            X_valid = imputer.transform(X.iloc[valid_idx])

            model = make_rf(
                params,
                seed + 1000 * config_number + inner_fold,
            )
            model.fit(X_fit, y[fit_idx], sample_weight=w[fit_idx])
            p = model.predict_proba(X_valid)[:, 1]

            scores.append(
                roc_auc_score(
                    y[valid_idx],
                    p,
                    sample_weight=w[valid_idx],
                )
            )

        rows.append(
            {
                "Config": config_number,
                "Parameters": params,
                "MeanWeightedInnerAUC": float(np.mean(scores)),
                "SDInnerAUC": float(np.std(scores, ddof=1)),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "MeanWeightedInnerAUC", ascending=False
    ).reset_index(drop=True)

    return result.loc[0, "Parameters"], result


# ============================================================
# 4. FLEXIBLE ECONOMETRIC DESIGN MATRICES
# ============================================================

class FlexibleLogitDesign:
    """
    Training-fold-only construction of a spline-logit design matrix.

    Specifications:
        linear_income
        spline_income
        spline_income_wealth
        linear_full
        spline_full_additive
        spline_full_income_wealth_interaction
    """

    def __init__(self, specification):
        self.specification = specification
        self.raw_imputer = None
        self.wage_spline = None
        self.wealth_spline = None
        self.age_spline = None
        self.scaler = None
        self.columns_used = None

    @staticmethod
    def _ihs_wealth(x):
        # $100,000 scaling keeps numerical magnitudes convenient.  Because the
        # spline knots are quantile-based, results are not driven by units.
        return np.arcsinh(np.asarray(x, dtype=float) / 100000.0)

    def _needed_columns(self):
        if self.specification in ["linear_income", "spline_income"]:
            return ["Wage"]

        if self.specification == "spline_income_wealth":
            return ["Wage", "Wealth"]

        if self.specification in [
            "linear_full",
            "spline_full_additive",
            "spline_full_income_wealth_interaction",
        ]:
            return FULL_FEATURES

        raise ValueError(f"Unknown specification: {self.specification}")

    def fit_transform(self, df):
        self.columns_used = self._needed_columns()
        self.raw_imputer = SimpleImputer(strategy="median")

        raw = self.raw_imputer.fit_transform(df[self.columns_used])
        work = pd.DataFrame(raw, columns=self.columns_used, index=df.index)

        X = self._build_features(work, fit=True)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        return Xs

    def transform(self, df):
        raw = self.raw_imputer.transform(df[self.columns_used])
        work = pd.DataFrame(raw, columns=self.columns_used, index=df.index)

        X = self._build_features(work, fit=False)
        return self.scaler.transform(X)

    def _fit_or_transform_spline(self, values, which, fit):
        values = np.asarray(values, dtype=float).reshape(-1, 1)

        if which == "wage":
            attr = "wage_spline"
        elif which == "wealth":
            attr = "wealth_spline"
        elif which == "age":
            attr = "age_spline"
        else:
            raise ValueError(which)

        transformer = getattr(self, attr)

        if fit:
            transformer = SplineTransformer(
                n_knots=SPLINE_KNOTS,
                degree=SPLINE_DEGREE,
                include_bias=False,
                knots="quantile",
                extrapolation="linear",
            )
            out = transformer.fit_transform(values)
            setattr(self, attr, transformer)
            return out

        return transformer.transform(values)

    def _build_features(self, work, fit):
        spec = self.specification

        if spec == "linear_income":
            return work[["Wage"]].to_numpy(dtype=float)

        if spec == "spline_income":
            return self._fit_or_transform_spline(
                work["Wage"], "wage", fit
            )

        if spec == "spline_income_wealth":
            wage_basis = self._fit_or_transform_spline(
                work["Wage"], "wage", fit
            )
            wealth_basis = self._fit_or_transform_spline(
                self._ihs_wealth(work["Wealth"]), "wealth", fit
            )
            return np.column_stack([wage_basis, wealth_basis])

        if spec == "linear_full":
            return work[FULL_FEATURES].to_numpy(dtype=float)

        if spec in [
            "spline_full_additive",
            "spline_full_income_wealth_interaction",
        ]:
            wage_basis = self._fit_or_transform_spline(
                work["Wage"], "wage", fit
            )
            wealth_basis = self._fit_or_transform_spline(
                self._ihs_wealth(work["Wealth"]), "wealth", fit
            )
            age_basis = self._fit_or_transform_spline(
                work["Age"], "age", fit
            )

            discrete = work[FULL_DISCRETE_FEATURES].to_numpy(dtype=float)

            pieces = [wage_basis, wealth_basis, age_basis, discrete]

            if spec == "spline_full_income_wealth_interaction":
                # Tensor-product basis: every wage spline basis term interacted
                # with every wealth spline basis term.
                interaction = np.einsum(
                    "ij,ik->ijk", wage_basis, wealth_basis
                ).reshape(len(work), -1)
                pieces.append(interaction)

            return np.column_stack(pieces)

        raise ValueError(spec)


def crossfit_flexible_logit(df, y, weights, groups, outer_folds, specification):
    pred = np.full(len(df), np.nan)

    for fold_number, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        design = FlexibleLogitDesign(specification)

        X_train = design.fit_transform(df.iloc[train_idx])
        X_test = design.transform(df.iloc[test_idx])

        # Very weak L2 regularization approximates an unpenalized logit while
        # avoiding numerical failures with spline and interaction bases.
        model = LogisticRegression(
            C=1e4,
            solver="lbfgs",
            max_iter=10000,
            random_state=RANDOM_STATE + 100 * fold_number,
        )

        model.fit(
            X_train,
            y[train_idx],
            sample_weight=weights[train_idx],
        )

        pred[test_idx] = model.predict_proba(X_test)[:, 1]

    if np.isnan(pred).any():
        raise RuntimeError(
            f"Missing out-of-fold predictions for {specification}"
        )

    return pred


# ============================================================
# 5. ANALYSIS 1: FLEXIBLE ECONOMETRIC BENCHMARKS
# ============================================================

print("\n" + "=" * 100)
print("ANALYSIS 1: FLEXIBLE ECONOMETRIC BENCHMARKS")
print("=" * 100)

benchmark = load_psid(BENCHMARK_FILE)

check_columns(
    benchmark,
    FULL_FEATURES + ["Child Wage", "weight1991", "ID68"],
    "flexible econometric benchmark",
)

parent_median = weighted_quantile(
    benchmark["Wage"],
    benchmark["weight1991"],
    0.50,
)

benchmark["MedianMobility"] = np.where(
    benchmark["Child Wage"].notna(),
    (benchmark["Child Wage"] >= parent_median).astype(float),
    np.nan,
)

econ = benchmark.loc[
    benchmark["MedianMobility"].notna()
    & benchmark["weight1991"].notna()
    & (benchmark["weight1991"] > 0)
    & benchmark["ID68"].notna()
].copy().reset_index(drop=True)

y_econ = econ["MedianMobility"].astype(int).to_numpy()
w_econ = econ["weight1991"].astype(float).to_numpy()
g_econ = econ["ID68"].to_numpy()

outer_folds_econ = make_outer_folds(
    econ[FULL_FEATURES],
    y_econ,
    g_econ,
    seed=RANDOM_STATE,
)

econ_specs = {
    "Income-only linear logit": "linear_income",
    "Income-only spline logit": "spline_income",
    "Income + wealth spline logit": "spline_income_wealth",
    "Full-background linear logit": "linear_full",
    "Full-background additive spline logit": "spline_full_additive",
    "Full-background spline logit + income x wealth interaction":
        "spline_full_income_wealth_interaction",
}

econ_predictions = {}
for label, spec in econ_specs.items():
    print(f"   Fitting: {label}")
    econ_predictions[label] = crossfit_flexible_logit(
        econ,
        y_econ,
        w_econ,
        g_econ,
        outer_folds_econ,
        specification=spec,
    )
    econ[f"Pred_{len(econ_predictions)}"] = econ_predictions[label]

# Optionally bring the already-estimated RF/GB rows into the comparison table.
# This avoids re-running the expensive main models.
existing_table5_path = os.path.join(
    RESULTS_DIR, "table5_model_performance.csv"
)

rows = []
for label, pred in econ_predictions.items():
    rows.append({"Model": label, **weighted_metrics(y_econ, pred, w_econ)})

flexible_econ_table = pd.DataFrame(rows)

if os.path.exists(existing_table5_path):
    try:
        old = pd.read_csv(existing_table5_path)
        wanted = old.loc[
            old["Model"].isin(
                [
                    "Full-background Random Forest",
                    "Full-background Gradient Boosting",
                ]
            )
        ].copy()

        if not wanted.empty:
            # Normalize column names from the main script.
            rename_perf = {
                "Brier Score": "Brier",
                "Log Loss": "LogLoss",
            }
            wanted = wanted.rename(columns=rename_perf)

            keep = [
                x for x in ["Model", "AUC", "Brier", "LogLoss", "Accuracy"]
                if x in wanted.columns
            ]
            wanted = wanted[keep]

            flexible_econ_table = pd.concat(
                [flexible_econ_table, wanted],
                ignore_index=True,
                sort=False,
            )
    except Exception as exc:
        print("Could not append existing RF/GB performance rows:", exc)

flexible_econ_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_flexible_econometric_models.csv",
    ),
    index=False,
)

print("\nFlexible econometric benchmark performance:")
print(flexible_econ_table.round(4).to_string(index=False))


# ------------------------------------------------------------
# 5.1 Fast cluster-bootstrap metric differences
# ------------------------------------------------------------
# These intervals reuse the cross-fitted predictions and therefore do not
# refit the spline logits inside each bootstrap sample.

econ_boot_indices = make_cluster_bootstrap_indices(
    g_econ,
    ECON_BOOT_REPS,
    seed=RANDOM_STATE + 40000,
)

econ_pair_comparisons = [
    (
        "Income spline minus income linear",
        "Income-only spline logit",
        "Income-only linear logit",
    ),
    (
        "Income+wealth spline minus income spline",
        "Income + wealth spline logit",
        "Income-only spline logit",
    ),
    (
        "Full additive spline minus full linear",
        "Full-background additive spline logit",
        "Full-background linear logit",
    ),
    (
        "Income-wealth interaction minus full additive spline",
        "Full-background spline logit + income x wealth interaction",
        "Full-background additive spline logit",
    ),
]

metric_diff_rows = []

for comparison, richer_name, base_name in econ_pair_comparisons:
    richer = econ_predictions[richer_name]
    base = econ_predictions[base_name]

    point_r = weighted_metrics(y_econ, richer, w_econ)
    point_b = weighted_metrics(y_econ, base, w_econ)

    auc_draws = []
    brier_draws = []
    loss_draws = []

    for idx in econ_boot_indices:
        yy = y_econ[idx]
        ww = w_econ[idx]

        if len(np.unique(yy)) < 2:
            auc_draws.append(np.nan)
        else:
            auc_draws.append(
                safe_weighted_auc(yy, richer[idx], ww)
                - safe_weighted_auc(yy, base[idx], ww)
            )

        brier_draws.append(
            weighted_brier(yy, richer[idx], ww)
            - weighted_brier(yy, base[idx], ww)
        )

        loss_draws.append(
            log_loss(
                yy,
                np.clip(richer[idx], 1e-8, 1 - 1e-8),
                sample_weight=ww,
                labels=[0, 1],
            )
            - log_loss(
                yy,
                np.clip(base[idx], 1e-8, 1 - 1e-8),
                sample_weight=ww,
                labels=[0, 1],
            )
        )

    auc_ci = percentile_ci(auc_draws)
    brier_ci = percentile_ci(brier_draws)
    loss_ci = percentile_ci(loss_draws)

    metric_diff_rows.append(
        {
            "Comparison": comparison,
            "DeltaAUC": point_r["AUC"] - point_b["AUC"],
            "DeltaAUC_Lower": auc_ci[0],
            "DeltaAUC_Upper": auc_ci[1],
            "DeltaBrier": point_r["Brier"] - point_b["Brier"],
            "DeltaBrier_Lower": brier_ci[0],
            "DeltaBrier_Upper": brier_ci[1],
            "DeltaLogLoss": point_r["LogLoss"] - point_b["LogLoss"],
            "DeltaLogLoss_Lower": loss_ci[0],
            "DeltaLogLoss_Upper": loss_ci[1],
            "BootstrapType": "Fixed-prediction family-cluster bootstrap",
            "BootstrapReps": ECON_BOOT_REPS,
        }
    )

metric_differences = pd.DataFrame(metric_diff_rows)
metric_differences.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_flexible_econometric_metric_differences.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# 5.2 Economically interpretable validation of the spline logit
# ------------------------------------------------------------
# Hold the income-only spline prediction approximately fixed, then ask whether
# the richer additive-spline prediction identifies realized mobility differences.

income_spline_pred = econ_predictions["Income-only spline logit"]
full_additive_pred = econ_predictions["Full-background additive spline logit"]

econ["IncomeSplinePred"] = income_spline_pred
econ["FullAdditiveSplinePred"] = full_additive_pred
econ["SplineRevision"] = full_additive_pred - income_spline_pred

econ["IncomeSplineQuintile"] = assign_weighted_quantile_number(
    econ["IncomeSplinePred"],
    econ["weight1991"],
    5,
)

revision_validation_rows = []

for q in range(1, 6):
    temp = econ.loc[econ["IncomeSplineQuintile"] == q].copy()
    temp["RevisionTercile"] = assign_weighted_quantile_number(
        temp["SplineRevision"],
        temp["weight1991"],
        3,
    )

    for tercile in [1, 2, 3]:
        cell = temp.loc[temp["RevisionTercile"] == tercile].copy()
        if len(cell) == 0:
            continue

        point = weighted_mean(cell["MedianMobility"], cell["weight1991"])

        rng = np.random.default_rng(
            RANDOM_STATE + 50000 + 100 * q + tercile
        )
        draws = []
        for _ in range(ECON_BOOT_REPS):
            idx = cluster_bootstrap_indices(cell["ID68"], rng)
            z = cell.iloc[idx]
            draws.append(
                weighted_mean(z["MedianMobility"], z["weight1991"])
            )

        ci = percentile_ci(draws)

        revision_validation_rows.append(
            {
                "IncomeSplineQuintile": q,
                "RevisionTercile": tercile,
                "N": len(cell),
                "Families": cell["ID68"].nunique(),
                "RealizedMobility": point,
                "Lower": ci[0],
                "Upper": ci[1],
                "MeanIncomeSplinePrediction": weighted_mean(
                    cell["IncomeSplinePred"], cell["weight1991"]
                ),
                "MeanFullAdditivePrediction": weighted_mean(
                    cell["FullAdditiveSplinePred"], cell["weight1991"]
                ),
                "MeanRevision": weighted_mean(
                    cell["SplineRevision"], cell["weight1991"]
                ),
            }
        )

revision_validation = pd.DataFrame(revision_validation_rows)
revision_validation.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_flexible_logit_revision_validation.csv",
    ),
    index=False,
)

# Figure: flexible-logit model performance.
plot_perf = flexible_econ_table.loc[
    flexible_econ_table["Model"].isin(list(econ_specs.keys()))
].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

axes[0].barh(plot_perf["Model"], plot_perf["AUC"])
axes[0].set_xlabel("Weighted out-of-fold AUC")
axes[0].set_title("Discrimination")
axes[0].set_xlim(0.5, max(0.75, plot_perf["AUC"].max() + 0.02))

axes[1].barh(plot_perf["Model"], plot_perf["Brier"])
axes[1].set_xlabel("Weighted Brier score")
axes[1].set_title("Probability forecast error")
axes[1].invert_xaxis()

fig.suptitle(
    "Flexible Econometric Benchmarks for Upward Mobility",
    fontsize=13,
)
fig.subplots_adjust(left=0.36, right=0.97, bottom=0.16, top=0.84, wspace=0.32)
fig.savefig(
    os.path.join(
        RESULTS_DIR,
        "figure_robustness_flexible_econometric_models.pdf",
    ),
    bbox_inches="tight",
)
plt.show()

# Figure: realized outcomes by revision to income-only spline predictions.
fig, ax = plt.subplots(figsize=(8.6, 5.5))

for tercile, label in zip(
    [1, 2, 3],
    ["Low revision", "Middle revision", "High revision"],
):
    temp = revision_validation.loc[
        revision_validation["RevisionTercile"] == tercile
    ].sort_values("IncomeSplineQuintile")

    x = temp["IncomeSplineQuintile"].to_numpy()
    yv = temp["RealizedMobility"].to_numpy()
    lower = temp["Lower"].to_numpy()
    upper = temp["Upper"].to_numpy()

    ax.errorbar(
        x,
        yv,
        yerr=np.vstack([yv - lower, upper - yv]),
        marker="o",
        capsize=3,
        label=label,
    )

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["Lowest", "Q2", "Q3", "Q4", "Highest"])
ax.set_xlabel("Income-only spline predicted-mobility quintile")
ax.set_ylabel("Realized probability of upward mobility")
ax.set_ylim(0, 1)
ax.set_title(
    "Realized Mobility by Revisions to a Flexible Income-Only Logit"
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    os.path.join(
        RESULTS_DIR,
        "figure_robustness_flexible_logit_revision_validation.pdf",
    ),
    bbox_inches="tight",
)
plt.show()


# ============================================================
# 6. ANALYSIS 2: COMMON-SUPPORT RELATIVE UPWARD MOBILITY
# ============================================================

print("\n" + "=" * 100)
print("ANALYSIS 2: COMMON-SUPPORT RELATIVE UPWARD MOBILITY")
print("=" * 100)

relative = load_psid(RELATIVE_FILE)
relative_outcomes = ["DV1", "DV2", "DV3", "DV4"]

check_columns(
    relative,
    FULL_FEATURES
    + relative_outcomes
    + ["Wage Percentile", "weight1991", "ID68"],
    "common-support relative mobility analysis",
)

# SAME SAMPLE FOR +10, +20, +30, +40.
# Parents at or below the 60th percentile are capable of moving upward by all
# four thresholds.  Requiring all four outcomes observed makes the row set
# identical across thresholds.
common = relative.loc[
    relative["Wage Percentile"].notna()
    & (relative["Wage Percentile"] <= 60)
    & relative["weight1991"].notna()
    & (relative["weight1991"] > 0)
    & relative["ID68"].notna()
    & relative[relative_outcomes].notna().all(axis=1)
].copy().reset_index(drop=True)

print(f"Common-support N: {len(common):,}")
print(f"Common-support families: {common['ID68'].nunique():,}")

w_common = common["weight1991"].astype(float).to_numpy()
g_common = common["ID68"].to_numpy()

# Use the most demanding outcome (+40) to create one common set of outer folds.
# The exact same outer folds are then used for all four thresholds.
y_for_folds = common["DV4"].astype(int).to_numpy()
common_outer_folds = make_outer_folds(
    common[FULL_FEATURES],
    y_for_folds,
    g_common,
    seed=RANDOM_STATE + 60000,
)

common["OuterFold"] = np.nan
for fold_number, (_, test_idx) in enumerate(common_outer_folds, start=1):
    common.loc[test_idx, "OuterFold"] = fold_number
common["OuterFold"] = common["OuterFold"].astype(int)

threshold_map = {
    10: "DV1",
    20: "DV2",
    30: "DV3",
    40: "DV4",
}

# These objects are retained so the same family-bootstrap resamples can be used
# across thresholds, giving paired trend comparisons.
relative_prediction_objects = {}
relative_tuning_rows = []

for threshold, outcome in threshold_map.items():
    print("\n" + "-" * 80)
    print(f"Relative upward mobility: +{threshold} percentile points")
    print("-" * 80)

    y_rel = common[outcome].astype(int).to_numpy()

    baseline_oof = np.full(len(common), np.nan)
    perm_income_oof = np.full(
        (len(common), N_IMPORTANCE_PERMUTATIONS), np.nan
    )
    perm_wealth_oof = np.full(
        (len(common), N_IMPORTANCE_PERMUTATIONS), np.nan
    )

    for fold_number, (train_idx, test_idx) in enumerate(
        common_outer_folds, start=1
    ):
        X_train = common.iloc[train_idx][FULL_FEATURES]
        X_test = common.iloc[test_idx][FULL_FEATURES]

        y_train = y_rel[train_idx]
        y_test = y_rel[test_idx]
        w_train = w_common[train_idx]
        group_train = g_common[train_idx]

        best_params, tune_detail = weighted_inner_rf_search(
            X_train,
            y_train,
            w_train,
            group_train,
            seed=RANDOM_STATE + 70000 + 1000 * threshold + fold_number,
        )

        relative_tuning_rows.append(
            {
                "Threshold": threshold,
                "Outcome": outcome,
                "OuterFold": fold_number,
                "BestParameters": json.dumps(best_params),
                "BestInnerAUC": tune_detail.iloc[0]["MeanWeightedInnerAUC"],
            }
        )

        imputer = SimpleImputer(strategy="median")
        X_train_i = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=FULL_FEATURES,
        )
        X_test_i = pd.DataFrame(
            imputer.transform(X_test),
            columns=FULL_FEATURES,
        )

        model = make_rf(
            best_params,
            seed=RANDOM_STATE + 80000 + 1000 * threshold + fold_number,
        )
        model.fit(X_train_i, y_train, sample_weight=w_train)

        baseline_oof[test_idx] = model.predict_proba(X_test_i)[:, 1]

        for r in range(N_IMPORTANCE_PERMUTATIONS):
            # Permutations are performed WITHIN the held-out fold.  Because
            # income and wealth are single-variable groups, this is equivalent
            # to grouped permutation for those economic-resource dimensions.
            rng_income = np.random.default_rng(
                RANDOM_STATE
                + 90000
                + 100000 * threshold
                + 1000 * fold_number
                + r
            )
            rng_wealth = np.random.default_rng(
                RANDOM_STATE
                + 100000
                + 100000 * threshold
                + 1000 * fold_number
                + r
            )

            order_income = rng_income.permutation(len(X_test_i))
            order_wealth = rng_wealth.permutation(len(X_test_i))

            Xi = X_test_i.copy()
            Xi["Wage"] = X_test_i["Wage"].iloc[order_income].to_numpy()
            perm_income_oof[test_idx, r] = model.predict_proba(Xi)[:, 1]

            Xw = X_test_i.copy()
            Xw["Wealth"] = X_test_i["Wealth"].iloc[order_wealth].to_numpy()
            perm_wealth_oof[test_idx, r] = model.predict_proba(Xw)[:, 1]

    if np.isnan(baseline_oof).any():
        raise RuntimeError(f"Missing baseline OOF predictions at +{threshold}")
    if np.isnan(perm_income_oof).any() or np.isnan(perm_wealth_oof).any():
        raise RuntimeError(f"Missing permuted OOF predictions at +{threshold}")

    relative_prediction_objects[threshold] = {
        "Outcome": outcome,
        "y": y_rel,
        "baseline": baseline_oof,
        "perm_income": perm_income_oof,
        "perm_wealth": perm_wealth_oof,
    }

pd.DataFrame(relative_tuning_rows).to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_relative_common_support_tuning.csv",
    ),
    index=False,
)


# ============================================================
# 7. FIXED-MODEL CLUSTER-BOOTSTRAP IMPORTANCE UNCERTAINTY
# ============================================================

def permutation_importance_from_predictions(y, w, baseline, perm_matrix):
    base_auc = safe_weighted_auc(y, baseline, w)
    if not np.isfinite(base_auc):
        return np.nan

    perm_auc = []
    for r in range(perm_matrix.shape[1]):
        val = safe_weighted_auc(y, perm_matrix[:, r], w)
        if np.isfinite(val):
            perm_auc.append(val)

    if len(perm_auc) == 0:
        return np.nan

    return base_auc - float(np.mean(perm_auc))


common_boot_indices = make_cluster_bootstrap_indices(
    g_common,
    EVAL_BOOT_REPS,
    seed=RANDOM_STATE + 110000,
)

importance_rows = []
importance_boot = {}

for threshold, obj in relative_prediction_objects.items():
    y_rel = obj["y"]

    income_point = permutation_importance_from_predictions(
        y_rel,
        w_common,
        obj["baseline"],
        obj["perm_income"],
    )
    wealth_point = permutation_importance_from_predictions(
        y_rel,
        w_common,
        obj["baseline"],
        obj["perm_wealth"],
    )

    denom = income_point + wealth_point
    wealth_share_point = (
        wealth_point / denom
        if np.isfinite(denom) and denom > 0
        else np.nan
    )

    income_draws = []
    wealth_draws = []
    share_draws = []

    for idx in common_boot_indices:
        yy = y_rel[idx]
        ww = w_common[idx]

        income_imp = permutation_importance_from_predictions(
            yy,
            ww,
            obj["baseline"][idx],
            obj["perm_income"][idx, :],
        )
        wealth_imp = permutation_importance_from_predictions(
            yy,
            ww,
            obj["baseline"][idx],
            obj["perm_wealth"][idx, :],
        )

        income_draws.append(income_imp)
        wealth_draws.append(wealth_imp)

        d = income_imp + wealth_imp
        if np.isfinite(d) and d > 0:
            share_draws.append(wealth_imp / d)
        else:
            share_draws.append(np.nan)

    income_ci = percentile_ci(income_draws)
    wealth_ci = percentile_ci(wealth_draws)
    share_ci = percentile_ci(share_draws)

    importance_boot[threshold] = {
        "Income": np.asarray(income_draws, dtype=float),
        "Wealth": np.asarray(wealth_draws, dtype=float),
        "WealthShare": np.asarray(share_draws, dtype=float),
    }

    importance_rows.extend(
        [
            {
                "Threshold": threshold,
                "Group": "Income",
                "PermutationImportance": income_point,
                "Lower": income_ci[0],
                "Upper": income_ci[1],
                "WealthShareOfIncomePlusWealthImportance": np.nan,
                "WealthShareLower": np.nan,
                "WealthShareUpper": np.nan,
                "N": len(common),
                "Families": common["ID68"].nunique(),
                "BootstrapType": "Fixed-model held-out family-cluster bootstrap",
                "BootstrapReps": EVAL_BOOT_REPS,
                "Permutations": N_IMPORTANCE_PERMUTATIONS,
            },
            {
                "Threshold": threshold,
                "Group": "Wealth",
                "PermutationImportance": wealth_point,
                "Lower": wealth_ci[0],
                "Upper": wealth_ci[1],
                "WealthShareOfIncomePlusWealthImportance": wealth_share_point,
                "WealthShareLower": share_ci[0],
                "WealthShareUpper": share_ci[1],
                "N": len(common),
                "Families": common["ID68"].nunique(),
                "BootstrapType": "Fixed-model held-out family-cluster bootstrap",
                "BootstrapReps": EVAL_BOOT_REPS,
                "Permutations": N_IMPORTANCE_PERMUTATIONS,
            },
        ]
    )

importance_table = pd.DataFrame(importance_rows)
importance_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_relative_common_support_permutation_importance.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# 7.1 Paired tests of whether wealth becomes more important
# ------------------------------------------------------------
# Because every threshold uses the same sample, folds, and bootstrap cluster
# resamples, the +40 minus +10 comparison is paired.

wealth_10 = importance_boot[10]["Wealth"]
wealth_40 = importance_boot[40]["Wealth"]
share_10 = importance_boot[10]["WealthShare"]
share_40 = importance_boot[40]["WealthShare"]

wealth_diff_draws = wealth_40 - wealth_10
share_diff_draws = share_40 - share_10

# Bootstrap slope of wealth importance on threshold, expressed per +10 pp.
thresholds = np.array([10, 20, 30, 40], dtype=float)
wealth_draw_matrix = np.column_stack(
    [importance_boot[t]["Wealth"] for t in thresholds.astype(int)]
)

slope_draws = []
for row in wealth_draw_matrix:
    if np.all(np.isfinite(row)):
        # x is divided by 10 so the slope is change per additional 10 pp.
        slope_draws.append(np.polyfit(thresholds / 10.0, row, 1)[0])
    else:
        slope_draws.append(np.nan)

# Point-estimate counterparts.
point_by_threshold = (
    importance_table.loc[importance_table["Group"] == "Wealth"]
    .set_index("Threshold")["PermutationImportance"]
)
share_point_by_threshold = (
    importance_table.loc[importance_table["Group"] == "Wealth"]
    .set_index("Threshold")["WealthShareOfIncomePlusWealthImportance"]
)

wealth_diff_point = point_by_threshold.loc[40] - point_by_threshold.loc[10]
share_diff_point = share_point_by_threshold.loc[40] - share_point_by_threshold.loc[10]
slope_point = np.polyfit(
    thresholds / 10.0,
    point_by_threshold.reindex(thresholds.astype(int)).to_numpy(),
    1,
)[0]

trend_rows = []
for label, point, draws in [
    (
        "Wealth permutation importance: +40 minus +10",
        wealth_diff_point,
        wealth_diff_draws,
    ),
    (
        "Wealth share of income+wealth importance: +40 minus +10",
        share_diff_point,
        share_diff_draws,
    ),
    (
        "Linear trend in wealth permutation importance per additional 10 pp",
        slope_point,
        slope_draws,
    ),
]:
    ci = percentile_ci(draws)
    clean = np.asarray(draws, dtype=float)
    clean = clean[np.isfinite(clean)]

    trend_rows.append(
        {
            "Contrast": label,
            "Estimate": point,
            "Lower": ci[0],
            "Upper": ci[1],
            "BootstrapProbabilityPositive": (
                float(np.mean(clean > 0)) if len(clean) else np.nan
            ),
            "BootstrapType": "Fixed-model held-out family-cluster bootstrap",
            "BootstrapReps": EVAL_BOOT_REPS,
        }
    )

trend_table = pd.DataFrame(trend_rows)
trend_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_relative_common_support_trend_tests.csv",
    ),
    index=False,
)

print("\nCommon-support income/wealth permutation importance:")
print(importance_table.round(4).to_string(index=False))
print("\nPaired wealth-importance trend summaries:")
print(trend_table.round(4).to_string(index=False))


# ------------------------------------------------------------
# 7.2 Figure: importance across larger movements
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))

for group in ["Income", "Wealth"]:
    temp = importance_table.loc[
        importance_table["Group"] == group
    ].sort_values("Threshold")

    x = temp["Threshold"].to_numpy()
    yv = temp["PermutationImportance"].to_numpy()
    lower = temp["Lower"].to_numpy()
    upper = temp["Upper"].to_numpy()

    axes[0].errorbar(
        x,
        yv,
        yerr=np.vstack([yv - lower, upper - yv]),
        marker="o",
        capsize=3,
        label=group,
    )

axes[0].axhline(0, linewidth=0.8)
axes[0].set_xticks([10, 20, 30, 40])
axes[0].set_xlabel("Required upward rank movement (percentile points)")
axes[0].set_ylabel("Held-out AUC loss after permutation")
axes[0].set_title("Income and wealth predictive importance")
axes[0].legend(frameon=False)

wealth_plot = importance_table.loc[
    importance_table["Group"] == "Wealth"
].sort_values("Threshold")

x = wealth_plot["Threshold"].to_numpy()
yv = wealth_plot["WealthShareOfIncomePlusWealthImportance"].to_numpy()
lower = wealth_plot["WealthShareLower"].to_numpy()
upper = wealth_plot["WealthShareUpper"].to_numpy()

axes[1].errorbar(
    x,
    yv,
    yerr=np.vstack([yv - lower, upper - yv]),
    marker="o",
    capsize=3,
)
axes[1].set_xticks([10, 20, 30, 40])
axes[1].set_xlabel("Required upward rank movement (percentile points)")
axes[1].set_ylabel("Wealth share of income + wealth importance")
axes[1].set_ylim(0, 1)
axes[1].set_title("Relative importance of wealth")

fig.suptitle(
    "Income and Wealth as Upward Mobility Becomes More Demanding\n"
    "Common support: parental rank <= 60",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(
    os.path.join(
        RESULTS_DIR,
        "figure_robustness_income_wealth_importance_thresholds.pdf",
    ),
    bbox_inches="tight",
)
plt.show()


# ============================================================
# 8. ANALYSIS 3: DIRECT STANDARDIZED WEALTH GRADIENTS
# ============================================================

print("\n" + "=" * 100)
print("ANALYSIS 3: STANDARDIZED REALIZED WEALTH GRADIENTS")
print("=" * 100)

# Direct wealth comparisons require observed wealth rather than imputed wealth.
wealth_direct = common.loc[common["Wealth"].notna()].copy().reset_index(drop=True)

wealth_direct["WealthQuartile"] = assign_weighted_quantile_number(
    wealth_direct["Wealth"],
    wealth_direct["weight1991"],
    4,
)

# Three broad starting-rank bands preserve support and keep cell sizes useful.
wealth_direct["ParentRankBand"] = pd.cut(
    wealth_direct["Wage Percentile"],
    bins=[-np.inf, 20, 40, 60],
    labels=["0-20", "20-40", "40-60"],
    right=True,
)


def standardized_wealth_rates(df, outcome):
    """
    Standardize Q1 and Q4 wealth groups to the SAME parental starting-rank
    distribution using three 20-percentile-point rank bands.
    """
    z = df.loc[
        df[outcome].notna()
        & df["WealthQuartile"].isin([1, 4])
        & df["ParentRankBand"].notna()
    ].copy()

    # Standardization weights use the full Q1/Q4 sample's rank-band shares.
    rank_weight = (
        z.groupby("ParentRankBand", observed=True)["weight1991"]
        .sum()
    )
    rank_share = rank_weight / rank_weight.sum()

    rates = {}
    for q in [1, 4]:
        group = z.loc[z["WealthQuartile"] == q]
        standardized = 0.0

        for band, share in rank_share.items():
            cell = group.loc[group["ParentRankBand"] == band]
            if len(cell) == 0 or cell["weight1991"].sum() <= 0:
                return np.nan, np.nan, np.nan, np.nan

            cell_rate = weighted_mean(cell[outcome], cell["weight1991"])
            standardized += float(share) * cell_rate

        rates[q] = standardized

    gap = rates[4] - rates[1]
    ratio = rates[4] / rates[1] if rates[1] > 0 else np.nan
    return rates[1], rates[4], gap, ratio


standardized_rows = []
standardized_boot_draws = {}

for threshold, outcome in threshold_map.items():
    q1_point, q4_point, gap_point, ratio_point = standardized_wealth_rates(
        wealth_direct,
        outcome,
    )

    rng = np.random.default_rng(
        RANDOM_STATE + 120000 + threshold
    )

    q1_draws = []
    q4_draws = []
    gap_draws = []
    ratio_draws = []

    for _ in range(STANDARDIZATION_BOOT_REPS):
        idx = cluster_bootstrap_indices(wealth_direct["ID68"], rng)
        z = wealth_direct.iloc[idx]

        q1, q4, gap, ratio = standardized_wealth_rates(z, outcome)
        q1_draws.append(q1)
        q4_draws.append(q4)
        gap_draws.append(gap)
        ratio_draws.append(ratio)

    q1_ci = percentile_ci(q1_draws)
    q4_ci = percentile_ci(q4_draws)
    gap_ci = percentile_ci(gap_draws)
    ratio_ci = percentile_ci(ratio_draws)

    standardized_boot_draws[threshold] = {
        "Q1": np.asarray(q1_draws),
        "Q4": np.asarray(q4_draws),
        "Gap": np.asarray(gap_draws),
        "Ratio": np.asarray(ratio_draws),
    }

    standardized_rows.append(
        {
            "Threshold": threshold,
            "LowWealthStandardizedRate": q1_point,
            "LowWealthLower": q1_ci[0],
            "LowWealthUpper": q1_ci[1],
            "HighWealthStandardizedRate": q4_point,
            "HighWealthLower": q4_ci[0],
            "HighWealthUpper": q4_ci[1],
            "HighMinusLowWealthGap": gap_point,
            "GapLower": gap_ci[0],
            "GapUpper": gap_ci[1],
            "HighToLowWealthRiskRatio": ratio_point,
            "RatioLower": ratio_ci[0],
            "RatioUpper": ratio_ci[1],
            "N": len(wealth_direct),
            "Families": wealth_direct["ID68"].nunique(),
        }
    )

standardized_table = pd.DataFrame(standardized_rows)
standardized_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_standardized_wealth_gradient.csv",
    ),
    index=False,
)

print("\nRank-standardized realized wealth gradients:")
print(standardized_table.round(4).to_string(index=False))


# ------------------------------------------------------------
# 8.1 Wealth gradients within parental starting-rank bands
# ------------------------------------------------------------

starting_rank_rows = []

for threshold, outcome in threshold_map.items():
    for band in ["0-20", "20-40", "40-60"]:
        for wealth_q in [1, 2, 3, 4]:
            cell = wealth_direct.loc[
                (wealth_direct["ParentRankBand"] == band)
                & (wealth_direct["WealthQuartile"] == wealth_q)
            ].copy()

            if len(cell) == 0:
                continue

            starting_rank_rows.append(
                {
                    "Threshold": threshold,
                    "ParentRankBand": band,
                    "WealthQuartile": wealth_q,
                    "N": len(cell),
                    "Families": cell["ID68"].nunique(),
                    "WeightedRealizedMobility": weighted_mean(
                        cell[outcome], cell["weight1991"]
                    ),
                }
            )

starting_rank_table = pd.DataFrame(starting_rank_rows)
starting_rank_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_wealth_by_starting_rank.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# 8.2 Figure: standardized high- vs low-wealth rates
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

for prefix, label in [
    ("LowWealth", "Lowest wealth quartile"),
    ("HighWealth", "Highest wealth quartile"),
]:
    yv = standardized_table[f"{prefix}StandardizedRate"].to_numpy()
    lower = standardized_table[f"{prefix}Lower"].to_numpy()
    upper = standardized_table[f"{prefix}Upper"].to_numpy()
    x = standardized_table["Threshold"].to_numpy()

    axes[0].errorbar(
        x,
        yv,
        yerr=np.vstack([yv - lower, upper - yv]),
        marker="o",
        capsize=3,
        label=label,
    )

axes[0].set_xticks([10, 20, 30, 40])
axes[0].set_xlabel("Required upward rank movement (percentile points)")
axes[0].set_ylabel("Rank-standardized realized mobility rate")
axes[0].set_ylim(0, 1)
axes[0].set_title("Realized mobility by parental wealth")
axes[0].legend(frameon=False)

x = standardized_table["Threshold"].to_numpy()
yv = standardized_table["HighToLowWealthRiskRatio"].to_numpy()
lower = standardized_table["RatioLower"].to_numpy()
upper = standardized_table["RatioUpper"].to_numpy()

axes[1].errorbar(
    x,
    yv,
    yerr=np.vstack([yv - lower, upper - yv]),
    marker="o",
    capsize=3,
)
axes[1].axhline(1, linewidth=0.8)
axes[1].set_xticks([10, 20, 30, 40])
axes[1].set_xlabel("Required upward rank movement (percentile points)")
axes[1].set_ylabel("High-wealth / low-wealth realized mobility")
axes[1].set_title("Relative wealth gradient")

fig.suptitle(
    "Parental Wealth and Increasingly Large Upward Rank Movements\n"
    "Common support and standardized parental starting-rank distribution",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.89])
fig.savefig(
    os.path.join(
        RESULTS_DIR,
        "figure_robustness_standardized_wealth_gradient.pdf",
    ),
    bbox_inches="tight",
)
plt.show()


# ------------------------------------------------------------
# 8.3 Figure: wealth gradient separately by starting-rank band
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.ravel()

for ax, threshold in zip(axes, [10, 20, 30, 40]):
    temp = starting_rank_table.loc[
        starting_rank_table["Threshold"] == threshold
    ].copy()

    for band in ["0-20", "20-40", "40-60"]:
        z = temp.loc[temp["ParentRankBand"] == band].sort_values(
            "WealthQuartile"
        )
        ax.plot(
            z["WealthQuartile"],
            z["WeightedRealizedMobility"],
            marker="o",
            label=f"Parent rank {band}",
        )

    ax.set_title(f"Up at least {threshold} percentile points")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1)

for ax in axes[2:]:
    ax.set_xlabel("Parental wealth quartile")
for ax in [axes[0], axes[2]]:
    ax.set_ylabel("Weighted realized mobility rate")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.suptitle(
    "Wealth Gradients by Parental Starting Rank and Mobility Threshold",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.savefig(
    os.path.join(
        RESULTS_DIR,
        "figure_robustness_wealth_gradient_by_starting_rank.pdf",
    ),
    bbox_inches="tight",
)
plt.show()


# ============================================================
# 9. COMPACT TEXT SUMMARY FOR THE RESEARCHER
# ============================================================

print("\n" + "=" * 100)
print("KEY ROBUSTNESS OUTPUTS")
print("=" * 100)

print("\n1. FLEXIBLE ECONOMETRIC MODELS")
print(flexible_econ_table.round(4).to_string(index=False))

print("\n2. COMMON-SUPPORT INCOME/WEALTH IMPORTANCE")
print(importance_table.round(4).to_string(index=False))

print("\n3. WEALTH-IMPORTANCE TREND TESTS")
print(trend_table.round(4).to_string(index=False))

print("\n4. STANDARDIZED REALIZED WEALTH GRADIENTS")
print(standardized_table.round(4).to_string(index=False))

print("\nInterpretation guide:")
print(
    "- If wealth importance rises from +10 to +40 on the common-support sample, "
    "the original result is not driven by changing mechanical eligibility."
)
print(
    "- If the paired +40-minus-+10 interval is mostly above zero, the upward "
    "wealth-importance pattern is especially persuasive."
)
print(
    "- If the standardized realized wealth gradient also strengthens for larger "
    "movements, the result is visible directly in outcomes and not only in ML importance."
)
print(
    "- If the full additive spline logit closes much of the RF performance gap, "
    "nonlinear additive structure explains part of the ML gain.  If the income x wealth "
    "interaction adds further predictive content, that points specifically to resource interactions."
)
print(
    "- If the flexible-logit revision-validation figure shows realized gradients, "
    "the central beyond-income result is not specific to Random Forest."
)

print("\nDone. Outputs saved to:")
print(RESULTS_DIR)
