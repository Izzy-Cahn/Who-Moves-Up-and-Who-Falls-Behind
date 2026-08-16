############################################################
# WHO MOVES UP AND WHO FALLS BEHIND
#
# APPENDIX A.7 AND A.8
#
# Alternative mobility definitions:
#
# A.7.1 Relative mobility:
#       DV1-DV8
#
# A.7.2 Parent-child absolute mobility:
#       I1-I8
#
# A.8 Benchmark-based mobility:
#       A1-A8
#
#
# For EVERY outcome we estimate:
#
#   1. Random Forest
#   2. Gradient Boosting
#
# using:
#
#   - 5-fold cross-fitting
#   - 3-fold hyperparameter tuning WITHIN each training fold
#   - PSID sampling weights
#
#
# We calculate:
#
#   1. Impurity-based importance
#   2. Grouped permutation importance
#   3. Grouped SHAP importance
#
#
# Main output:
#
#   3 figures total:
#
#   figure27_relative_importance.pdf
#   figure28_absolute_importance.pdf
#   figure29_benchmark_importance.pdf
#
# Each figure has:
#
#   Panel A: Impurity importance ranking
#   Panel B: Permutation importance ranking
#   Panel C: SHAP importance ranking
#
# Rankings are averaged across RF and GB.
#
# All model-specific numerical results are saved to CSV.
############################################################


############################################################
# 1. PACKAGES
############################################################

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    roc_auc_score,
    log_loss
)


############################################################
# 2. PATHS
############################################################

DATA_DIR = (
    "/Users/yisroelcahn/Library/Mobile Documents/"
    "com~apple~CloudDocs/Documents/Who Moves Up/Data"
)

RESULTS_DIR = (
    "/Users/yisroelcahn/Library/Mobile Documents/"
    "com~apple~CloudDocs/Documents/Who Moves Up/Results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


############################################################
# 3. GENERAL SETTINGS
############################################################

RANDOM_STATE = 12345

# Paper uses five-fold cross-fitting
N_OUTER_FOLDS = 5

# Hyperparameter tuning within each training fold
N_INNER_FOLDS = 3

# Permutation repetitions
N_PERMUTATIONS = 20

# To keep SHAP computation reasonable.
# Set to None if you want every held-out observation.
SHAP_MAX_OBS = 300


############################################################
# 4. VARIABLE NAMES
############################################################

rename_dict = {

    # Demographics
    "page1991": "Age",
    "pselfemployed91": "Self-employed",
    "pfemale91": "Female",

    # Occupation
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

    # Industry
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

    # Race
    "pBlack": "African-American",

    # Geography
    "pNortheast": "Northeast",
    "pNorthCentral": "Northcentral",
    "pSouth": "South",
    "pWest": "West",

    # Education
    "psomeHS": "Some High School",
    "pHS": "High School",
    "psomeCollege": "Some College",
    "pCollege": "College",

    # Family structure
    "pmarried": "Married",

    # Economic resources
    "wage8594_p": "Wage Percentile",
    "wageL8594": "Wage",
    "wageL0919": "Child Wage",
    "wealth89": "Wealth",
}


############################################################
# 5. LOAD DATA
############################################################

def load_psid(filename):

    df = pd.read_stata(
        os.path.join(
            DATA_DIR,
            filename
        ),
        convert_categoricals=False
    )

    available_names = {
        old: new
        for old, new in rename_dict.items()
        if old in df.columns
    }

    df = df.rename(
        columns=available_names
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


relative_data = load_psid(
    "psidcleaned_data1.dta"
)

absolute_data = load_psid(
    "psidcleaned_data2.dta"
)

benchmark_data = load_psid(
    "psidcleaned_data3.dta"
)


datasets = {

    "Relative":
        relative_data,

    "Parent-child absolute":
        absolute_data,

    "Benchmark":
        benchmark_data
}


############################################################
# 6. PARENTAL PREDICTORS
#
# IMPORTANT:
#
# Only parental-generation characteristics enter X.
# Child income, child characteristics, ID, outcomes, etc.
# cannot accidentally leak into the model.
############################################################

candidate_features = [

    # Economic resources
    "Wage",
    "Wealth",

    # Demographics
    "Age",
    "Female",
    "African-American",

    # Education
    "Some High School",
    "High School",
    "Some College",
    "College",

    # Family structure / employment
    "Married",
    "Self-employed",

    # Occupation
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

    # Industry
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

    # Geography
    "Northeast",
    "Northcentral",
    "South",
    "West"
]


# Use predictors available in all three files
features = [

    x for x in candidate_features

    if all(
        x in df.columns
        for df in datasets.values()
    )
]


print("\nNumber of predictors:", len(features))

print("\nPredictors:")
for x in features:
    print("   ", x)


############################################################
# 7. ECONOMIC FEATURE GROUPS
#
# These are used for the compact appendix figures.
############################################################

feature_groups = {

    "Income": [
        "Wage"
    ],

    "Wealth": [
        "Wealth"
    ],

    "Education": [
        "Some High School",
        "High School",
        "Some College",
        "College"
    ],

    "Demographics": [
        "Age",
        "Female",
        "African-American"
    ],

    "Family / employment": [
        "Married",
        "Self-employed"
    ],

    "Occupation": [
        x for x in features
        if x.startswith(
            "Occupation ="
        )
    ],

    "Industry": [
        x for x in features
        if x.startswith(
            "Industry ="
        )
    ],

    "Geography": [
        "Northeast",
        "Northcentral",
        "South",
        "West"
    ]
}


# Remove variables that are not actually available
feature_groups = {

    group: [
        x for x in variables
        if x in features
    ]

    for group, variables
    in feature_groups.items()
}


# Remove empty groups
feature_groups = {

    group: variables

    for group, variables
    in feature_groups.items()

    if len(variables) > 0
}


GROUP_ORDER = [

    "Income",
    "Wealth",
    "Education",
    "Demographics",
    "Family / employment",
    "Occupation",
    "Industry",
    "Geography"
]


GROUP_ORDER = [

    x for x in GROUP_ORDER
    if x in feature_groups
]


############################################################
# 8. OUTCOME DEFINITIONS
############################################################

outcomes = [

    ########################################################
    # RELATIVE UPWARD
    ########################################################

    {
        "Family": "Relative",
        "Outcome": "DV1",
        "Direction": "Up",
        "Threshold": 10,
        "Label": "Up 10 pp"
    },

    {
        "Family": "Relative",
        "Outcome": "DV2",
        "Direction": "Up",
        "Threshold": 20,
        "Label": "Up 20 pp"
    },

    {
        "Family": "Relative",
        "Outcome": "DV3",
        "Direction": "Up",
        "Threshold": 30,
        "Label": "Up 30 pp"
    },

    {
        "Family": "Relative",
        "Outcome": "DV4",
        "Direction": "Up",
        "Threshold": 40,
        "Label": "Up 40 pp"
    },


    ########################################################
    # RELATIVE DOWNWARD
    ########################################################

    {
        "Family": "Relative",
        "Outcome": "DV5",
        "Direction": "Down",
        "Threshold": 10,
        "Label": "Down 10 pp"
    },

    {
        "Family": "Relative",
        "Outcome": "DV6",
        "Direction": "Down",
        "Threshold": 20,
        "Label": "Down 20 pp"
    },

    {
        "Family": "Relative",
        "Outcome": "DV7",
        "Direction": "Down",
        "Threshold": 30,
        "Label": "Down 30 pp"
    },

    {
        "Family": "Relative",
        "Outcome": "DV8",
        "Direction": "Down",
        "Threshold": 40,
        "Label": "Down 40 pp"
    },


    ########################################################
    # PARENT-CHILD ABSOLUTE UPWARD
    ########################################################

    {
        "Family": "Parent-child absolute",
        "Outcome": "I1",
        "Direction": "Up",
        "Threshold": 5000,
        "Label": "Up $5k"
    },

    {
        "Family": "Parent-child absolute",
        "Outcome": "I2",
        "Direction": "Up",
        "Threshold": 10000,
        "Label": "Up $10k"
    },

    {
        "Family": "Parent-child absolute",
        "Outcome": "I3",
        "Direction": "Up",
        "Threshold": 15000,
        "Label": "Up $15k"
    },

    {
        "Family": "Parent-child absolute",
        "Outcome": "I4",
        "Direction": "Up",
        "Threshold": 20000,
        "Label": "Up $20k"
    },


    ########################################################
    # PARENT-CHILD ABSOLUTE DOWNWARD
    ########################################################

    {
        "Family": "Parent-child absolute",
        "Outcome": "I5",
        "Direction": "Down",
        "Threshold": 5000,
        "Label": "Down $5k"
    },

    {
        "Family": "Parent-child absolute",
        "Outcome": "I6",
        "Direction": "Down",
        "Threshold": 10000,
        "Label": "Down $10k"
    },

    {
        "Family": "Parent-child absolute",
        "Outcome": "I7",
        "Direction": "Down",
        "Threshold": 15000,
        "Label": "Down $15k"
    },

    {
        "Family": "Parent-child absolute",
        "Outcome": "I8",
        "Direction": "Down",
        "Threshold": 20000,
        "Label": "Down $20k"
    },


    ########################################################
    # BENCHMARK UPWARD
    ########################################################

    {
        "Family": "Benchmark",
        "Outcome": "A1",
        "Direction": "Up",
        "Threshold": 5000,
        "Label": "Up $5k"
    },

    {
        "Family": "Benchmark",
        "Outcome": "A2",
        "Direction": "Up",
        "Threshold": 10000,
        "Label": "Up $10k"
    },

    {
        "Family": "Benchmark",
        "Outcome": "A3",
        "Direction": "Up",
        "Threshold": 15000,
        "Label": "Up $15k"
    },

    {
        "Family": "Benchmark",
        "Outcome": "A4",
        "Direction": "Up",
        "Threshold": 20000,
        "Label": "Up $20k"
    },


    ########################################################
    # BENCHMARK DOWNWARD
    ########################################################

    {
        "Family": "Benchmark",
        "Outcome": "A5",
        "Direction": "Down",
        "Threshold": 5000,
        "Label": "Down $5k"
    },

    {
        "Family": "Benchmark",
        "Outcome": "A6",
        "Direction": "Down",
        "Threshold": 10000,
        "Label": "Down $10k"
    },

    {
        "Family": "Benchmark",
        "Outcome": "A7",
        "Direction": "Down",
        "Threshold": 15000,
        "Label": "Down $15k"
    },

    {
        "Family": "Benchmark",
        "Outcome": "A8",
        "Direction": "Down",
        "Threshold": 20000,
        "Label": "Down $20k"
    }
]


############################################################
# 9. REASONABLE HYPERPARAMETER GRIDS
#
# Rather than evaluating every possible combination, these
# are four economically/reasonably different configurations.
#
# This keeps nested tuning manageable while still checking:
#
# RF:
#   - number of trees
#   - depth
#   - features per split
#   - minimum leaf size
#   - splitting criterion
#
# GB:
#   - number of trees
#   - learning rate
#   - depth
#   - features per split
#   - minimum leaf size
#   - subsampling
############################################################


RF_CONFIGS = [

    {
        "n_estimators": 250,
        "max_depth": 5,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "criterion": "gini"
    },

    {
        "n_estimators": 250,
        "max_depth": 10,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "criterion": "entropy"
    },

    {
        "n_estimators": 350,
        "max_depth": 10,
        "max_features": "log2",
        "min_samples_leaf": 10,
        "criterion": "gini"
    },

    {
        "n_estimators": 350,
        "max_depth": None,
        "max_features": 0.5,
        "min_samples_leaf": 10,
        "criterion": "entropy"
    }
]


GB_CONFIGS = [

    {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 2,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "subsample": 1.0
    },

    {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 2,
        "max_features": "sqrt",
        "min_samples_leaf": 10,
        "subsample": 1.0
    },

    {
        "n_estimators": 180,
        "learning_rate": 0.05,
        "max_depth": 3,
        "max_features": "log2",
        "min_samples_leaf": 5,
        "subsample": 0.8
    },

    {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 3,
        "max_features": None,
        "min_samples_leaf": 10,
        "subsample": 0.8
    }
]


############################################################
# 10. MODEL CONSTRUCTOR
############################################################

def make_model(
    model_name,
    params,
    seed
):

    if model_name == "Random Forest":

        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params
        )


    if model_name == "Gradient Boosting":

        return GradientBoostingClassifier(
            random_state=seed,
            **params
        )


    raise ValueError(
        "Unknown model: " + model_name
    )


############################################################
# 11. GET CORRECT ANALYSIS SAMPLE
############################################################

def get_analysis_sample(
    df,
    spec
):

    outcome = spec["Outcome"]

    d = df.copy()


    if outcome not in d.columns:

        raise ValueError(
            f"{outcome} not found in dataset"
        )


        # Outcomes, weights, and family identifiers must be observed.
    # ID68 is required so related PSID observations can be kept
    # together in both outer and inner cross-validation folds.
    if "ID68" not in d.columns:
        raise KeyError(
            "ID68 is required for family-grouped cross-validation."
        )

    d = d.loc[
        d[outcome].notna()
        &
        d["weight1991"].notna()
        &
        (d["weight1991"] > 0)
        &
        d["ID68"].notna()
    ].copy()


    ########################################################
    # RELATIVE MOBILITY:
    #
    # Exclude parents for whom the specified movement is
    # mechanically impossible because they begin too near
    # the top/bottom of the income distribution.
    ########################################################

    if spec["Family"] == "Relative":

        threshold = spec["Threshold"]

        d = d.loc[
            d["Wage Percentile"].notna()
        ].copy()


        if spec["Direction"] == "Up":

            d = d.loc[
                d["Wage Percentile"]
                <= (100 - threshold)
            ].copy()


        elif spec["Direction"] == "Down":

            d = d.loc[
                d["Wage Percentile"]
                > threshold
            ].copy()


    return d


############################################################
# 12. WEIGHTED BRIER SCORE
############################################################

def weighted_brier(
    y,
    probability,
    weights
):

    return np.average(
        (
            np.asarray(y)
            -
            np.asarray(probability)
        ) ** 2,
        weights=np.asarray(weights)
    )


############################################################
# 13. INNER HYPERPARAMETER TUNING
#
# Manual tuning lets us use PSID weights BOTH:
#
#   - when fitting
#   - when evaluating validation AUC
#
# This avoids relying on GridSearchCV's handling of
# sample weights in the scorer.
############################################################

def tune_model(
    X,
    y,
    weights,
    groups,
    model_name,
    seed
):

    if model_name == "Random Forest":
        configurations = RF_CONFIGS
    else:
        configurations = GB_CONFIGS


    min_class = min(
        int(y.sum()),
        int(len(y) - y.sum())
    )


    # Also make sure there are enough distinct PSID families
    # to form the requested number of grouped folds.
    n_groups = pd.Series(groups).nunique()


    inner_folds = min(
        N_INNER_FOLDS,
        min_class,
        n_groups
    )


    if inner_folds < 2:

        raise ValueError(
            "Too few outcome observations or PSID families "
            "for family-grouped hyperparameter tuning."
        )


    ########################################################
    # FAMILY-GROUPED INNER CROSS-VALIDATION
    ########################################################

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed
    )


    tuning_rows = []


    for config_number, params in enumerate(
        configurations,
        start=1
    ):

        fold_scores = []


        for inner_fold, (
            train_idx,
            valid_idx
        ) in enumerate(

            inner_cv.split(
                X,
                y,
                groups=groups
            ),

            start=1
        ):


            X_train = X.iloc[
                train_idx
            ].copy()

            X_valid = X.iloc[
                valid_idx
            ].copy()


            y_train = y.iloc[
                train_idx
            ].copy()

            y_valid = y.iloc[
                valid_idx
            ].copy()


            w_train = weights.iloc[
                train_idx
            ].copy()

            w_valid = weights.iloc[
                valid_idx
            ].copy()


            ###############################################
            # Imputation fitted on INNER training sample
            ###############################################

            imputer = SimpleImputer(
                strategy="median"
            )


            X_train_imp = pd.DataFrame(
                imputer.fit_transform(
                    X_train
                ),
                columns=X.columns
            )


            X_valid_imp = pd.DataFrame(
                imputer.transform(
                    X_valid
                ),
                columns=X.columns
            )


            ###############################################
            # Estimate model
            ###############################################

            model = make_model(
                model_name,
                params,
                seed
                + config_number
                + inner_fold
            )


            model.fit(
                X_train_imp,
                y_train,
                sample_weight=w_train
            )


            probability = (
                model.predict_proba(
                    X_valid_imp
                )[:, 1]
            )


            auc = roc_auc_score(
                y_valid,
                probability,
                sample_weight=w_valid
            )


            fold_scores.append(
                auc
            )


        tuning_rows.append({

            "Config":
                config_number,

            "Parameters":
                params,

            "MeanAUC":
                np.mean(
                    fold_scores
                )
        })


    tuning_df = pd.DataFrame(
        tuning_rows
    )


    best_index = (
        tuning_df["MeanAUC"]
        .idxmax()
    )


    best_params = (
        tuning_df.loc[
            best_index,
            "Parameters"
        ]
    )


    best_auc = (
        tuning_df.loc[
            best_index,
            "MeanAUC"
        ]
    )


    return (
        best_params,
        best_auc,
        tuning_df
    )


############################################################
# 14. IMPURITY IMPORTANCE
############################################################

def impurity_group_importance(
    model,
    feature_names
):

    feature_importance = np.asarray(
        model.feature_importances_
    )


    # Normalize to share of total importance
    if feature_importance.sum() > 0:

        feature_importance = (
            feature_importance
            /
            feature_importance.sum()
        )


    importance_by_feature = dict(
        zip(
            feature_names,
            feature_importance
        )
    )


    rows = []


    for group, variables in feature_groups.items():

        value = sum(
            importance_by_feature.get(
                variable,
                0
            )
            for variable in variables
        )


        rows.append({

            "Method": "Impurity",
            "Group": group,
            "Value": value
        })


    return pd.DataFrame(
        rows
    )


############################################################
# 15. GROUPED PERMUTATION IMPORTANCE
#
# Rather than separately permuting each education dummy,
# for example, the entire education vector is permuted
# jointly.
#
# The importance measure is:
#
#   baseline held-out AUC
#       minus
#   held-out AUC after permutation
############################################################

def permutation_group_importance(
    model,
    X_test,
    y_test,
    weights_test,
    seed
):

    base_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    baseline_auc = roc_auc_score(
        y_test,
        base_probability,
        sample_weight=weights_test
    )


    rows = []


    for group_number, (
        group,
        variables
    ) in enumerate(
        feature_groups.items(),
        start=1
    ):


        variables = [
            x for x in variables
            if x in X_test.columns
        ]


        if len(variables) == 0:
            continue


        decreases = []


        for r in range(
            N_PERMUTATIONS
        ):


            rng = np.random.default_rng(
                seed
                + 1000 * group_number
                + r
            )


            order = rng.permutation(
                len(X_test)
            )


            X_permuted = (
                X_test.copy()
            )


            # Same permutation for the whole group:
            # preserves relationships within the group.
            X_permuted.loc[
                :,
                variables
            ] = (
                X_test[
                    variables
                ]
                .iloc[order]
                .to_numpy()
            )


            probability = (
                model.predict_proba(
                    X_permuted
                )[:, 1]
            )


            permuted_auc = roc_auc_score(
                y_test,
                probability,
                sample_weight=weights_test
            )


            decreases.append(
                baseline_auc
                -
                permuted_auc
            )


        rows.append({

            "Method":
                "Permutation",

            "Group":
                group,

            "Value":
                np.mean(
                    decreases
                )
        })


    return pd.DataFrame(
        rows
    )


############################################################
# 16. SHAP HELPER
#
# Handles several SHAP output formats across package
# versions for binary classifiers.
############################################################

def get_class1_shap_values(
    model,
    X
):

    explainer = shap.TreeExplainer(
        model
    )


    values = explainer.shap_values(
        X
    )


    ############################################
    # Older SHAP versions:
    # list[class0, class1]
    ############################################

    if isinstance(
        values,
        list
    ):

        if len(values) == 2:

            values = values[1]

        else:

            values = values[0]


    values = np.asarray(
        values
    )


    ############################################
    # Newer SHAP RF output:
    # n x p x 2
    ############################################

    if values.ndim == 3:

        if values.shape[-1] == 2:

            values = (
                values[:, :, 1]
            )


        elif values.shape[0] == 2:

            values = (
                values[1, :, :]
            )


    if values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP output shape: "
            + str(values.shape)
        )


    return values


############################################################
# 17. GROUPED SHAP IMPORTANCE
#
# Mean absolute SHAP values are calculated using survey
# weights and then normalized so that group values sum to 1.
############################################################

def shap_group_importance(
    model,
    X_test,
    weights_test,
    seed
):

    ########################################################
    # Optional subsample for speed
    ########################################################

    if (
        SHAP_MAX_OBS is not None
        and
        len(X_test) > SHAP_MAX_OBS
    ):

        rng = np.random.default_rng(
            seed
        )


        indices = rng.choice(
            len(X_test),
            size=SHAP_MAX_OBS,
            replace=False
        )


        X_shap = (
            X_test.iloc[
                indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        w_shap = np.asarray(
            weights_test
        )[indices]


    else:

        X_shap = (
            X_test
            .copy()
            .reset_index(
                drop=True
            )
        )


        w_shap = np.asarray(
            weights_test
        )


    ############################################
    # Calculate class-1 SHAP values
    ############################################

    shap_values = (
        get_class1_shap_values(
            model,
            X_shap
        )
    )


    ############################################
    # Weighted mean absolute SHAP
    ############################################

    feature_shap = np.average(
        np.abs(
            shap_values
        ),
        axis=0,
        weights=w_shap
    )


    ############################################
    # Normalize to shares
    ############################################

    if feature_shap.sum() > 0:

        feature_shap = (
            feature_shap
            /
            feature_shap.sum()
        )


    importance_by_feature = dict(
        zip(
            X_test.columns,
            feature_shap
        )
    )


    rows = []


    for group, variables in feature_groups.items():

        value = sum(
            importance_by_feature.get(
                variable,
                0
            )
            for variable in variables
        )


        rows.append({

            "Method":
                "SHAP",

            "Group":
                group,

            "Value":
                value
        })


    return pd.DataFrame(
        rows
    )


############################################################
# 18. CROSS-FIT ONE OUTCOME / ONE MODEL
############################################################

def crossfit_model(
    df,
    spec,
    model_name
):


    d = get_analysis_sample(
        df,
        spec
    )


    X = (
        d[features]
        .reset_index(
            drop=True
        )
    )


    y = (
        d[spec["Outcome"]]
        .astype(int)
        .reset_index(
            drop=True
        )
    )


    weights = (
        d["weight1991"]
        .astype(float)
        .reset_index(
            drop=True
        )
    )
    
    groups = (
        d["ID68"]
        .reset_index(
            drop=True
            )
        )


    ############################################
    # Check class sizes
    ############################################

    positives = int(
        y.sum()
    )


    negatives = int(
        len(y)
        -
        y.sum()
    )


    min_class = min(
        positives,
        negatives
    )


    if min_class < 2:

        print(
            "Skipping",
            spec["Outcome"],
            "because one class is too small."
        )

        return (
            None,
            None,
            None
        )


    n_groups = groups.nunique()


    outer_folds = min(
        N_OUTER_FOLDS,
        min_class,
        n_groups
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    crossfit_probability = np.full(
        len(y),
        np.nan
    )


    importance_rows = []

    tuning_rows = []


    ########################################################
    # OUTER CROSS-FITTING LOOP
    ########################################################

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        outer_cv.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):


        print(
            "   ",
            model_name,
            "- fold",
            fold,
            "of",
            outer_folds
        )


        X_train = (
            X.iloc[
                train_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        X_test = (
            X.iloc[
                test_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        y_train = (
            y.iloc[
                train_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        y_test = (
            y.iloc[
                test_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        w_train = (
            weights.iloc[
                train_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        w_test = (
            weights.iloc[
                test_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )
        
        g_train = (
            groups.iloc[
                train_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        g_test = (
            groups.iloc[
                test_index
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        # Safety check: no PSID family may appear in both
        # the outer training and held-out samples.
        overlap = set(g_train).intersection(
            set(g_test)
        )


        if len(overlap) > 0:
            raise RuntimeError(
                "Family leakage detected in outer cross-validation."
            )


        ####################################################
        # Hyperparameter tuning INSIDE outer training fold
        ####################################################

        best_params, best_inner_auc, tuning_detail = (
            tune_model(
                X_train,
                y_train,
                w_train,
                g_train,
                model_name,
                seed=(
                    RANDOM_STATE
                    + 100 * fold
                )
            )
        )


        tuning_rows.append({

            "Family":
                spec["Family"],

            "Outcome":
                spec["Outcome"],

            "Label":
                spec["Label"],

            "Model":
                model_name,

            "Fold":
                fold,

            "InnerAUC":
                best_inner_auc,

            "BestParameters":
                json.dumps(
                    best_params
                )
        })


        ####################################################
        # Impute using OUTER training fold only
        ####################################################

        imputer = SimpleImputer(
            strategy="median"
        )


        X_train_imp = pd.DataFrame(

            imputer.fit_transform(
                X_train
            ),

            columns=features
        )


        X_test_imp = pd.DataFrame(

            imputer.transform(
                X_test
            ),

            columns=features
        )


        ####################################################
        # Fit selected model
        ####################################################

        model = make_model(
            model_name,
            best_params,
            seed=(
                RANDOM_STATE
                + 10000
                + fold
            )
        )


        model.fit(
            X_train_imp,
            y_train,
            sample_weight=w_train
        )


        ####################################################
        # Cross-fitted probability
        ####################################################

        probability = (
            model.predict_proba(
                X_test_imp
            )[:, 1]
        )


        crossfit_probability[
            test_index
        ] = probability


        ####################################################
        # IMPURITY IMPORTANCE
        ####################################################

        impurity = (
            impurity_group_importance(
                model,
                features
            )
        )


        ####################################################
        # PERMUTATION IMPORTANCE
        ####################################################

        permutation = (
            permutation_group_importance(
                model,
                X_test_imp,
                y_test,
                w_test,
                seed=(
                    RANDOM_STATE
                    + 20000
                    + fold
                )
            )
        )


        ####################################################
        # SHAP IMPORTANCE
        ####################################################

        shap_importance = (
            shap_group_importance(
                model,
                X_test_imp,
                w_test,
                seed=(
                    RANDOM_STATE
                    + 30000
                    + fold
                )
            )
        )


        ####################################################
        # Combine importance measures
        ####################################################

        fold_importance = pd.concat(
            [
                impurity,
                permutation,
                shap_importance
            ],
            ignore_index=True
        )


        fold_importance[
            "Family"
        ] = spec["Family"]


        fold_importance[
            "Outcome"
        ] = spec["Outcome"]


        fold_importance[
            "Label"
        ] = spec["Label"]


        fold_importance[
            "Direction"
        ] = spec["Direction"]


        fold_importance[
            "Threshold"
        ] = spec["Threshold"]


        fold_importance[
            "Model"
        ] = model_name


        fold_importance[
            "Fold"
        ] = fold


        importance_rows.append(
            fold_importance
        )


    ########################################################
    # CROSS-FITTED PERFORMANCE
    ########################################################

    auc = roc_auc_score(
        y,
        crossfit_probability,
        sample_weight=weights
    )


    brier = weighted_brier(
        y,
        crossfit_probability,
        weights
    )


    loss = log_loss(
        y,
        crossfit_probability,
        sample_weight=weights,
        labels=[0, 1]
    )


    event_rate = np.average(
        y,
        weights=weights
    )


    performance = {

        "Family":
            spec["Family"],

        "Outcome":
            spec["Outcome"],

        "Label":
            spec["Label"],

        "Direction":
            spec["Direction"],

        "Threshold":
            spec["Threshold"],

        "Model":
            model_name,

        "N":
            len(y),

        "EventRate":
            event_rate,

        "AUC":
            auc,

        "Brier":
            brier,

        "LogLoss":
            loss
    }


    importance_df = pd.concat(
        importance_rows,
        ignore_index=True
    )


    tuning_df = pd.DataFrame(
        tuning_rows
    )


    return (
        performance,
        importance_df,
        tuning_df
    )


############################################################
# 19. RUN EVERY OUTCOME
############################################################

performance_results = []

importance_results = []

tuning_results = []


for number, spec in enumerate(
    outcomes,
    start=1
):


    print("\n")
    print("=" * 80)

    print(
        f"Outcome {number} of {len(outcomes)}:"
    )

    print(
        spec["Family"],
        "|",
        spec["Label"]
    )

    print("=" * 80)


    df = datasets[
        spec["Family"]
    ]


    for model_name in [

        "Random Forest",
        "Gradient Boosting"

    ]:


        performance, importance, tuning = (
            crossfit_model(
                df,
                spec,
                model_name
            )
        )


        if performance is None:
            continue


        performance_results.append(
            performance
        )


        importance_results.append(
            importance
        )


        tuning_results.append(
            tuning
        )


############################################################
# 20. COMBINE EVERYTHING
############################################################

performance = pd.DataFrame(
    performance_results
)


importance_fold = pd.concat(
    importance_results,
    ignore_index=True
)


tuning = pd.concat(
    tuning_results,
    ignore_index=True
)


############################################################
# 21. SAVE FOLD-LEVEL RESULTS
############################################################

performance.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_performance_all.csv"
    ),

    index=False
)


importance_fold.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_importance_fold_level.csv"
    ),

    index=False
)


tuning.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_tuning_results.csv"
    ),

    index=False
)


############################################################
# 22. AVERAGE IMPORTANCE ACROSS OUTER FOLDS
############################################################

importance = (

    importance_fold

    .groupby(
        [
            "Family",
            "Outcome",
            "Label",
            "Direction",
            "Threshold",
            "Model",
            "Method",
            "Group"
        ],
        as_index=False
    )

    .agg(
        Value=("Value", "mean"),
        FoldSD=("Value", "std")
    )
)


############################################################
# 23. RANK EACH FEATURE GROUP
#
# Rank 1 = most important.
#
# Ranking is done separately by:
#
#   outcome
#   model
#   importance method
############################################################

importance["Rank"] = (

    importance

    .groupby(
        [
            "Family",
            "Outcome",
            "Model",
            "Method"
        ]
    )["Value"]

    .rank(
        ascending=False,
        method="average"
    )
)


importance.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_importance_summary.csv"
    ),

    index=False
)


############################################################
# 24. CONSENSUS IMPORTANCE
#
# Average rank across:
#
#    RF + GB
#
# while keeping the three methods separate.
############################################################

method_rank = (

    importance

    .groupby(
        [
            "Family",
            "Outcome",
            "Label",
            "Direction",
            "Threshold",
            "Method",
            "Group"
        ],
        as_index=False
    )

    .agg(
        AverageRank=("Rank", "mean")
    )
)


############################################################
# 25. OVERALL CONSENSUS RANK
#
# Average across:
#
#   RF
#   GB
#   impurity
#   permutation
#   SHAP
############################################################

consensus = (

    importance

    .groupby(
        [
            "Family",
            "Outcome",
            "Label",
            "Direction",
            "Threshold",
            "Group"
        ],
        as_index=False
    )

    .agg(
        ConsensusRank=("Rank", "mean")
    )
)


consensus.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_consensus_importance.csv"
    ),

    index=False
)


############################################################
# 26. COMPACT PERFORMANCE TABLE
############################################################

def make_performance_table(
    family
):


    temp = performance.loc[
        performance["Family"]
        ==
        family
    ].copy()


    base = (

        temp.loc[
            temp["Model"]
            ==
            "Random Forest",
            [
                "Outcome",
                "Label",
                "Direction",
                "Threshold",
                "N",
                "EventRate"
            ]
        ]

        .drop_duplicates()
    )


    rf = (

        temp.loc[
            temp["Model"]
            ==
            "Random Forest",
            [
                "Outcome",
                "AUC",
                "Brier",
                "LogLoss"
            ]
        ]

        .rename(
            columns={
                "AUC": "RF AUC",
                "Brier": "RF Brier",
                "LogLoss": "RF Log Loss"
            }
        )
    )


    gb = (

        temp.loc[
            temp["Model"]
            ==
            "Gradient Boosting",
            [
                "Outcome",
                "AUC",
                "Brier",
                "LogLoss"
            ]
        ]

        .rename(
            columns={
                "AUC": "GB AUC",
                "Brier": "GB Brier",
                "LogLoss": "GB Log Loss"
            }
        )
    )


    table = (

        base

        .merge(
            rf,
            on="Outcome"
        )

        .merge(
            gb,
            on="Outcome"
        )
    )


    desired_order = [

        spec["Outcome"]

        for spec in outcomes

        if spec["Family"]
        ==
        family
    ]


    order_map = {

        outcome: number

        for number, outcome
        in enumerate(
            desired_order
        )
    }


    table["order"] = (
        table["Outcome"]
        .map(
            order_map
        )
    )


    table = (

        table

        .sort_values(
            "order"
        )

        .drop(
            columns="order"
        )

        .reset_index(
            drop=True
        )
    )


    return table


############################################################
# 27. CREATE THREE PERFORMANCE TABLES
############################################################

relative_performance = (
    make_performance_table(
        "Relative"
    )
)


absolute_performance = (
    make_performance_table(
        "Parent-child absolute"
    )
)


benchmark_performance = (
    make_performance_table(
        "Benchmark"
    )
)


relative_performance.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_relative_performance.csv"
    ),

    index=False
)


absolute_performance.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_absolute_performance.csv"
    ),

    index=False
)


benchmark_performance.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A8_benchmark_performance.csv"
    ),

    index=False
)


############################################################
# 28. PRINT PERFORMANCE RESULTS
############################################################

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)


print("\n\n")
print("=" * 110)
print("A.7.1 RELATIVE MOBILITY")
print("=" * 110)

print(
    relative_performance
    .round(3)
    .to_string(
        index=False
    )
)


print("\n\n")
print("=" * 110)
print("A.7.2 PARENT-CHILD ABSOLUTE MOBILITY")
print("=" * 110)

print(
    absolute_performance
    .round(3)
    .to_string(
        index=False
    )
)


print("\n\n")
print("=" * 110)
print("A.8 BENCHMARK-BASED MOBILITY")
print("=" * 110)

print(
    benchmark_performance
    .round(3)
    .to_string(
        index=False
    )
)


############################################################
# 29. FIGURE FUNCTION
#
# ONE figure per mobility definition.
#
# Panel A = impurity rank
# Panel B = permutation rank
# Panel C = SHAP rank
#
# Values are average ranks across RF and GB.
#
# Rank = 1 means most important.
############################################################

def plot_importance_rank_figure(
    family,
    filename,
    title
):


    family_specs = [

        spec

        for spec in outcomes

        if spec["Family"]
        ==
        family
    ]


    outcome_labels = [

        spec["Label"]
        for spec in family_specs
    ]


    methods = [

        "Impurity",
        "Permutation",
        "SHAP"
    ]


    fig, axes = plt.subplots(

        1,
        3,

        figsize=(19, 7),

        sharey=True
    )


    last_image = None


    for ax, method in zip(
        axes,
        methods
    ):


        temp = method_rank.loc[

            (
                method_rank["Family"]
                ==
                family
            )

            &

            (
                method_rank["Method"]
                ==
                method
            )

        ].copy()


        matrix = temp.pivot(

            index="Label",
            columns="Group",
            values="AverageRank"
        )


        matrix = matrix.reindex(
            outcome_labels
        )


        matrix = matrix.reindex(
            columns=GROUP_ORDER
        )


        last_image = ax.imshow(

            matrix.values,

            aspect="auto",

            vmin=1,

            vmax=len(
                GROUP_ORDER
            )
        )


        ax.set_xticks(
            np.arange(
                len(
                    GROUP_ORDER
                )
            )
        )


        ax.set_xticklabels(

            GROUP_ORDER,

            rotation=45,

            ha="right"
        )


        ax.set_title(
            method
        )


        ############################################
        # Write numerical ranks in cells
        ############################################

        for i in range(
            matrix.shape[0]
        ):

            for j in range(
                matrix.shape[1]
            ):

                value = (
                    matrix.iloc[
                        i,
                        j
                    ]
                )


                if pd.notna(
                    value
                ):

                    ax.text(

                        j,
                        i,

                        f"{value:.1f}",

                        ha="center",

                        va="center",

                        fontsize=8
                    )


    ########################################################
    # Y labels
    ########################################################

    axes[0].set_yticks(
        np.arange(
            len(
                outcome_labels
            )
        )
    )


    axes[0].set_yticklabels(
        outcome_labels
    )


    axes[0].set_ylabel(
        "Mobility outcome"
    )


    ########################################################
    # Overall title
    ########################################################

    fig.suptitle(
        title
        + "\nAverage importance rank across Random Forest "
          "and Gradient Boosting",
        fontsize=14
    )


    ########################################################
    # Leave space on the right for the colorbar
    ########################################################

    fig.subplots_adjust(
        left=0.09,
        right=0.87,
        bottom=0.22,
        top=0.84,
        wspace=0.15
    )


    ########################################################
    # Put colorbar in its own axis to the right
    ########################################################

    cbar_ax = fig.add_axes([
        0.90,   # left
        0.25,   # bottom
        0.015,  # width
        0.50    # height
    ])

    colorbar = fig.colorbar(
        last_image,
        cax=cbar_ax
    )

    colorbar.set_label(
        "Importance rank (1 = most important)"
    )


    ########################################################
    # Save figure
    ########################################################

    fig.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        bbox_inches="tight"
    )

    plt.show()


############################################################
# 30. THREE MAIN FIGURES TOTAL
############################################################

plot_importance_rank_figure(

    family="Relative",

    filename=(
        "figure27_relative_importance.pdf"
    ),

    title=(
        "Predictive Importance Across "
        "Relative-Mobility Thresholds"
    )
)


plot_importance_rank_figure(

    family="Parent-child absolute",

    filename=(
        "figure28_absolute_importance.pdf"
    ),

    title=(
        "Predictive Importance Across "
        "Parent-Child Absolute-Mobility Thresholds"
    )
)


plot_importance_rank_figure(

    family="Benchmark",

    filename=(
        "figure29_benchmark_importance.pdf"
    ),

    title=(
        "Predictive Importance Across "
        "Benchmark-Based Mobility Thresholds"
    )
)


############################################################
# 31. PRINT TOP THREE BACKGROUND DIMENSIONS
#
# Consensus across:
#
#   Random Forest
#   Gradient Boosting
#   Impurity
#   Permutation
#   SHAP
############################################################

top3 = (

    consensus

    .sort_values(
        [
            "Family",
            "Outcome",
            "ConsensusRank"
        ]
    )

    .groupby(
        [
            "Family",
            "Outcome"
        ]
    )

    .head(3)

    .reset_index(
        drop=True
    )
)


top3.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_top3_predictor_groups.csv"
    ),

    index=False
)


print("\n\n")
print("=" * 110)
print("TOP THREE PREDICTOR GROUPS BY OUTCOME")
print("=" * 110)

print(

    top3[[
        "Family",
        "Label",
        "Group",
        "ConsensusRank"
    ]]

    .round(2)

    .to_string(
        index=False
    )
)


############################################################
# 32. RF-GB IMPORTANCE AGREEMENT
#
# Spearman rank correlation between RF and GB importance
# rankings, separately for each importance method/outcome.
############################################################

agreement_rows = []


for spec in outcomes:

    for method in [

        "Impurity",
        "Permutation",
        "SHAP"

    ]:


        temp = importance.loc[

            (
                importance["Family"]
                ==
                spec["Family"]
            )

            &

            (
                importance["Outcome"]
                ==
                spec["Outcome"]
            )

            &

            (
                importance["Method"]
                ==
                method
            )

        ].copy()


        rf = (

            temp.loc[
                temp["Model"]
                ==
                "Random Forest",
                [
                    "Group",
                    "Rank"
                ]
            ]

            .rename(
                columns={
                    "Rank": "RF Rank"
                }
            )
        )


        gb = (

            temp.loc[
                temp["Model"]
                ==
                "Gradient Boosting",
                [
                    "Group",
                    "Rank"
                ]
            ]

            .rename(
                columns={
                    "Rank": "GB Rank"
                }
            )
        )


        merged = rf.merge(
            gb,
            on="Group"
        )


        if len(
            merged
        ) >= 3:

            correlation = (

                merged["RF Rank"]

                .corr(
                    merged["GB Rank"],
                    method="spearman"
                )
            )

        else:

            correlation = np.nan


        agreement_rows.append({

            "Family":
                spec["Family"],

            "Outcome":
                spec["Outcome"],

            "Label":
                spec["Label"],

            "Method":
                method,

            "RF_GB_Spearman":
                correlation
        })


agreement = pd.DataFrame(
    agreement_rows
)


agreement.to_csv(

    os.path.join(
        RESULTS_DIR,
        "A7_A8_RF_GB_importance_agreement.csv"
    ),

    index=False
)


print("\n\n")
print("=" * 110)
print("RF-GB IMPORTANCE RANK AGREEMENT")
print("=" * 110)

print(

    agreement

    .round(3)

    .to_string(
        index=False
    )
)








"""
Who Moves Up and Who Falls Behind?
Main Python analysis
=================================

Purpose
-------
This script reproduces the paper's MAIN machine-learning analysis for the
benchmark-based upward-mobility outcome:

    T_u(0) = 1{offspring long-run income >= weighted median
               of parental long-run income}.

The script is deliberately organized so that every reported probability is
out-of-fold, every model uses the same outer folds, missing-value imputation is
estimated only on training data, and PSID sampling weights are used in model
fitting and performance evaluation.

The alternative mobility definitions in Appendix A.7/A.8 are estimated in the
separate A7/A8 script.

Important design choices
------------------------
1. Five-fold FAMILY-GROUPED cross-fitting is used. Observations sharing ID68
   are kept in the same fold to prevent closely related observations from
   appearing in both training and held-out samples.
2. Random Forest is the primary model. Its hyperparameters are selected by a
   weighted inner grid search within each outer training fold.
3. The same fold-specific RF hyperparameters selected for the full model are
   used for the nested information-set models in that outer fold. This isolates
   the contribution of added information without repeatedly changing the
   algorithm as the information set changes.
4. Gradient Boosting is tuned separately by weighted inner grid search and is
   used as an algorithmic robustness benchmark.
5. All subgroup summaries use PSID sampling weights.
6. Bootstrap uncertainty resamples ID68 clusters, not individual rows.
7. The conventional age-adjusted IGE comparison used in Figure 1 is
   reproduced here using weighted wealth groups, family-clustered standard
   errors, and family-cluster bootstrap confidence intervals. The broader
   lifecycle IGE/rank-rank tables remain Stata-based.

Outputs
-------
The script saves:
- table5_model_performance.csv
- table6_nested_information_sets.csv
- table6_nested_differences_vs_income.csv
- main_rf_selected_hyperparameters.csv
- main_gb_selected_hyperparameters.csv
- final_crossfitted_analysis_data.csv
- Figure 1 conventional IGE-versus-mobility results
- figures 2-19 (main machine-learning analysis)
- appendix A.6 descriptive figures 22-26
- appendix A.7/A.8 importance figures 27-29
- figure12_revision_validation.pdf
- cv_structure_robustness.csv
- calibration_statistics.csv

Run time
--------
The two nested grid searches are the expensive part. The grids are deliberately
moderate (16 configurations each). With N around 2,700, this should be
manageable on a modern laptop while still checking several meaningful
hyperparameters.
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
import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    ParameterGrid
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 12345
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
BOOT_REPS = 1000

DATA_PATH = (
    "/Users/yisroelcahn/Library/Mobile Documents/"
    "com~apple~CloudDocs/Documents/Who Moves Up/Data/"
    "psidcleaned_data3.dta"
)

RESULTS_PATH = (
    "/Users/yisroelcahn/Library/Mobile Documents/"
    "com~apple~CloudDocs/Documents/Who Moves Up/Results"
)

os.makedirs(RESULTS_PATH, exist_ok=True)


# Moderate RF grid: 2 x 2 x 2 x 2 = 16 configurations.
# We tune number of trees, depth, features considered at a split,
# and minimum leaf size. Gini is fixed as the splitting criterion.
RF_GRID = {
    "n_estimators": [250, 400],
    "max_depth": [5, 10],
    "max_features": ["sqrt", 0.50],
    "min_samples_leaf": [5, 15],
}

# Moderate GB grid: 2 x 2 x 2 x 2 = 16 configurations.
# Subsample and max_features are fixed to keep computation reasonable.
GB_GRID = {
    "n_estimators": [150, 250],
    "learning_rate": [0.03, 0.05],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 15],
}


# ============================================================
# 1. GENERAL WEIGHTED HELPERS
# ============================================================

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return np.average(values[ok], weights=weights[ok])


def weighted_quantile(values, weights, quantiles):
    """
    Weighted empirical quantile with linear interpolation over the
    cumulative-weight distribution.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.atleast_1d(quantiles).astype(float)

    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]

    if len(values) == 0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative = cumulative / weights.sum()

    result = np.interp(quantiles, cumulative, values)

    if len(quantiles) == 1:
        return float(result[0])
    return result


def assign_weighted_quantile_group(values, weights, q, labels=None):
    """
    Assign observations to approximately equal-weight quantile groups.
    Missing values receive NaN.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    cut_probs = np.arange(1, q) / q
    cuts = np.asarray(weighted_quantile(values, weights, cut_probs))

    # Remove duplicate cutpoints if the variable has mass points.
    cuts = np.unique(cuts[np.isfinite(cuts)])

    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    out[ok] = np.digitize(values[ok], cuts, right=True) + 1

    if labels is None:
        return pd.Series(out)

    mapping = {i + 1: labels[i] for i in range(min(len(labels), len(cuts) + 1))}
    return pd.Series(out).map(mapping)


def weighted_brier(y, p, w):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    return weighted_mean((y - p) ** 2, w)


def weighted_accuracy(y, p, w, threshold=0.50):
    yhat = (np.asarray(p) >= threshold).astype(int)
    return weighted_mean(yhat == np.asarray(y), w)


def weighted_metrics(y, p, w):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    w = np.asarray(w, dtype=float)

    return {
        "AUC": roc_auc_score(y, p, sample_weight=w),
        "Brier": weighted_brier(y, p, w),
        "Log Loss": log_loss(y, p, sample_weight=w, labels=[0, 1]),
        "Accuracy": weighted_accuracy(y, p, w),
    }


def cluster_bootstrap_indices(groups, rng):
    """
    Resample family clusters with replacement and return row indices.
    If a family is sampled twice, all of its rows appear twice.
    """
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)

    group_to_rows = {
        g: np.flatnonzero(groups == g)
        for g in unique_groups
    }

    return np.concatenate([group_to_rows[g] for g in sampled])


def cluster_bootstrap_ci(
    df,
    statistic_function,
    group_col="ID68",
    reps=BOOT_REPS,
    seed=RANDOM_STATE
):
    rng = np.random.default_rng(seed)
    groups = df[group_col].to_numpy()

    draws = []
    for _ in range(reps):
        idx = cluster_bootstrap_indices(groups, rng)
        try:
            value = statistic_function(df.iloc[idx])
            if np.isfinite(value):
                draws.append(value)
        except Exception:
            continue

    if len(draws) < max(50, reps // 10):
        return np.nan, np.nan

    return tuple(np.percentile(draws, [2.5, 97.5]))


# ============================================================
# 2. LOAD DATA AND RENAME VARIABLES
# ============================================================

raw = pd.read_stata(DATA_PATH, convert_categoricals=False)

rename_parent = {
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

raw = raw.rename(columns=rename_parent)
raw = raw.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 3. DEFINE THE MAIN OUTCOME
# ============================================================
# We use the PSID-weighted median of parental long-run income.
# This keeps the Python outcome definition aligned with the corrected
# Stata construction of the benchmark-based mobility variables.

required_outcome_vars = ["Wage", "Child Wage", "weight1991", "ID68"]
missing = [x for x in required_outcome_vars if x not in raw.columns]
if missing:
    raise KeyError(f"Required variables missing from data: {missing}")

parent_median = weighted_quantile(
    raw["Wage"],
    raw["weight1991"],
    0.50
)

raw["MedianMobility"] = np.where(
    raw["Child Wage"].notna(),
    (raw["Child Wage"] >= parent_median).astype(float),
    np.nan
)

print("\n" + "=" * 72)
print("MAIN OUTCOME")
print("=" * 72)
print(f"Weighted parental-generation median: ${parent_median:,.2f}")


# ============================================================
# 4. DEFINE PREDICTORS AND NESTED INFORMATION SETS
# ============================================================
# IMPORTANT: these are parental-generation variables only.
# Child characteristics and child income never enter X.

income_cols = ["Wage"]
wealth_cols = ["Wealth"]

education_cols = [
    "Some High School",
    "High School",
    "Some College",
    "College",
]

demographic_cols = [
    "Age",
    "Female",
    "African-American",
    "Married",
]

employment_cols = ["Self-employed"]

region_cols = [
    "Northeast",
    "Northcentral",
    "South",
    "West",
]

occupation_cols = [
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

industry_cols = [
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

full_features = (
    income_cols
    + wealth_cols
    + education_cols
    + demographic_cols
    + employment_cols
    + region_cols
    + occupation_cols
    + industry_cols
)

information_sets = {
    "Income": income_cols,
    "Income + wealth": income_cols + wealth_cols,
    "Income + wealth + education":
        income_cols + wealth_cols + education_cols,
    "Income + wealth + education + demographics":
        income_cols + wealth_cols + education_cols + demographic_cols,
    "Full background": full_features,
}

required = sorted(
    set(full_features + ["MedianMobility", "weight1991", "ID68", "Child Wage"])
)
missing = [x for x in required if x not in raw.columns]
if missing:
    raise KeyError(
        "The following variables required by the replication script are missing:\n"
        + "\n".join(missing)
    )

print(f"Number of full-model predictors: {len(full_features)}")


# ============================================================
# 5. DEFINE THE ANALYSIS SAMPLE
# ============================================================

analysis = raw.loc[
    raw["MedianMobility"].notna()
    & raw["weight1991"].notna()
    & (raw["weight1991"] > 0)
    & raw["ID68"].notna()
].copy().reset_index(drop=True)

y = analysis["MedianMobility"].astype(int).to_numpy()
weights = analysis["weight1991"].astype(float).to_numpy()
groups = analysis["ID68"].to_numpy()

print(f"Final predictive sample N: {len(analysis):,}")
print(
    "Weighted realized mobility rate: "
    f"{weighted_mean(y, weights):.3f}"
)


# ============================================================
# 6. CONVENTIONAL IGE VS. REALIZED MOBILITY BY PARENTAL WEALTH
#    (FIGURE 1 AND APPENDIX TABLE 13)
# ============================================================
#
# This section reproduces the conventional comparison used in Figure 1
# and the wealth-by-education robustness table in Appendix A.5.2.
#
# Corrections relative to the older standalone code:
#   1. Wealth quartiles are defined using PSID sampling weights.
#   2. IGE regressions use PSID sampling weights.
#   3. IGE standard errors are clustered by PSID family (ID68).
#   4. Mobility confidence intervals resample PSID families, not rows.
#   5. The overall IGE is estimated on the full age-eligible IGE sample;
#      wealth is required only for wealth-specific analyses.
# ============================================================

required_ige_vars = [
    "Wage",
    "Child Wage",
    "Age",
    "age2017",
    "weight1991",
    "ID68",
    "MedianMobility",
    "College",
    "Wealth",
]

missing_ige = [x for x in required_ige_vars if x not in analysis.columns]
if missing_ige:
    raise KeyError(
        "Variables required for Figure 1 / Appendix Table 13 are missing:\n"
        + "\n".join(missing_ige)
    )


# ------------------------------------------------------------
# 6.1 Full IGE sample
# ------------------------------------------------------------

ige_full = analysis.loc[
    (analysis["Wage"] > 0)
    & (analysis["Child Wage"] > 0)
    & analysis["Age"].notna()
    & analysis["age2017"].notna()
    & analysis["weight1991"].notna()
    & (analysis["weight1991"] > 0)
    & analysis["ID68"].notna()
    & analysis["MedianMobility"].notna()
].copy().reset_index(drop=True)

ige_full["LogParentIncome"] = np.log(ige_full["Wage"])
ige_full["LogChildIncome"] = np.log(ige_full["Child Wage"])
ige_full["ParentAge2"] = ige_full["Age"] ** 2
ige_full["ChildAge"] = ige_full["age2017"]
ige_full["ChildAge2"] = ige_full["ChildAge"] ** 2


def estimate_age_adjusted_ige(temp):
    """
    Weighted age-adjusted IGE with family-clustered standard errors.
    """
    regression_vars = [
        "LogParentIncome",
        "LogChildIncome",
        "Age",
        "ParentAge2",
        "ChildAge",
        "ChildAge2",
        "weight1991",
        "ID68",
    ]

    d = temp[regression_vars].dropna().copy()

    X = sm.add_constant(
        d[
            [
                "LogParentIncome",
                "Age",
                "ParentAge2",
                "ChildAge",
                "ChildAge2",
            ]
        ],
        has_constant="add",
    )

    model = sm.WLS(
        d["LogChildIncome"],
        X,
        weights=d["weight1991"],
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["ID68"]},
    )

    beta = float(model.params["LogParentIncome"])
    se = float(model.bse["LogParentIncome"])

    return {
        "N": len(d),
        "Families": d["ID68"].nunique(),
        "IGE": beta,
        "IGE_SE": se,
        "IGE_Lower": beta - 1.96 * se,
        "IGE_Upper": beta + 1.96 * se,
    }


overall_ige = estimate_age_adjusted_ige(ige_full)

pd.DataFrame([overall_ige]).to_csv(
    os.path.join(
        RESULTS_PATH,
        "figure1_overall_ige.csv",
    ),
    index=False,
)

print("\n" + "=" * 90)
print("FIGURE 1: OVERALL AGE-ADJUSTED IGE")
print("=" * 90)
print(pd.DataFrame([overall_ige]).round(3).to_string(index=False))


# ------------------------------------------------------------
# 6.2 Wealth-specific IGE sample and weighted wealth quartiles
# ------------------------------------------------------------

ige_wealth = (
    ige_full.loc[ige_full["Wealth"].notna()]
    .copy()
    .reset_index(drop=True)
)

ige_wealth["WealthQuartile"] = assign_weighted_quantile_group(
    ige_wealth["Wealth"],
    ige_wealth["weight1991"],
    4,
    labels=["Wealth Q1", "Wealth Q2", "Wealth Q3", "Wealth Q4"],
).to_numpy()

wealth_labels_ige = [
    "Wealth Q1",
    "Wealth Q2",
    "Wealth Q3",
    "Wealth Q4",
]


# ------------------------------------------------------------
# 6.3 Figure 1 numerical results
# ------------------------------------------------------------

figure1_rows = []

for j, wealth_group in enumerate(wealth_labels_ige):
    temp = ige_wealth.loc[
        ige_wealth["WealthQuartile"] == wealth_group
    ].copy()

    if len(temp) == 0:
        continue

    ige_result = estimate_age_adjusted_ige(temp)

    mobility_rate = weighted_mean(
        temp["MedianMobility"],
        temp["weight1991"],
    )

    mobility_lower, mobility_upper = cluster_bootstrap_ci(
        temp,
        lambda z: weighted_mean(
            z["MedianMobility"],
            z["weight1991"],
        ),
        group_col="ID68",
        reps=BOOT_REPS,
        seed=RANDOM_STATE + 700 + j,
    )

    figure1_rows.append(
        {
            "WealthQuartile": wealth_group,
            "N": len(temp),
            "Families": temp["ID68"].nunique(),
            "IGE": ige_result["IGE"],
            "IGE_SE": ige_result["IGE_SE"],
            "IGE_Lower": ige_result["IGE_Lower"],
            "IGE_Upper": ige_result["IGE_Upper"],
            "RealizedMobility": mobility_rate,
            "MobilityLower": mobility_lower,
            "MobilityUpper": mobility_upper,
        }
    )

figure1_results = pd.DataFrame(figure1_rows)

figure1_results.to_csv(
    os.path.join(
        RESULTS_PATH,
        "figure1_ige_mobility_by_parental_wealth.csv",
    ),
    index=False,
)

print("\n" + "=" * 110)
print("FIGURE 1: IGE AND REALIZED MOBILITY BY PARENTAL-WEALTH QUARTILE")
print("=" * 110)
print(
    figure1_results[
        [
            "WealthQuartile",
            "N",
            "Families",
            "IGE",
            "IGE_Lower",
            "IGE_Upper",
            "RealizedMobility",
            "MobilityLower",
            "MobilityUpper",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 6.4 Figure 1A: age-adjusted IGE by wealth quartile
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.5))

x_fig1 = np.arange(len(figure1_results))
ige_values = figure1_results["IGE"].to_numpy()

ige_error = np.vstack(
    [
        ige_values - figure1_results["IGE_Lower"].to_numpy(),
        figure1_results["IGE_Upper"].to_numpy() - ige_values,
    ]
)

ax.errorbar(
    x_fig1,
    ige_values,
    yerr=ige_error,
    fmt="o",
    capsize=5,
    linewidth=1.5,
    markersize=7,
)

ax.axhline(
    overall_ige["IGE"],
    linestyle="--",
    linewidth=1.2,
    label=f"Overall IGE = {overall_ige['IGE']:.3f}",
)

ax.set_xticks(x_fig1)
ax.set_xticklabels(["Q1\nLowest", "Q2", "Q3", "Q4\nHighest"])
ax.set_xlabel("Parental wealth quartile")
ax.set_ylabel("Age-adjusted intergenerational income elasticity")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(
    os.path.join(
        RESULTS_PATH,
        "figure1_ige_by_parental_wealth.pdf",
    ),
    bbox_inches="tight",
)
plt.show()


# ------------------------------------------------------------
# 6.5 Figure 1B: realized benchmark mobility by wealth quartile
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.5))

mobility_values = figure1_results["RealizedMobility"].to_numpy()

mobility_error = np.vstack(
    [
        mobility_values - figure1_results["MobilityLower"].to_numpy(),
        figure1_results["MobilityUpper"].to_numpy() - mobility_values,
    ]
)

ax.errorbar(
    x_fig1,
    mobility_values,
    yerr=mobility_error,
    fmt="o",
    capsize=5,
    linewidth=1.5,
    markersize=7,
)

ax.set_xticks(x_fig1)
ax.set_xticklabels(["Q1\nLowest", "Q2", "Q3", "Q4\nHighest"])
ax.set_xlabel("Parental wealth quartile")
ax.set_ylabel(
    "Probability of exceeding\n"
    "the parental-generation median"
)
ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(
    os.path.join(
        RESULTS_PATH,
        "figure1_mobility_by_parental_wealth.pdf",
    ),
    bbox_inches="tight",
)
plt.show()


# ------------------------------------------------------------
# 6.6 Pairwise wealth-quartile contrasts
# ------------------------------------------------------------

figure1_pair_rows = []

for i in range(len(figure1_results)):
    for j in range(i + 1, len(figure1_results)):
        g1 = figure1_results.iloc[i]
        g2 = figure1_results.iloc[j]

        figure1_pair_rows.append(
            {
                "Group1": g1["WealthQuartile"],
                "Group2": g2["WealthQuartile"],
                "IGE1": g1["IGE"],
                "IGE2": g2["IGE"],
                "IGE_Difference": abs(g1["IGE"] - g2["IGE"]),
                "Mobility1": g1["RealizedMobility"],
                "Mobility2": g2["RealizedMobility"],
                "Mobility_Difference":
                    abs(g1["RealizedMobility"] - g2["RealizedMobility"]),
            }
        )

figure1_pair_results = pd.DataFrame(figure1_pair_rows)

figure1_pair_results.to_csv(
    os.path.join(
        RESULTS_PATH,
        "figure1_wealth_pair_comparisons.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# 6.7 Appendix Table 13:
#     wealth quartile x parental college status
# ------------------------------------------------------------

# Build the education-group variable explicitly as an object column.
# This avoids NumPy dtype-promotion errors from mixing strings
# with missing numeric values.

ige_wealth["EducationGroup"] = pd.Series(
    pd.NA,
    index=ige_wealth.index,
    dtype="object",
)

ige_wealth.loc[
    ige_wealth["College"] == 1,
    "EducationGroup",
] = "College"

ige_wealth.loc[
    ige_wealth["College"] == 0,
    "EducationGroup",
] = "No college"

table13_rows = []

for wealth_group in wealth_labels_ige:
    for education_group in ["No college", "College"]:

        temp = ige_wealth.loc[
            (ige_wealth["WealthQuartile"] == wealth_group)
            & (ige_wealth["EducationGroup"] == education_group)
        ].copy()

        # Match the paper's minimum cell-size rule.
        if len(temp) < 40:
            continue

        ige_result = estimate_age_adjusted_ige(temp)

        mobility_rate = weighted_mean(
            temp["MedianMobility"],
            temp["weight1991"],
        )

        mobility_lower, mobility_upper = cluster_bootstrap_ci(
            temp,
            lambda z: weighted_mean(
                z["MedianMobility"],
                z["weight1991"],
            ),
            group_col="ID68",
            reps=BOOT_REPS,
            seed=RANDOM_STATE + 900 + len(table13_rows),
        )

        table13_rows.append(
            {
                "Group": f"{wealth_group} / {education_group}",
                "WealthQuartile": wealth_group,
                "Education": education_group,
                "N": len(temp),
                "Families": temp["ID68"].nunique(),
                "IGE": ige_result["IGE"],
                "IGE_SE": ige_result["IGE_SE"],
                "IGE_Lower": ige_result["IGE_Lower"],
                "IGE_Upper": ige_result["IGE_Upper"],
                "RealizedMobility": mobility_rate,
                "MobilityLower": mobility_lower,
                "MobilityUpper": mobility_upper,
            }
        )

table13_results = pd.DataFrame(table13_rows)

table13_results.to_csv(
    os.path.join(
        RESULTS_PATH,
        "table13_ige_wealth_education.csv",
    ),
    index=False,
)

print("\n" + "=" * 110)
print("APPENDIX TABLE 13: IGE AND MOBILITY BY WEALTH x EDUCATION")
print("=" * 110)
print(
    table13_results[
        [
            "Group",
            "N",
            "Families",
            "IGE",
            "IGE_Lower",
            "IGE_Upper",
            "RealizedMobility",
            "MobilityLower",
            "MobilityUpper",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 6. CREATE ONE COMMON SET OF FAMILY-GROUPED OUTER FOLDS
# ============================================================
# Keeping ID68 families together is more conservative than ordinary
# individual-level stratified CV and prevents family-linked observations from
# appearing in both training and test samples.

outer_cv = StratifiedGroupKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

outer_folds = list(
    outer_cv.split(
        analysis[full_features],
        y,
        groups=groups
    )
)

# Save fold assignment so every result is exactly reproducible.
analysis["OuterFold"] = np.nan
for fold_number, (_, test_idx) in enumerate(outer_folds, start=1):
    analysis.loc[test_idx, "OuterFold"] = fold_number
analysis["OuterFold"] = analysis["OuterFold"].astype(int)


# ============================================================
# 7. MODEL CONSTRUCTORS
# ============================================================

def make_rf(params, seed):
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        criterion="gini",
        **params
    )


def make_gb(params, seed):
    return GradientBoostingClassifier(
        random_state=seed,
        subsample=0.80,
        max_features="sqrt",
        **params
    )


# ============================================================
# 8. WEIGHTED INNER GRID SEARCH
# ============================================================
# We implement the grid search explicitly rather than relying on GridSearchCV's
# scorer, because this lets us use PSID weights both when fitting and when
# evaluating inner-fold AUC.

def weighted_inner_grid_search(
    X,
    y_train,
    w_train,
    group_train,
    model_type,
    seed
):
    X = X.reset_index(drop=True)
    y_train = np.asarray(y_train)
    w_train = np.asarray(w_train)
    group_train = np.asarray(group_train)

    inner_cv = StratifiedGroupKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=seed
    )

    if model_type == "RF":
        param_grid = list(ParameterGrid(RF_GRID))
    elif model_type == "GB":
        param_grid = list(ParameterGrid(GB_GRID))
    else:
        raise ValueError("model_type must be 'RF' or 'GB'.")

    rows = []

    for config_number, params in enumerate(param_grid, start=1):
        scores = []

        for inner_fold, (fit_idx, valid_idx) in enumerate(
            inner_cv.split(X, y_train, groups=group_train),
            start=1
        ):
            X_fit = X.iloc[fit_idx]
            X_valid = X.iloc[valid_idx]

            y_fit = y_train[fit_idx]
            y_valid = y_train[valid_idx]

            w_fit = w_train[fit_idx]
            w_valid = w_train[valid_idx]

            imputer = SimpleImputer(strategy="median")
            X_fit_i = imputer.fit_transform(X_fit)
            X_valid_i = imputer.transform(X_valid)

            if model_type == "RF":
                model = make_rf(
                    params,
                    seed + config_number * 100 + inner_fold
                )
            else:
                model = make_gb(
                    params,
                    seed + config_number * 100 + inner_fold
                )

            model.fit(X_fit_i, y_fit, sample_weight=w_fit)
            p_valid = model.predict_proba(X_valid_i)[:, 1]

            scores.append(
                roc_auc_score(
                    y_valid,
                    p_valid,
                    sample_weight=w_valid
                )
            )

        rows.append({
            "Config": config_number,
            "Parameters": params,
            "Mean weighted inner AUC": np.mean(scores),
            "SD inner AUC": np.std(scores, ddof=1),
        })

    results = pd.DataFrame(rows).sort_values(
        "Mean weighted inner AUC",
        ascending=False
    ).reset_index(drop=True)

    best_params = results.loc[0, "Parameters"]
    best_auc = results.loc[0, "Mean weighted inner AUC"]

    return best_params, best_auc, results


# ============================================================
# 9. PRIMARY RF: NESTED TUNING + ALL NESTED INFORMATION SETS
# ============================================================
# Within each OUTER fold:
#   (a) tune RF using the FULL predictor set and only outer-training data;
#   (b) fit each nested information-set model using the SAME chosen RF
#       configuration for that fold;
#   (c) predict the held-out outer fold.
#
# This produces genuinely out-of-fold probabilities and makes Table 6 an
# information-set comparison rather than a comparison of different algorithms.

rf_predictions = {
    name: np.full(len(analysis), np.nan)
    for name in information_sets
}

rf_tuning_rows = []

for fold_number, (train_idx, test_idx) in enumerate(outer_folds, start=1):

    print("\n" + "=" * 72)
    print(f"PRIMARY RANDOM FOREST: OUTER FOLD {fold_number}/{N_OUTER_FOLDS}")
    print("=" * 72)

    X_train_full = analysis.iloc[train_idx][full_features]

    best_params, best_inner_auc, tuning_detail = weighted_inner_grid_search(
        X_train_full,
        y[train_idx],
        weights[train_idx],
        groups[train_idx],
        model_type="RF",
        seed=RANDOM_STATE + fold_number
    )

    print("Selected RF parameters:", best_params)
    print(f"Inner weighted AUC: {best_inner_auc:.3f}")

    rf_tuning_rows.append({
        "Outer fold": fold_number,
        "Best parameters": json.dumps(best_params),
        "Inner weighted AUC": best_inner_auc,
    })

    # Use the fold-specific configuration for every nested information set.
    for set_number, (name, cols) in enumerate(information_sets.items(), start=1):

        X_train = analysis.iloc[train_idx][cols]
        X_test = analysis.iloc[test_idx][cols]

        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)

        model = make_rf(
            best_params,
            seed=RANDOM_STATE + 1000 * fold_number + set_number
        )

        model.fit(
            X_train_i,
            y[train_idx],
            sample_weight=weights[train_idx]
        )

        rf_predictions[name][test_idx] = model.predict_proba(X_test_i)[:, 1]

rf_tuning = pd.DataFrame(rf_tuning_rows)
rf_tuning.to_csv(
    os.path.join(RESULTS_PATH, "main_rf_selected_hyperparameters.csv"),
    index=False
)

for name, pred in rf_predictions.items():
    if np.isnan(pred).any():
        raise RuntimeError(f"Missing RF out-of-fold predictions for {name}")


# Convenient column names used throughout the remainder of the script.
analysis["Pred_Income"] = rf_predictions["Income"]
analysis["Pred_IncomeWealth"] = rf_predictions["Income + wealth"]
analysis["Pred_IncomeWealthEducation"] = (
    rf_predictions["Income + wealth + education"]
)
analysis["Pred_ResourcesDemographics"] = (
    rf_predictions["Income + wealth + education + demographics"]
)
analysis["Pred_Full"] = rf_predictions["Full background"]
analysis["PredictionRevision"] = analysis["Pred_Full"] - analysis["Pred_Income"]


# ============================================================
# 10. FULL GRADIENT BOOSTING: NESTED TUNING
# ============================================================

gb_pred = np.full(len(analysis), np.nan)
gb_tuning_rows = []

for fold_number, (train_idx, test_idx) in enumerate(outer_folds, start=1):

    print("\n" + "=" * 72)
    print(f"GRADIENT BOOSTING: OUTER FOLD {fold_number}/{N_OUTER_FOLDS}")
    print("=" * 72)

    X_train = analysis.iloc[train_idx][full_features]
    X_test = analysis.iloc[test_idx][full_features]

    best_params, best_inner_auc, _ = weighted_inner_grid_search(
        X_train,
        y[train_idx],
        weights[train_idx],
        groups[train_idx],
        model_type="GB",
        seed=RANDOM_STATE + 500 + fold_number
    )

    print("Selected GB parameters:", best_params)
    print(f"Inner weighted AUC: {best_inner_auc:.3f}")

    gb_tuning_rows.append({
        "Outer fold": fold_number,
        "Best parameters": json.dumps(best_params),
        "Inner weighted AUC": best_inner_auc,
    })

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    model = make_gb(
        best_params,
        seed=RANDOM_STATE + 2000 + fold_number
    )

    model.fit(
        X_train_i,
        y[train_idx],
        sample_weight=weights[train_idx]
    )

    gb_pred[test_idx] = model.predict_proba(X_test_i)[:, 1]

analysis["Pred_Full_GB"] = gb_pred

pd.DataFrame(gb_tuning_rows).to_csv(
    os.path.join(RESULTS_PATH, "main_gb_selected_hyperparameters.csv"),
    index=False
)


# ============================================================
# 11. CROSS-FITTED LOGISTIC BENCHMARKS
# ============================================================

def crossfit_logit(cols):
    pred = np.full(len(analysis), np.nan)

    for fold_number, (train_idx, test_idx) in enumerate(outer_folds, start=1):
        X_train = analysis.iloc[train_idx][cols]
        X_test = analysis.iloc[test_idx][cols]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)

        X_train_s = scaler.fit_transform(X_train_i)
        X_test_s = scaler.transform(X_test_i)

        # Very weak regularization approximates a conventional unpenalized logit
        # while remaining stable across scikit-learn versions.
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=10000,
            random_state=RANDOM_STATE + fold_number
        )

        model.fit(
            X_train_s,
            y[train_idx],
            sample_weight=weights[train_idx]
        )

        pred[test_idx] = model.predict_proba(X_test_s)[:, 1]

    return pred


analysis["Pred_Income_Logit"] = crossfit_logit(income_cols)
analysis["Pred_Full_Logit"] = crossfit_logit(full_features)


# ============================================================
# 12. TABLE 5: PERFORMANCE ACROSS MODELS AND INFORMATION SETS
# ============================================================

table5_specs = {
    "Income-only Logit": analysis["Pred_Income_Logit"].to_numpy(),
    "Full-background Logit": analysis["Pred_Full_Logit"].to_numpy(),
    "Income-only Random Forest": analysis["Pred_Income"].to_numpy(),
    "Full-background Random Forest": analysis["Pred_Full"].to_numpy(),
    "Full-background Gradient Boosting": analysis["Pred_Full_GB"].to_numpy(),
}

table5_rows = []
income_logit_auc = weighted_metrics(
    y,
    table5_specs["Income-only Logit"],
    weights
)["AUC"]

for name, pred in table5_specs.items():
    metrics = weighted_metrics(y, pred, weights)
    table5_rows.append({
        "Model": name,
        **metrics,
        "Delta AUC vs Income Logit": metrics["AUC"] - income_logit_auc,
    })

table5 = pd.DataFrame(table5_rows)

print("\n" + "=" * 100)
print("TABLE 5: CROSS-FITTED MODEL PERFORMANCE")
print("=" * 100)
print(table5.round(3).to_string(index=False))

table5.to_csv(
    os.path.join(RESULTS_PATH, "table5_model_performance.csv"),
    index=False
)


# ============================================================
# 13. TABLE 6: NESTED INFORMATION SETS
# ============================================================
# IMPORTANT: Table 5's income-only RF and full RF are EXACTLY the same
# prediction vectors used here, eliminating the inconsistency that can arise
# when duplicate model pipelines are run with different settings.

nested_rows = []

for name, pred in rf_predictions.items():
    metrics = weighted_metrics(y, pred, weights)
    nested_rows.append({
        "Information Set": name,
        "AUC": metrics["AUC"],
        "Brier": metrics["Brier"],
        "Log Loss": metrics["Log Loss"],
    })

table6 = pd.DataFrame(nested_rows)


# ============================================================
# 14. CLUSTER-BOOTSTRAP CIs FOR TABLE 6 AND FIGURE 12
# ============================================================

def bootstrap_nested_performance(reps=BOOT_REPS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    groups_array = analysis["ID68"].to_numpy()

    boot = {
        name: {"AUC": [], "Brier": []}
        for name in information_sets
    }

    for _ in range(reps):
        idx = cluster_bootstrap_indices(groups_array, rng)
        y_b = y[idx]
        w_b = weights[idx]

        if len(np.unique(y_b)) < 2:
            continue

        for name, pred in rf_predictions.items():
            p_b = pred[idx]
            boot[name]["AUC"].append(
                roc_auc_score(y_b, p_b, sample_weight=w_b)
            )
            boot[name]["Brier"].append(
                weighted_brier(y_b, p_b, w_b)
            )

    return boot


boot_nested = bootstrap_nested_performance()

table6_ci_rows = []
for name in information_sets:
    row = table6.loc[table6["Information Set"] == name].iloc[0]

    auc_ci = np.percentile(boot_nested[name]["AUC"], [2.5, 97.5])
    brier_ci = np.percentile(boot_nested[name]["Brier"], [2.5, 97.5])

    table6_ci_rows.append({
        "Information Set": name,
        "AUC": row["AUC"],
        "AUC Lower": auc_ci[0],
        "AUC Upper": auc_ci[1],
        "Brier": row["Brier"],
        "Brier Lower": brier_ci[0],
        "Brier Upper": brier_ci[1],
    })

table6_ci = pd.DataFrame(table6_ci_rows)

income_auc_boot = np.asarray(boot_nested["Income"]["AUC"])
income_brier_boot = np.asarray(boot_nested["Income"]["Brier"])
income_point = table6_ci.loc[table6_ci["Information Set"] == "Income"].iloc[0]

difference_rows = []
for name in list(information_sets.keys())[1:]:
    row = table6_ci.loc[table6_ci["Information Set"] == name].iloc[0]

    auc_diff = np.asarray(boot_nested[name]["AUC"]) - income_auc_boot
    brier_diff = np.asarray(boot_nested[name]["Brier"]) - income_brier_boot

    difference_rows.append({
        "Information Set": name,
        "Delta AUC vs Income": row["AUC"] - income_point["AUC"],
        "Delta AUC Lower": np.percentile(auc_diff, 2.5),
        "Delta AUC Upper": np.percentile(auc_diff, 97.5),
        "Delta Brier vs Income": row["Brier"] - income_point["Brier"],
        "Delta Brier Lower": np.percentile(brier_diff, 2.5),
        "Delta Brier Upper": np.percentile(brier_diff, 97.5),
    })

table6_diffs = pd.DataFrame(difference_rows)

print("\n" + "=" * 100)
print("TABLE 6: NESTED INFORMATION SETS")
print("=" * 100)
print(table6_ci.round(3).to_string(index=False))

print("\nPAIRED DIFFERENCES RELATIVE TO INCOME ONLY")
print(table6_diffs.round(3).to_string(index=False))

table6_ci.to_csv(
    os.path.join(RESULTS_PATH, "table6_nested_information_sets.csv"),
    index=False
)
table6_diffs.to_csv(
    os.path.join(RESULTS_PATH, "table6_nested_differences_vs_income.csv"),
    index=False
)


# ============================================================
# 15. CALIBRATION HELPER
# ============================================================

def calibration_table(df, pred_col, n_bins=10):
    work = df[["MedianMobility", pred_col, "weight1991", "ID68"]].dropna().copy()

    work["CalibrationBin"] = assign_weighted_quantile_group(
        work[pred_col],
        work["weight1991"],
        q=n_bins
    ).to_numpy()

    rows = []
    for b in sorted(work["CalibrationBin"].dropna().unique()):
        temp = work.loc[work["CalibrationBin"] == b]
        rows.append({
            "Bin": int(b),
            "N": len(temp),
            "Mean predicted":
                weighted_mean(temp[pred_col], temp["weight1991"]),
            "Observed rate":
                weighted_mean(temp["MedianMobility"], temp["weight1991"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# 16. FIGURE 2: CALIBRATION OF MAIN CROSS-FITTED RF
# ============================================================

calibration = calibration_table(analysis, "Pred_Full", n_bins=10)
calibration.to_csv(
    os.path.join(RESULTS_PATH, "figure2_calibration_data.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(
    calibration["Mean predicted"],
    calibration["Observed rate"],
    marker="o",
    linewidth=1.5
)
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed upward mobility rate")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure2_calibration.pdf"),
    bbox_inches="tight"
)
plt.show()


# Calibration-in-the-large and a simple weighted calibration slope.
# Slope/intercept are descriptive diagnostics; inference is not used.
p_clip = np.clip(analysis["Pred_Full"].to_numpy(), 1e-6, 1 - 1e-6)
logit_p = np.log(p_clip / (1 - p_clip))

# Weighted least-squares probability calibration slope is not the formal
# logistic calibration slope, so we report only calibration-in-the-large
# and ECE here to avoid overinterpreting a linear approximation.
calibration_in_large = (
    weighted_mean(y, weights)
    - weighted_mean(p_clip, weights)
)

ece = np.average(
    np.abs(calibration["Observed rate"] - calibration["Mean predicted"]),
    weights=calibration["N"]
)

calibration_stats = pd.DataFrame([{
    "Weighted observed rate": weighted_mean(y, weights),
    "Weighted mean prediction": weighted_mean(p_clip, weights),
    "Calibration in the large (observed - predicted)": calibration_in_large,
    "Decile ECE": ece,
}])

calibration_stats.to_csv(
    os.path.join(RESULTS_PATH, "calibration_statistics.csv"),
    index=False
)


# ============================================================
# 17. FIGURE 3: DISTRIBUTION OF PREDICTED MOBILITY
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(
    analysis["Pred_Full"],
    bins=20,
    weights=analysis["weight1991"],
    density=True,
    edgecolor="black",
    linewidth=0.4
)
mean_pred = weighted_mean(analysis["Pred_Full"], analysis["weight1991"])
axes[0].axvline(
    mean_pred,
    linestyle="--",
    linewidth=1.2,
    label=f"Weighted mean = {mean_pred:.3f}"
)
axes[0].set_xlabel("Predicted probability of upward mobility")
axes[0].set_ylabel("Weighted density")
axes[0].legend(frameon=False)

q_grid = np.linspace(0.01, 0.99, 99)
pred_quantiles = weighted_quantile(
    analysis["Pred_Full"],
    analysis["weight1991"],
    q_grid
)
axes[1].plot(q_grid * 100, pred_quantiles, linewidth=2)
axes[1].set_xlabel("Percentile of predicted mobility")
axes[1].set_ylabel("Predicted probability of upward mobility")
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure3_predicted_mobility_distribution.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 18. COMMON ECONOMIC GROUPS
# ============================================================

# Weighted income/wealth quartiles and deciles.
analysis["IncomeQuartile"] = assign_weighted_quantile_group(
    analysis["Wage"],
    analysis["weight1991"],
    4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)

analysis["WealthQuartile"] = assign_weighted_quantile_group(
    analysis["Wealth"],
    analysis["weight1991"],
    4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)

analysis["IncomeQuintile"] = assign_weighted_quantile_group(
    analysis["Wage"],
    analysis["weight1991"],
    5,
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)

analysis["IncomeDecile"] = assign_weighted_quantile_group(
    analysis["Wage"],
    analysis["weight1991"],
    10,
    labels=[f"D{i}" for i in range(1, 11)]
)

analysis["WealthDecile"] = assign_weighted_quantile_group(
    analysis["Wealth"],
    analysis["weight1991"],
    10,
    labels=[f"D{i}" for i in range(1, 11)]
)

wealth_median = weighted_quantile(
    analysis["Wealth"],
    analysis["weight1991"],
    0.50
)

analysis["High Wealth"] = np.where(
    analysis["Wealth"].notna(),
    (analysis["Wealth"] >= wealth_median).astype(float),
    np.nan
)

analysis["HighAdvantage"] = np.where(
    analysis["High Wealth"].notna(),
    ((analysis["College"] == 1) & (analysis["High Wealth"] == 1)).astype(float),
    np.nan
)

analysis["LowAdvantage"] = np.where(
    analysis["High Wealth"].notna(),
    ((analysis["College"] == 0) & (analysis["High Wealth"] == 0)).astype(float),
    np.nan
)


# ============================================================
# 19. FIGURE 4: REALIZED AND PREDICTED MOBILITY BY
#     PARENTAL INCOME x WEALTH QUARTILE
# ============================================================

cell_rows = []

for iq in ["Q1", "Q2", "Q3", "Q4"]:
    for wq in ["Q1", "Q2", "Q3", "Q4"]:
        temp = analysis.loc[
            (analysis["IncomeQuartile"] == iq)
            & (analysis["WealthQuartile"] == wq)
        ]

        if len(temp) == 0:
            continue

        cell_rows.append({
            "IncomeQuartile": iq,
            "WealthQuartile": wq,
            "N": len(temp),
            "Realized":
                weighted_mean(temp["MedianMobility"], temp["weight1991"]),
            "Predicted":
                weighted_mean(temp["Pred_Full"], temp["weight1991"]),
        })

cell_results = pd.DataFrame(cell_rows)
cell_results.to_csv(
    os.path.join(RESULTS_PATH, "figure4_income_wealth_cells.csv"),
    index=False
)

realized_matrix = cell_results.pivot(
    index="IncomeQuartile",
    columns="WealthQuartile",
    values="Realized"
).reindex(index=["Q1", "Q2", "Q3", "Q4"], columns=["Q1", "Q2", "Q3", "Q4"])

predicted_matrix = cell_results.pivot(
    index="IncomeQuartile",
    columns="WealthQuartile",
    values="Predicted"
).reindex(index=["Q1", "Q2", "Q3", "Q4"], columns=["Q1", "Q2", "Q3", "Q4"])

n_matrix = cell_results.pivot(
    index="IncomeQuartile",
    columns="WealthQuartile",
    values="N"
).reindex(index=["Q1", "Q2", "Q3", "Q4"], columns=["Q1", "Q2", "Q3", "Q4"])

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

for ax, matrix, panel_title in [
    (axes[0], realized_matrix, "Realized mobility"),
    (axes[1], predicted_matrix, "Cross-fitted predicted mobility")
]:
    im = ax.imshow(matrix.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Parental wealth quartile")
    ax.set_ylabel("Parental income quartile")
    ax.set_title(panel_title)

    for i in range(4):
        for j in range(4):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                if panel_title == "Realized mobility":
                    text = f"{value:.2f}\nN={int(n_matrix.iloc[i, j])}"
                else:
                    text = f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)

# Reserve space to the right of BOTH panels, then place the colorbar
# in its own axis. This prevents the colorbar from covering panel B.
fig.subplots_adjust(
    left=0.08,
    right=0.86,
    bottom=0.12,
    top=0.90,
    wspace=0.28
)

cbar_ax = fig.add_axes([
    0.89,   # left
    0.18,   # bottom
    0.018,  # width
    0.64    # height
])

colorbar = fig.colorbar(
    im,
    cax=cbar_ax
)
colorbar.set_label("Probability")

fig.savefig(
    os.path.join(RESULTS_PATH, "figure4_income_wealth_realized_predicted.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 20. FIGURE 5: WITHIN-INCOME HETEROGENEITY
# ============================================================
# Weighted 10th, median, and 90th percentiles of predicted mobility
# within each parental-income decile.

heterogeneity_rows = []

for d in [f"D{i}" for i in range(1, 11)]:
    temp = analysis.loc[analysis["IncomeDecile"] == d]
    if len(temp) == 0:
        continue

    q10, q50, q90 = weighted_quantile(
        temp["Pred_Full"],
        temp["weight1991"],
        [0.10, 0.50, 0.90]
    )

    heterogeneity_rows.append({
        "Income decile": d,
        "P10": q10,
        "Median": q50,
        "P90": q90,
        "N": len(temp),
    })

figure5_data = pd.DataFrame(heterogeneity_rows)
figure5_data.to_csv(
    os.path.join(RESULTS_PATH, "figure5_within_income_heterogeneity.csv"),
    index=False
)

x = np.arange(1, 11)
med = figure5_data["Median"].to_numpy()

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.errorbar(
    x,
    med,
    yerr=[
        med - figure5_data["P10"].to_numpy(),
        figure5_data["P90"].to_numpy() - med
    ],
    fmt="o",
    capsize=4
)
ax.plot(x, med, linewidth=1)
ax.set_xticks(x)
ax.set_xlabel("Parental income decile")
ax.set_ylabel("Cross-fitted predicted probability of upward mobility")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure5_within_income_heterogeneity.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 21. FIGURE 6: REALIZED MOBILITY WITHIN INCOME QUINTILES
#     BY PREDICTED-OPPORTUNITY TERCILE
# ============================================================

analysis["OpportunityGroup"] = np.nan

for iq in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    mask = analysis["IncomeQuintile"] == iq
    temp = analysis.loc[mask]

    groups_temp = assign_weighted_quantile_group(
        temp["Pred_Full"],
        temp["weight1991"],
        3,
        labels=["Low", "Middle", "High"]
    )

    analysis.loc[mask, "OpportunityGroup"] = groups_temp.to_numpy()

opportunity_rows = []

for iq in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    for opportunity in ["Low", "Middle", "High"]:
        temp = analysis.loc[
            (analysis["IncomeQuintile"] == iq)
            & (analysis["OpportunityGroup"] == opportunity)
        ].copy()

        if len(temp) == 0:
            continue

        point = weighted_mean(temp["MedianMobility"], temp["weight1991"])

        low, high = cluster_bootstrap_ci(
            temp,
            lambda z: weighted_mean(z["MedianMobility"], z["weight1991"]),
            reps=BOOT_REPS,
            seed=RANDOM_STATE + len(opportunity_rows)
        )

        opportunity_rows.append({
            "Income Quintile": iq,
            "Opportunity Group": opportunity,
            "Realized Mobility": point,
            "CI Lower": low,
            "CI Upper": high,
            "Mean Predicted Mobility":
                weighted_mean(temp["Pred_Full"], temp["weight1991"]),
            "N": len(temp),
        })

opportunity_results = pd.DataFrame(opportunity_rows)
opportunity_results.to_csv(
    os.path.join(RESULTS_PATH, "figure6_realized_within_income.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9, 6))
income_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
opportunity_order = ["Low", "Middle", "High"]
offsets = {"Low": -0.18, "Middle": 0.00, "High": 0.18}
markers = {"Low": "o", "Middle": "s", "High": "^"}

for opportunity in opportunity_order:
    temp = (
        opportunity_results.loc[
            opportunity_results["Opportunity Group"] == opportunity
        ]
        .set_index("Income Quintile")
        .reindex(income_order)
    )

    x_pos = np.arange(5) + offsets[opportunity]
    point = temp["Realized Mobility"].to_numpy()

    ax.errorbar(
        x_pos,
        point,
        yerr=[
            point - temp["CI Lower"].to_numpy(),
            temp["CI Upper"].to_numpy() - point
        ],
        fmt=markers[opportunity],
        capsize=3,
        label=f"{opportunity} predicted opportunity"
    )

ax.set_xticks(np.arange(5))
ax.set_xticklabels(["Lowest", "Q2", "Q3", "Q4", "Highest"])
ax.set_xlabel("Parental-income quintile")
ax.set_ylabel(
    "Realized probability of exceeding\n"
    "the parental-generation median"
)
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure6_realized_mobility_within_income.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 22. FIGURES 7 AND 8: JOINT INCOME-WEALTH GRADIENTS
# ============================================================

def weighted_cell_mean_table(row_var, col_var):
    rows = []
    row_levels = ["Q1", "Q2", "Q3", "Q4"]
    col_levels = ["Q1", "Q2", "Q3", "Q4"]

    for r in row_levels:
        for c in col_levels:
            temp = analysis.loc[
                (analysis[row_var] == r)
                & (analysis[col_var] == c)
            ]
            if len(temp) == 0:
                continue
            rows.append({
                row_var: r,
                col_var: c,
                "Predicted":
                    weighted_mean(temp["Pred_Full"], temp["weight1991"])
            })

    return pd.DataFrame(rows)


# Figure 7: wealth groups within income quartiles.
f7 = weighted_cell_mean_table("IncomeQuartile", "WealthQuartile")

fig, ax = plt.subplots(figsize=(8, 5.5))
for wq in ["Q1", "Q2", "Q3", "Q4"]:
    temp = (
        f7.loc[f7["WealthQuartile"] == wq]
        .set_index("IncomeQuartile")
        .reindex(["Q1", "Q2", "Q3", "Q4"])
    )
    ax.plot(
        np.arange(1, 5),
        temp["Predicted"],
        marker="o",
        label=f"Wealth {wq}"
    )

ax.set_xticks(np.arange(1, 5))
ax.set_xlabel("Parental income quartile")
ax.set_ylabel("Predicted probability of upward mobility")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure7_similar_income_by_wealth.pdf"),
    bbox_inches="tight"
)
plt.show()

# Figure 8: income groups within wealth quartiles.
fig, ax = plt.subplots(figsize=(8, 5.5))
for iq in ["Q1", "Q2", "Q3", "Q4"]:
    temp = (
        f7.loc[f7["IncomeQuartile"] == iq]
        .set_index("WealthQuartile")
        .reindex(["Q1", "Q2", "Q3", "Q4"])
    )
    ax.plot(
        np.arange(1, 5),
        temp["Predicted"],
        marker="o",
        label=f"Income {iq}"
    )

ax.set_xticks(np.arange(1, 5))
ax.set_xlabel("Parental wealth quartile")
ax.set_ylabel("Predicted probability of upward mobility")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure8_similar_wealth_by_income.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 23. FIGURE 9: FULL VS INCOME-ONLY PREDICTIONS
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(
    analysis["Pred_Income"],
    analysis["Pred_Full"],
    alpha=0.35,
    s=14
)
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Predicted mobility: parental income only")
ax.set_ylabel("Predicted mobility: full family-background model")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure9_full_vs_income_only.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 24. FIGURE 10: DISTRIBUTION OF PREDICTION REVISIONS
# ============================================================

revision_q10, revision_q90 = weighted_quantile(
    analysis["PredictionRevision"],
    analysis["weight1991"],
    [0.10, 0.90]
)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.hist(
    analysis["PredictionRevision"],
    bins=30,
    weights=analysis["weight1991"],
    edgecolor="black",
    linewidth=0.4
)
ax.axvline(revision_q10, linestyle="--", linewidth=1, label="10th percentile")
ax.axvline(revision_q90, linestyle="--", linewidth=1, label="90th percentile")
ax.set_xlabel("Full-model prediction minus income-only prediction")
ax.set_ylabel("Weighted frequency")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure10_prediction_revisions.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 25. FIGURE 11: CHARACTERISTICS OF REVISION TAILS
# ============================================================

analysis["RevisionGroup"] = np.select(
    [
        analysis["PredictionRevision"] <= revision_q10,
        analysis["PredictionRevision"] >= revision_q90
    ],
    [
        "Lower than income-only",
        "Higher than income-only"
    ],
    default="Middle 80%"
)

revision_profile_vars = {
    "High wealth": "High Wealth",
    "College": "College",
    "Black": "African-American",
    "Married": "Married",
    "South": "South",
    "Female": "Female",
}

profile_rows = []

for group_name in ["Lower than income-only", "Middle 80%", "Higher than income-only"]:
    temp = analysis.loc[analysis["RevisionGroup"] == group_name]

    for label, var in revision_profile_vars.items():
        valid = temp[var].notna()
        profile_rows.append({
            "Revision Group": group_name,
            "Characteristic": label,
            "Share": weighted_mean(
                temp.loc[valid, var],
                temp.loc[valid, "weight1991"]
            )
        })

revision_profile = pd.DataFrame(profile_rows)
revision_profile.to_csv(
    os.path.join(RESULTS_PATH, "figure11_revision_profiles.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9, 6))
characteristics = list(revision_profile_vars.keys())
x = np.arange(len(characteristics))
width = 0.25

for j, group_name in enumerate(
    ["Lower than income-only", "Middle 80%", "Higher than income-only"]
):
    temp = (
        revision_profile.loc[revision_profile["Revision Group"] == group_name]
        .set_index("Characteristic")
        .reindex(characteristics)
    )

    ax.bar(
        x + (j - 1) * width,
        temp["Share"],
        width=width,
        label=group_name
    )

ax.set_xticks(x)
ax.set_xticklabels(characteristics, rotation=25, ha="right")
ax.set_ylabel("Weighted share of individuals")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure11_revision_characteristics.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 26. FIGURE 12: REVISION VALIDATION
#     DOES THE FULL-MODEL REVISION PREDICT REALIZED OUTCOMES
#     EVEN AMONG PEOPLE WITH SIMILAR INCOME-ONLY PREDICTIONS?
# ============================================================
# This is a stronger validation of "information beyond income" than simply
# showing that Pred_Full differs from Pred_Income.
#
# Step 1: place individuals into weighted quintiles of the INCOME-ONLY
#         out-of-fold predicted probability.
# Step 2: within each quintile, divide the full-minus-income revision into
#         weighted terciles.
# Step 3: plot weighted REALIZED mobility rates.
#
# If the high-revision group realizes more upward mobility than the
# low-revision group within the same income-only score range, the extra
# family-background information is predicting real subsequent differences.

analysis["IncomeOnlyScoreQuintile"] = assign_weighted_quantile_group(
    analysis["Pred_Income"],
    analysis["weight1991"],
    5,
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)

analysis["RevisionTercileWithinIncomeScore"] = np.nan

for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    mask = analysis["IncomeOnlyScoreQuintile"] == q
    temp = analysis.loc[mask]

    bins = assign_weighted_quantile_group(
        temp["PredictionRevision"],
        temp["weight1991"],
        3,
        labels=["Low revision", "Middle revision", "High revision"]
    )

    analysis.loc[mask, "RevisionTercileWithinIncomeScore"] = bins.to_numpy()

revision_validation_rows = []

for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    for revision_group in ["Low revision", "Middle revision", "High revision"]:
        temp = analysis.loc[
            (analysis["IncomeOnlyScoreQuintile"] == q)
            & (analysis["RevisionTercileWithinIncomeScore"] == revision_group)
        ].copy()

        if len(temp) == 0:
            continue

        point = weighted_mean(temp["MedianMobility"], temp["weight1991"])
        low, high = cluster_bootstrap_ci(
            temp,
            lambda z: weighted_mean(z["MedianMobility"], z["weight1991"]),
            reps=BOOT_REPS,
            seed=RANDOM_STATE + 400 + len(revision_validation_rows)
        )

        revision_validation_rows.append({
            "Income-only score quintile": q,
            "Revision group": revision_group,
            "Realized mobility": point,
            "Lower": low,
            "Upper": high,
            "N": len(temp),
        })

revision_validation = pd.DataFrame(revision_validation_rows)
revision_validation.to_csv(
    os.path.join(RESULTS_PATH, "figure12_revision_validation.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9, 6))
revision_order = ["Low revision", "Middle revision", "High revision"]
offsets = {"Low revision": -0.18, "Middle revision": 0.0, "High revision": 0.18}
markers = {"Low revision": "o", "Middle revision": "s", "High revision": "^"}

for revision_group in revision_order:
    temp = (
        revision_validation.loc[
            revision_validation["Revision group"] == revision_group
        ]
        .set_index("Income-only score quintile")
        .reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
    )

    point = temp["Realized mobility"].to_numpy()
    xpos = np.arange(5) + offsets[revision_group]

    ax.errorbar(
        xpos,
        point,
        yerr=[
            point - temp["Lower"].to_numpy(),
            temp["Upper"].to_numpy() - point
        ],
        fmt=markers[revision_group],
        capsize=3,
        label=revision_group
    )

ax.set_xticks(np.arange(5))
ax.set_xticklabels(["Lowest", "Q2", "Q3", "Q4", "Highest"])
ax.set_xlabel("Income-only predicted-mobility quintile")
ax.set_ylabel("Realized probability of upward mobility")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure12_revision_validation.pdf"),
    bbox_inches="tight"
)
plt.show()




# ============================================================
# 27. FIGURE 13: INCREMENTAL PREDICTIVE INFORMATION
# ============================================================

plot_labels = [
    "Income",
    "+ Wealth",
    "+ Education",
    "+ Demographics",
    "Full",
]

x = np.arange(len(table6_ci))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

auc = table6_ci["AUC"].to_numpy()
axes[0].errorbar(
    x,
    auc,
    yerr=[
        auc - table6_ci["AUC Lower"].to_numpy(),
        table6_ci["AUC Upper"].to_numpy() - auc
    ],
    fmt="o",
    capsize=4
)
axes[0].plot(x, auc, linewidth=1)
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot_labels, rotation=20, ha="right")
axes[0].set_ylabel("Cross-fitted weighted AUC")
axes[0].set_xlabel("Information set")

brier = table6_ci["Brier"].to_numpy()
axes[1].errorbar(
    x,
    brier,
    yerr=[
        brier - table6_ci["Brier Lower"].to_numpy(),
        table6_ci["Brier Upper"].to_numpy() - brier
    ],
    fmt="o",
    capsize=4
)
axes[1].plot(x, brier, linewidth=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(plot_labels, rotation=20, ha="right")
axes[1].set_ylabel("Cross-fitted weighted Brier score")
axes[1].set_xlabel("Information set")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure13_incremental_information.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 28. FIGURE 14: CHARACTERISTICS ACROSS PREDICTED MOBILITY
# ============================================================

pred_cut_probs = [0.10, 0.25, 0.50, 0.75, 0.90]
pred_cuts = weighted_quantile(
    analysis["Pred_Full"],
    analysis["weight1991"],
    pred_cut_probs
)

analysis["MobilityBin"] = pd.cut(
    analysis["Pred_Full"],
    bins=[-np.inf] + list(pred_cuts) + [np.inf],
    labels=["Bottom 10%", "10–25%", "25–50%", "50–75%", "75–90%", "Top 10%"]
)

profile_vars = {
    "African-American": "African-American",
    "College": "College",
    "High parental wealth": "High Wealth",
    "Married parents": "Married",
    "South": "South",
    "Female": "Female",
}

mobility_profile_rows = []

for group_name in analysis["MobilityBin"].cat.categories:
    temp = analysis.loc[analysis["MobilityBin"] == group_name]

    for label, var in profile_vars.items():
        valid = temp[var].notna()
        mobility_profile_rows.append({
            "Mobility Group": group_name,
            "Characteristic": label,
            "Share": weighted_mean(
                temp.loc[valid, var],
                temp.loc[valid, "weight1991"]
            )
        })

mobility_profile = pd.DataFrame(mobility_profile_rows)

fig, ax = plt.subplots(figsize=(9, 6))
for label in profile_vars.keys():
    temp = (
        mobility_profile.loc[mobility_profile["Characteristic"] == label]
        .set_index("Mobility Group")
        .reindex(analysis["MobilityBin"].cat.categories)
    )
    ax.plot(
        np.arange(len(temp)),
        temp["Share"],
        marker="o",
        linewidth=1.5,
        label=label
    )

ax.set_xticks(np.arange(6))
ax.set_xticklabels(analysis["MobilityBin"].cat.categories, rotation=20, ha="right")
ax.set_xlabel("Predicted mobility group")
ax.set_ylabel("Weighted share of individuals")
ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure14_mobility_profile.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 29. FIGURE 15: SOCIOECONOMIC MOBILITY GAPS
# ============================================================
# These are descriptive differences in cross-fitted predicted probabilities.
# The bootstrap resamples ID68 family clusters.

# Use true White indicator if it was saved from Stata.
if "White" in analysis.columns:
    white_mask = analysis["White"] == 1
    white_label = "White"
else:
    warnings.warn(
        "pWhite/White is not present. Black comparisons use 'non-Black'. "
        "For the final paper, save pWhite from Stata so the label can be White."
    )
    white_mask = analysis["African-American"] == 0
    white_label = "Non-Black"

main_comparisons = {
    "High wealth − Low wealth": (
        analysis["High Wealth"] == 1,
        analysis["High Wealth"] == 0
    ),
    "College − No college": (
        analysis["College"] == 1,
        analysis["College"] == 0
    ),
    "High advantage − Low advantage": (
        analysis["HighAdvantage"] == 1,
        analysis["LowAdvantage"] == 1
    ),
    f"Black − {white_label}": (
        analysis["African-American"] == 1,
        white_mask
    ),
    f"Black high wealth − {white_label} high wealth": (
        (analysis["African-American"] == 1) & (analysis["High Wealth"] == 1),
        white_mask & (analysis["High Wealth"] == 1)
    ),
    f"Black low wealth − {white_label} low wealth": (
        (analysis["African-American"] == 1) & (analysis["High Wealth"] == 0),
        white_mask & (analysis["High Wealth"] == 0)
    ),
}


def weighted_gap(df, g1_col, g0_col):
    a = df.loc[df[g1_col] == 1]
    b = df.loc[df[g0_col] == 1]
    return (
        weighted_mean(a["Pred_Full"], a["weight1991"])
        - weighted_mean(b["Pred_Full"], b["weight1991"])
    )


gap_rows = []

# Store masks as columns so they are automatically resampled by cluster bootstrap.
for j, (label, (g1, g0)) in enumerate(main_comparisons.items()):
    g1_col = f"_g1_{j}"
    g0_col = f"_g0_{j}"
    analysis[g1_col] = g1.astype(int)
    analysis[g0_col] = g0.astype(int)

    point = weighted_gap(analysis, g1_col, g0_col)

    low, high = cluster_bootstrap_ci(
        analysis,
        lambda z, a=g1_col, b=g0_col: weighted_gap(z, a, b),
        reps=BOOT_REPS,
        seed=RANDOM_STATE + 100 + j
    )

    gap_rows.append({
        "Comparison": label,
        "Difference": point,
        "Lower": low,
        "Upper": high,
    })

gap_results = pd.DataFrame(gap_rows).sort_values("Difference")
gap_results.to_csv(
    os.path.join(RESULTS_PATH, "figure15_socioeconomic_gaps.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9, 6))
y_pos = np.arange(len(gap_results))
x_val = gap_results["Difference"].to_numpy()

ax.errorbar(
    x_val,
    y_pos,
    xerr=[
        x_val - gap_results["Lower"].to_numpy(),
        gap_results["Upper"].to_numpy() - x_val
    ],
    fmt="o",
    capsize=4
)
ax.axvline(0, linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(gap_results["Comparison"])
ax.set_xlabel("Difference in predicted probability of upward mobility")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure15_mobility_gaps.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 30. FIGURE 16: RACIAL GAP ACROSS INCREASING INFORMATION SETS
# ============================================================
# This is a descriptive comparison of weighted mean cross-fitted probabilities
# between Black and White/non-Black observations under progressively richer
# information sets. It is NOT a causal decomposition.

race_prediction_specs = {
    "Unadjusted": None,
    "Income": "Pred_Income",
    "Income + wealth": "Pred_IncomeWealth",
    "Income + wealth + education": "Pred_IncomeWealthEducation",
}

race_gap_rows = []

for label, pred_col in race_prediction_specs.items():

    black = analysis.loc[analysis["African-American"] == 1]
    white = analysis.loc[white_mask]

    if pred_col is None:
        black_value = weighted_mean(black["MedianMobility"], black["weight1991"])
        white_value = weighted_mean(white["MedianMobility"], white["weight1991"])
    else:
        black_value = weighted_mean(black[pred_col], black["weight1991"])
        white_value = weighted_mean(white[pred_col], white["weight1991"])

    race_gap_rows.append({
        "Adjustment": label,
        "Black-minus-comparison gap": black_value - white_value,
    })

race_gap = pd.DataFrame(race_gap_rows)
race_gap.to_csv(
    os.path.join(RESULTS_PATH, "figure16_racial_gap_adjustments.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(
    np.arange(len(race_gap)),
    race_gap["Black-minus-comparison gap"],
    marker="o"
)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xticks(np.arange(len(race_gap)))
ax.set_xticklabels(race_gap["Adjustment"], rotation=20, ha="right")
ax.set_ylabel(f"Black − {white_label} difference in mobility")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure16_racial_gap_adjustments.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 31. FIGURES 17-19: ECONOMICALLY DISADVANTAGED FAMILIES
# ============================================================

analysis["Disadvantaged"] = (
    (analysis["IncomeQuartile"] == "Q1")
    & (analysis["WealthQuartile"] == "Q1")
)

disadv = analysis.loc[analysis["Disadvantaged"]].copy()

# Figure 16: distribution.
if len(disadv) > 0:
    cutoff = weighted_quantile(
        disadv["Pred_Full"],
        disadv["weight1991"],
        0.75
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.hist(
        disadv["Pred_Full"],
        bins=15,
        weights=disadv["weight1991"],
        edgecolor="black",
        linewidth=0.4
    )
    ax.axvline(cutoff, linestyle="--", linewidth=1.2, label="Top-quartile cutoff")
    ax.set_xlabel("Predicted probability of upward mobility")
    ax.set_ylabel("Weighted frequency")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "figure17_disadvantaged_distribution.pdf"),
        bbox_inches="tight"
    )
    plt.show()

    # Figure 17: high predicted mobility vs remainder.
    disadv["HighPredicted"] = disadv["Pred_Full"] >= cutoff

    disadvantage_vars = {
        "College": "College",
        "Some college": "Some College",
        "Married": "Married",
        "Female": "Female",
        "Black": "African-American",
        "Self-employed": "Self-employed",
    }

    f17_rows = []
    for group_name, group_value in [
        ("High predicted mobility", True),
        ("Other disadvantaged", False)
    ]:
        temp = disadv.loc[disadv["HighPredicted"] == group_value]

        for label, var in disadvantage_vars.items():
            f17_rows.append({
                "Group": group_name,
                "Characteristic": label,
                "Share": weighted_mean(temp[var], temp["weight1991"])
            })

    f17 = pd.DataFrame(f17_rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = list(disadvantage_vars.keys())
    x = np.arange(len(labels))
    width = 0.35

    for j, group_name in enumerate(["High predicted mobility", "Other disadvantaged"]):
        temp = (
            f17.loc[f17["Group"] == group_name]
            .set_index("Characteristic")
            .reindex(labels)
        )
        ax.bar(
            x + (j - 0.5) * width,
            temp["Share"],
            width=width,
            label=group_name
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Weighted share of individuals")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_PATH, "figure18_disadvantaged_characteristics.pdf"),
        bbox_inches="tight"
    )
    plt.show()


# Figure 18: calibration by economic background.
calibration_subgroup_rows = []

for group_name, mask in [
    ("Low income and low wealth", analysis["Disadvantaged"]),
    ("Other families", ~analysis["Disadvantaged"])
]:
    temp = analysis.loc[mask].copy()

    # Five groups are more stable for the smaller disadvantaged sample.
    cal = calibration_table(temp, "Pred_Full", n_bins=5)
    cal["Group"] = group_name
    calibration_subgroup_rows.append(cal)

calibration_subgroup = pd.concat(calibration_subgroup_rows, ignore_index=True)

fig, ax = plt.subplots(figsize=(7, 5.5))
for group_name in ["Low income and low wealth", "Other families"]:
    temp = calibration_subgroup.loc[calibration_subgroup["Group"] == group_name]
    ax.plot(
        temp["Mean predicted"],
        temp["Observed rate"],
        marker="o",
        label=group_name
    )

ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed mobility rate")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure19_calibration_by_background.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 32. CV-STRUCTURE ROBUSTNESS:
#     FAMILY-GROUPED VS ORDINARY INDIVIDUAL-LEVEL FOLDS
# ============================================================
# The main analysis uses family-grouped folds. This diagnostic quantifies how
# much performance would change if related observations were allowed to fall
# into different folds.
#
# To isolate fold structure, we reuse the five fold-specific RF configurations
# selected in the main grouped analysis rather than tuning again.

selected_rf_params = [
    json.loads(x)
    for x in rf_tuning["Best parameters"]
]

ordinary_cv = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

ordinary_folds = list(
    ordinary_cv.split(
        np.zeros(len(y)),
        y
    )
)

ordinary_pred = np.full(len(analysis), np.nan)

for fold_number, (train_idx, test_idx) in enumerate(ordinary_folds, start=1):
    X_train = analysis.iloc[train_idx][full_features]
    X_test = analysis.iloc[test_idx][full_features]

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    model = make_rf(
        selected_rf_params[fold_number - 1],
        seed=RANDOM_STATE + 9000 + fold_number
    )

    model.fit(
        X_train_i,
        y[train_idx],
        sample_weight=weights[train_idx]
    )

    ordinary_pred[test_idx] = model.predict_proba(X_test_i)[:, 1]

cv_robustness = pd.DataFrame([
    {
        "Cross-fitting structure": "Family-grouped by ID68",
        **weighted_metrics(y, analysis["Pred_Full"], weights)
    },
    {
        "Cross-fitting structure": "Individual-level stratified",
        **weighted_metrics(y, ordinary_pred, weights)
    }
])

cv_robustness.to_csv(
    os.path.join(RESULTS_PATH, "cv_structure_robustness.csv"),
    index=False
)

print("\n" + "=" * 100)
print("CV-STRUCTURE ROBUSTNESS")
print("=" * 100)
print(cv_robustness.round(3).to_string(index=False))


# ============================================================
# 33. APPENDIX A.6 FIGURE 22:
#     PARENTAL ECONOMIC RESOURCES AND PREDICTED MOBILITY
# ============================================================

def mean_prediction_by_decile(decile_col):
    rows = []
    for d in [f"D{i}" for i in range(1, 11)]:
        temp = analysis.loc[analysis[decile_col] == d]
        rows.append({
            "Decile": int(d[1:]),
            "Predicted": weighted_mean(temp["Pred_Full"], temp["weight1991"])
        })
    return pd.DataFrame(rows)


wealth_gradient = mean_prediction_by_decile("WealthDecile")
income_gradient = mean_prediction_by_decile("IncomeDecile")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
axes[0].plot(wealth_gradient["Decile"], wealth_gradient["Predicted"], marker="o")
axes[0].set_xlabel("Parental wealth decile")
axes[0].set_ylabel("Predicted probability of upward mobility")
axes[0].set_ylim(0, 1)

axes[1].plot(income_gradient["Decile"], income_gradient["Predicted"], marker="o")
axes[1].set_xlabel("Parental income decile")
axes[1].set_ylabel("Predicted probability of upward mobility")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure22_parental_resources.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 34. APPENDIX A.6 FIGURE 23:
#     WEALTH GRADIENT BY PARENTAL EDUCATION
# ============================================================

def education_label(row):
    if row["College"] == 1:
        return "College"
    if row["Some College"] == 1:
        return "Some college"
    if row["High School"] == 1:
        return "High school"
    if row["Some High School"] == 1:
        return "Some high school"
    return "Omitted/other education"

analysis["EducationGroup"] = analysis.apply(education_label, axis=1)

edu_order = ["Some high school", "High school", "Some college", "College"]
edu_wealth_rows = []

for edu in edu_order:
    for d in [f"D{i}" for i in range(1, 11)]:
        temp = analysis.loc[
            (analysis["EducationGroup"] == edu)
            & (analysis["WealthDecile"] == d)
        ]

        if len(temp) < 10:
            continue

        edu_wealth_rows.append({
            "Education": edu,
            "Wealth decile": int(d[1:]),
            "Predicted": weighted_mean(temp["Pred_Full"], temp["weight1991"]),
            "N": len(temp)
        })

edu_wealth = pd.DataFrame(edu_wealth_rows)

fig, ax = plt.subplots(figsize=(8, 5.5))
for edu in edu_order:
    temp = edu_wealth.loc[edu_wealth["Education"] == edu]
    ax.plot(
        temp["Wealth decile"],
        temp["Predicted"],
        marker="o",
        label=edu
    )

ax.set_xlabel("Parental wealth decile")
ax.set_ylabel("Predicted probability of upward mobility")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure23_wealth_by_education.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 35. APPENDIX A.6 FIGURE 24:
#     BOTTOM PREDICTED-MOBILITY TAIL BY WEALTH
# ============================================================

bottom10_cut = weighted_quantile(
    analysis["Pred_Full"],
    analysis["weight1991"],
    0.10
)
analysis["Bottom10Predicted"] = analysis["Pred_Full"] <= bottom10_cut

bottom_rows = []

for d in [f"D{i}" for i in range(1, 11)]:
    temp = analysis.loc[analysis["WealthDecile"] == d]
    bottom_rows.append({
        "Wealth decile": int(d[1:]),
        "Share": weighted_mean(temp["Bottom10Predicted"], temp["weight1991"])
    })

bottom_by_wealth = pd.DataFrame(bottom_rows)

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.plot(
    bottom_by_wealth["Wealth decile"],
    bottom_by_wealth["Share"],
    marker="o"
)
ax.set_xlabel("Parental wealth decile")
ax.set_ylabel("Probability of being in bottom 10% of predicted mobility")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure24_bottom_tail_by_wealth.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 36. APPENDIX A.6 FIGURE 25:
#     BETWEEN-GROUP VARIATION IN PREDICTED MOBILITY
# ============================================================
# This is a descriptive weighted eta-squared measure:
# between-group variance / total weighted variance.
# The measures are NOT additive because the dimensions overlap.

def weighted_eta_squared(values, group, weights):
    work = pd.DataFrame({
        "value": values,
        "group": group,
        "weight": weights
    }).dropna()

    overall = weighted_mean(work["value"], work["weight"])
    total = weighted_mean((work["value"] - overall) ** 2, work["weight"])

    if not np.isfinite(total) or total <= 0:
        return np.nan

    between_num = 0.0
    total_weight = work["weight"].sum()

    for _, temp in work.groupby("group", observed=False):
        wg = temp["weight"].sum()
        mg = weighted_mean(temp["value"], temp["weight"])
        between_num += wg * (mg - overall) ** 2

    between = between_num / total_weight
    return between / total


analysis["RaceGroup"] = np.where(
    analysis["African-American"] == 1,
    "Black",
    np.where(
        analysis.get("White", pd.Series(False, index=analysis.index)) == 1,
        "White",
        "Other/non-Black"
    )
)

analysis["RegionGroup"] = np.select(
    [
        analysis["Northeast"] == 1,
        analysis["Northcentral"] == 1,
        analysis["South"] == 1,
        analysis["West"] == 1,
    ],
    ["Northeast", "Northcentral", "South", "West"],
    default="Other"
)

eta_specs = {
    "Income quartile": "IncomeQuartile",
    "Wealth quartile": "WealthQuartile",
    "Education": "EducationGroup",
    "Race": "RaceGroup",
    "Region": "RegionGroup",
}

eta_rows = []

for label, group_col in eta_specs.items():
    eta_rows.append({
        "Dimension": label,
        "Share of variation":
            weighted_eta_squared(
                analysis["Pred_Full"],
                analysis[group_col],
                analysis["weight1991"]
            )
    })

eta = pd.DataFrame(eta_rows).sort_values("Share of variation")

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(eta["Dimension"], eta["Share of variation"])
ax.set_xlabel(
    "Share of variance in predicted mobility\n"
    "associated with between-group differences"
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure25_between_group_variation.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 37. APPENDIX A.6 FIGURE 26:
#     WEALTH GRADIENT BY REGION
# ============================================================

region_order = ["Northeast", "Northcentral", "South", "West"]
region_rows = []

for region in region_order:
    for d in [f"D{i}" for i in range(1, 11)]:
        temp = analysis.loc[
            (analysis["RegionGroup"] == region)
            & (analysis["WealthDecile"] == d)
        ]

        if len(temp) < 10:
            continue

        region_rows.append({
            "Region": region,
            "Wealth decile": int(d[1:]),
            "Predicted": weighted_mean(temp["Pred_Full"], temp["weight1991"]),
            "N": len(temp)
        })

region_wealth = pd.DataFrame(region_rows)

fig, ax = plt.subplots(figsize=(8, 5.5))
for region in region_order:
    temp = region_wealth.loc[region_wealth["Region"] == region]
    ax.plot(
        temp["Wealth decile"],
        temp["Predicted"],
        marker="o",
        label=region
    )

ax.set_xlabel("Parental wealth decile")
ax.set_ylabel("Predicted probability of upward mobility")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_PATH, "figure26_wealth_by_region.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 38. SAVE FINAL ANALYSIS DATA AND HEADLINE RESULTS
# ============================================================

analysis.to_csv(
    os.path.join(RESULTS_PATH, "final_crossfitted_analysis_data.csv"),
    index=False
)

income_metrics = weighted_metrics(
    y,
    analysis["Pred_Income"],
    weights
)
full_metrics = weighted_metrics(
    y,
    analysis["Pred_Full"],
    weights
)

headline = pd.DataFrame([{
    "Weighted parental median": parent_median,
    "Weighted realized mobility rate": weighted_mean(y, weights),
    "Income-only RF AUC": income_metrics["AUC"],
    "Full RF AUC": full_metrics["AUC"],
    "Delta AUC": full_metrics["AUC"] - income_metrics["AUC"],
    "Income-only RF Brier": income_metrics["Brier"],
    "Full RF Brier": full_metrics["Brier"],
    "Percent Brier reduction":
        100 * (income_metrics["Brier"] - full_metrics["Brier"])
        / income_metrics["Brier"],
    "Revision P10": revision_q10,
    "Revision P90": revision_q90,
}])

headline.to_csv(
    os.path.join(RESULTS_PATH, "headline_results.csv"),
    index=False
)

print("\n" + "=" * 100)
print("HEADLINE RESULTS")
print("=" * 100)
print(headline.round(3).to_string(index=False))

print("\nDONE.")
print(
    "Next: send me table5_model_performance.csv, "
    "table6_nested_information_sets.csv, headline_results.csv, "
    "cv_structure_robustness.csv, and figure12_revision_validation.csv."
)








