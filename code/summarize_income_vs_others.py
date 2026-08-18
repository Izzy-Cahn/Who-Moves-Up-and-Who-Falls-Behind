from pathlib import Path

import numpy as np
import pandas as pd


# Repository-relative input path
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "A7_A8_importance_summary.csv"


# Read saved predictor-group importance results
df = pd.read_csv(INPUT)

required_columns = {
    "Family",
    "Outcome",
    "Direction",
    "Threshold",
    "Model",
    "Method",
    "Group",
    "Value",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


# Impurity and SHAP importance are normalized within each
# outcome-model specification and can therefore be aggregated
# into Income versus All Other Characteristics.
#
# Permutation importance is intentionally excluded because
# groupwise permutation losses are not additive.
x = df[df["Method"].isin(["Impurity", "SHAP"])].copy()

specification_columns = [
    "Family",
    "Outcome",
    "Direction",
    "Threshold",
    "Model",
    "Method",
]

# Verify that the saved importance values are normalized.
totals = x.groupby(specification_columns)["Value"].sum()
if not np.allclose(totals.to_numpy(), 1.0, atol=1e-6):
    raise ValueError(
        "Impurity/SHAP importance does not sum to one within every "
        "outcome-model specification."
    )


# Collapse the eight predictor groups into:
#   1. Income
#   2. All Other Characteristics
x["Category"] = np.where(
    x["Group"].eq("Income"),
    "Income",
    "All Other Characteristics",
)

by_specification = (
    x.groupby(
        specification_columns + ["Category"],
        as_index=False,
    )["Value"]
    .sum()
)


# Average across the four thresholds and the two ML methods
# within each mobility definition and direction.
summary = (
    by_specification.groupby(
        ["Family", "Direction", "Method", "Category"],
        as_index=False,
    )["Value"]
    .mean()
)

wide = summary.pivot(
    index=["Family", "Direction"],
    columns=["Method", "Category"],
    values="Value",
)


# Final table values, in percentage points.
table = pd.DataFrame(
    {
        "Family": wide.index.get_level_values("Family"),
        "Direction": wide.index.get_level_values("Direction"),
        "Impurity: Income": 100 * wide[("Impurity", "Income")].to_numpy(),
        "Impurity: All Others": 100
        * wide[("Impurity", "All Other Characteristics")].to_numpy(),
        "SHAP: Income": 100 * wide[("SHAP", "Income")].to_numpy(),
        "SHAP: All Others": 100
        * wide[("SHAP", "All Other Characteristics")].to_numpy(),
    }
)


# Match the ordering used in the paper.
family_order = [
    "Relative",
    "Parent-child absolute",
    "Benchmark",
]

direction_order = [
    "Up",
    "Down",
]

table["Family"] = pd.Categorical(
    table["Family"],
    categories=family_order,
    ordered=True,
)

table["Direction"] = pd.Categorical(
    table["Direction"],
    categories=direction_order,
    ordered=True,
)

table = (
    table.sort_values(["Family", "Direction"])
    .reset_index(drop=True)
)

# Round only for the manuscript table.
table[
    [
        "Impurity: Income",
        "Impurity: All Others",
        "SHAP: Income",
        "SHAP: All Others",
    ]
] = table[
    [
        "Impurity: Income",
        "Impurity: All Others",
        "SHAP: Income",
        "SHAP: All Others",
    ]
].round(1)
