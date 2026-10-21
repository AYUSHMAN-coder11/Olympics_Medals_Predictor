import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Page configuration
st.set_page_config(
    page_title="MedalCast | Predicted vs Actual Medals",
    layout="wide"
)

# 2. Load saved model artifacts
APP_DIR = Path(__file__).resolve().parent


def find_file(file_name):
    # Look next to app.py, then inside the unzipped olympic_medal_predictor folder
    for folder in (APP_DIR, APP_DIR / "olympic_medal_predictor"):
        if (folder / file_name).exists():
            return folder / file_name

    raise FileNotFoundError(file_name)


@st.cache_resource
def load_model():
    model = joblib.load(find_file("medal_model.pkl"))
    feature_cols = joblib.load(find_file("feature_cols.pkl"))
    comparison = pd.read_csv(find_file("olympic_comparison.csv"))

    return model, feature_cols, comparison


try:
    rf_model, feature_cols, comparison = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Place medal_model.pkl, feature_cols.pkl and "
        "olympic_comparison.csv (or the unzipped olympic_medal_predictor "
        "folder) in the same folder as app.py."
    )
    st.stop()


# 3. Prepare the comparison data for display
comparison = comparison.copy()
comparison["Year"] = comparison["Year"].astype(int)

comparison = comparison.rename(columns={
    "actual_medals": "Actual medals",
    "predicted_medals": "Predicted medals",
    "difference": "Difference"
})

years = sorted(comparison["Year"].unique().tolist(), reverse=True)

country_names = (
    comparison.drop_duplicates("NOC")
    .set_index("NOC")["Country"]
)

DISPLAY_COLUMNS = [
    "Year", "Country", "Actual medals", "Predicted medals", "Difference"
]


def country_label(noc):
    return f"{country_names[noc]} ({noc})"


# 4. Comparison and prediction functions
def accuracy_summary(data):
    errors = data["Predicted medals"] - data["Actual medals"]

    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    within_3 = (errors.abs() <= 3).mean() * 100

    return mae, rmse, within_3


def top_countries(year, n=10):
    return (
        comparison[comparison["Year"] == year]
        .sort_values("Actual medals", ascending=False)
        .head(n)
    )


def compare_country(noc):
    return (
        comparison[comparison["NOC"] == noc]
        .sort_values("Year")
    )


# 5. Design the application UI
st.title("MedalCast")
st.write(
    "Compare the medals the model predicted with the medals each country "
    f"actually won at the {' and '.join(str(y) for y in sorted(years))} "
    "Summer Olympics."
)

mae, rmse, within_3 = accuracy_summary(comparison)

col1, col2, col3 = st.columns(3)

col1.metric("Average miss (MAE)", f"{mae:.2f} medals")
col2.metric("RMSE", f"{rmse:.2f} medals")
col3.metric("Predictions within 3 medals", f"{within_3:.0f}%")

st.divider()

top_tab, country_tab = st.tabs([
    "Top Countries",
    "Country Lookup"
])

# 6. Top countries: predicted vs actual
with top_tab:
    st.header("Predicted vs actual for the top medal winners")

    selected_year = st.selectbox(
        "Choose the Games",
        years,
        key="top_year"
    )

    top_n = st.slider(
        "Number of countries",
        min_value=5,
        max_value=20,
        value=10,
        key="top_n"
    )

    if st.button("Compare", type="primary", key="top_button"):
        results = top_countries(selected_year, top_n)

        st.markdown(f"### Top {top_n} countries by actual medals, {selected_year}")

        st.bar_chart(
            results.set_index("Country")[["Actual medals", "Predicted medals"]]
        )

        st.dataframe(
            results[DISPLAY_COLUMNS],
            hide_index=True
        )

# 7. Country lookup
with country_tab:
    st.header("Look up one country")

    noc_options = sorted(
        country_names.index.tolist(),
        key=lambda noc: country_names[noc]
    )

    default_index = noc_options.index("USA") if "USA" in noc_options else 0

    selected_noc = st.selectbox(
        "Choose a country",
        noc_options,
        index=default_index,
        format_func=country_label,
        key="selected_country"
    )

    if st.button("Show Comparison", type="primary", key="country_button"):
        results = compare_country(selected_noc)

        st.markdown(f"### {country_names[selected_noc]}")

        columns = st.columns(len(results))

        for column, (_, row) in zip(columns, results.iterrows()):
            column.metric(
                f"{int(row['Year'])}: actual {int(row['Actual medals'])}",
                f"{row['Predicted medals']:.1f} predicted",
                f"{row['Difference']:+.1f} vs actual",
                delta_color="off"
            )

        st.dataframe(
            results[DISPLAY_COLUMNS],
            hide_index=True
        )

st.divider()

st.caption(
    f"About the model: Random Forest with {rf_model.n_estimators} trees that "
    "predicts the change from the previous Games' medals, using "
    + ", ".join(name.replace("_", " ") for name in feature_cols)
    + "."
)

st.caption(
    "Predictions use only each country's previous Games, so large changes "
    "are hard to foresee. Positive differences mean the model predicted "
    "more medals than the country won."
)
