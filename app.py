
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Page setup
# -----------------------------
st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI-Enabled Battery Digital Twin")
st.subheader("Temperature Prediction and Battery Monitoring")

st.write(
    "This prototype uses recorded battery data to predict the next "
    "temperature window. It is not connected to live sensors."
)

DATA_FILE = Path("data/battery_window_features.csv")

FEATURES = [
    "Step",
    "Avg_Current",
    "Current_Std",
    "Avg_Voltage",
    "Voltage_Change",
    "Temperature_Std",
    "Temperature_Change",
    "Discharge_Ah",
    "Discharge_Energy_Wh",
    "Avg_Temperature",
]

# -----------------------------
# 2. Load data and train models
# -----------------------------
@st.cache_resource
def load_and_train():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}"
        )

    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(["Step", "Start_Time"]).copy()

    # Target = temperature in the next window of the same step
    df["Target_Temperature"] = (
        df.groupby("Step")["Avg_Temperature"].shift(-1)
    )

    df = df.dropna(subset=FEATURES + ["Target_Temperature"]).copy()

    # Chronological 80/20 split within each step
    df["Row_Number"] = df.groupby("Step").cumcount()
    df["Group_Size"] = df.groupby("Step")["Step"].transform("size")
    df["Train_Row_Limit"] = (df["Group_Size"] * 0.8).apply(int)

    train = df[df["Row_Number"] < df["Train_Row_Limit"]].copy()
    test = df[df["Row_Number"] >= df["Train_Row_Limit"]].copy()

    X_train = train[FEATURES]
    y_train = train["Target_Temperature"]
    X_test = test[FEATURES]
    y_test = test["Target_Temperature"]

    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    predictions = {}
    metrics = []

    # Evaluate each model on unseen chronological test rows
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred

        metrics.append({
            "Algorithm": name,
            "MAE (°C)": mean_absolute_error(y_test, pred),
            "RMSE (°C)": mean_squared_error(y_test, pred) ** 0.5,
            "R²": r2_score(y_test, pred)
        })

    # Persistence baseline: next temperature = current temperature
    baseline_pred = X_test["Avg_Temperature"].to_numpy()
    predictions["Persistence Baseline"] = baseline_pred

    metrics.append({
        "Algorithm": "Persistence Baseline",
        "MAE (°C)": mean_absolute_error(y_test, baseline_pred),
        "RMSE (°C)": mean_squared_error(y_test, baseline_pred) ** 0.5,
        "R²": r2_score(y_test, baseline_pred)
    })

    comparison = pd.DataFrame(metrics).sort_values("MAE (°C)")

    return df, models, comparison, X_test, y_test, predictions


try:
    df, models, comparison, X_test, y_test, predictions = load_and_train()
except Exception as error:
    st.error(f"Could not load the dataset or train models: {error}")
    st.stop()


# -----------------------------
# 3. Dataset overview
# -----------------------------
st.markdown("### Dataset overview")

c1, c2, c3 = st.columns(3)
c1.metric("Available data windows", len(pd.read_csv(DATA_FILE)))
c2.metric("Battery steps", df["Step"].nunique())
c3.metric("Test records", len(y_test))

st.caption(
    "Model training is performed locally using the recorded feature dataset."
)

# -----------------------------
# 4. User input panel
# -----------------------------
st.markdown("### Enter battery measurements")

st.info(
    "Use measurements from the same dataset and units for meaningful results. "
    "The default values are taken from the dataset."
)

with st.form("prediction_form"):
    left, middle, right = st.columns(3)

    with left:
        step_value = st.selectbox(
            "Battery step",
            options=sorted(df["Step"].unique().tolist())
        )
        avg_current = st.number_input(
            "Average current (A)",
            value=float(df["Avg_Current"].median()),
            format="%.5f"
        )
        current_std = st.number_input(
            "Current standard deviation (A)",
            value=float(df["Current_Std"].median()),
            min_value=0.0,
            format="%.5f"
        )
        avg_voltage = st.number_input(
            "Average voltage (V)",
            value=float(df["Avg_Voltage"].median()),
            format="%.5f"
        )

    with middle:
        voltage_change = st.number_input(
            "Voltage change (V)",
            value=float(df["Voltage_Change"].median()),
            format="%.5f"
        )
        temperature_std = st.number_input(
            "Temperature standard deviation (°C)",
            value=float(df["Temperature_Std"].median()),
            min_value=0.0,
            format="%.5f"
        )
        temperature_change = st.number_input(
            "Temperature change (°C)",
            value=float(df["Temperature_Change"].median()),
            format="%.5f"
        )

    with right:
        discharge_ah = st.number_input(
            "Discharge capacity (Ah)",
            value=float(df["Discharge_Ah"].median()),
            format="%.5f"
        )
        discharge_energy = st.number_input(
            "Discharge energy (Wh)",
            value=float(df["Discharge_Energy_Wh"].median()),
            format="%.5f"
        )
        avg_temperature = st.number_input(
            "Current average temperature (°C)",
            value=float(df["Avg_Temperature"].median()),
            format="%.5f"
        )

    submitted = st.form_submit_button("Predict next-window temperature")


# -----------------------------
# 5. Show predictions
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Step": step_value,
        "Avg_Current": avg_current,
        "Current_Std": current_std,
        "Avg_Voltage": avg_voltage,
        "Voltage_Change": voltage_change,
        "Temperature_Std": temperature_std,
        "Temperature_Change": temperature_change,
        "Discharge_Ah": discharge_ah,
        "Discharge_Energy_Wh": discharge_energy,
        "Avg_Temperature": avg_temperature,
    }], columns=FEATURES)

    st.markdown("### Predicted next-window temperature")

    results = {
        "Gradient Boosting": float(
            models["Gradient Boosting"].predict(input_data)[0]
        ),
        "Random Forest": float(
            models["Random Forest"].predict(input_data)[0]
        ),
        "Persistence Baseline": float(avg_temperature),
    }

    a, b, c = st.columns(3)
    a.metric("Gradient Boosting", f"{results['Gradient Boosting']:.3f} °C")
    b.metric("Random Forest", f"{results['Random Forest']:.3f} °C")
    c.metric("Persistence baseline", f"{results['Persistence Baseline']:.3f} °C")

    st.dataframe(
        pd.DataFrame({
            "Method": list(results.keys()),
            "Predicted temperature (°C)": list(results.values())
        }),
        hide_index=True,
        width="stretch"
    )

    fig, ax = plt.subplots()
    ax.bar(list(results.keys()), list(results.values()))
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Next-window temperature prediction")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        "These are model predictions, not actual future measurements. "
        "A real prediction error can only be calculated after the next "
        "window's temperature is measured."
    )


# -----------------------------
# 6. Test-set model comparison
# -----------------------------
st.markdown("### Model comparison on the test dataset")

st.write(
    "MAE and RMSE measure prediction error in °C; lower is better. "
    "R² measures how well predictions fit the test observations; higher is generally better."
)

display_metrics = comparison.copy()
for col in ["MAE (°C)", "RMSE (°C)", "R²"]:
    display_metrics[col] = display_metrics[col].round(5)

st.dataframe(display_metrics, hide_index=True, width="stretch")

best_model = comparison.iloc[0]
st.success(
    f"Lowest test MAE in this run: {best_model['Algorithm']} "
    f"({best_model['MAE (°C)']:.5f} °C)."
)

# -----------------------------
# 7. Actual vs predicted graph
# -----------------------------
st.markdown("### Actual vs predicted temperature")

plot_df = pd.DataFrame({
    "Actual temperature": y_test.to_numpy(),
    **predictions
}).reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(
    plot_df.index,
    plot_df["Actual temperature"],
    label="Actual",
    marker="o",
    markersize=3
)

for name in predictions:
    ax2.plot(
        plot_df.index,
        plot_df[name],
        label=name,
        alpha=0.85
    )

ax2.set_xlabel("Test sample")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title("Actual vs predicted next-window temperature")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)

# -----------------------------
# 8. Download results
# -----------------------------
st.download_button(
    "Download model comparison CSV",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="battery_model_comparison.csv",
    mime="text/csv"
)

st.download_button(
    "Download test predictions CSV",
    data=plot_df.to_csv(index=False).encode("utf-8"),
    file_name="battery_test_predictions.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "Research prototype: offline dataset-based prediction. "
    "Live hardware acquisition, validated battery safety limits, "
    "and adaptive charging control are not implemented by this demo."
)
