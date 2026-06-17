import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Global Conflict Escalation Radar", layout="wide")

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/best_xgb_model.pkl')
        features = joblib.load('models/feature_names.pkl')
        pair_stats = joblib.load('models/pair_stats.pkl')
        df = pd.read_csv('data/processed/features_labels.csv')
        df['SQLDATE'] = pd.to_datetime(df['SQLDATE'])
        return model, features, pair_stats, df
    except FileNotFoundError:
        return None, None, None, None

model, feature_names, pair_stats, df = load_artifacts()

if model is None:
    st.error("Artifacts missing! Please run scripts 01, 02, and 03 to generate the data and models.")
    st.stop()

st.title("🌍 Global Conflict Escalation Radar")
st.markdown("A **Dual-Engine Matrix** evaluating both **Breakout Acceleration** and **Absolute Structural Hostility**.")

# Get Latest Data Snapshot
latest_date = df['SQLDATE'].max()
st.subheader(f"Risk Assessment as of: {latest_date.strftime('%Y-%m-%d')}")

latest_data = df[df['SQLDATE'] == latest_date].copy()

if latest_data.empty:
    st.warning("No data available for the latest date.")
    st.stop()

def map_dashboard_pair_stats(pid):
    stats = pair_stats.get(pid, {'Pair_Historical_Freq': 0, 'Pair_Escalation_Rate': 0.0})
    return pd.Series([stats['Pair_Historical_Freq'], stats['Pair_Escalation_Rate']])

latest_data[['Pair_Historical_Freq', 'Pair_Escalation_Rate']] = latest_data['Pair_ID'].apply(map_dashboard_pair_stats)

# Enforce strict column alignment for the model
X_dash = latest_data.reindex(columns=feature_names, fill_value=0)

# Generate True Probabilities
probs = model.predict_proba(X_dash)[:, 1]
latest_data['Escalation_Probability'] = probs

# Sort by Risk Shock Probability
latest_data = latest_data.sort_values(by='Escalation_Probability', ascending=False)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Active Corridors Monitored", len(latest_data))
col2.metric("Critical Alerts (>25% Risk)", len(latest_data[latest_data['Escalation_Probability'] > 0.25]))
col3.metric("Highest Structural Risk", latest_data.iloc[0]['Pair_ID'])

st.divider()

# Interactive Hybrid Table
st.subheader("Tactical Watchlist")

display_df = latest_data[['Pair_ID', 'V_b', 'Escalation_Probability', 'Tone_7d_mean', 'Total_Events']].copy()
display_df['V_b'] = display_df['V_b'].round(1).astype(str) + ' /week'
display_df['Tone_7d_mean'] = display_df['Tone_7d_mean'].round(2)
display_df['Escalation_Probability'] = (display_df['Escalation_Probability'] * 100).round(1).astype(str) + '%'

display_df = display_df.rename(columns={
    'Pair_ID': 'Corridor',
    'V_b': '30-Day Weekly Baseline (V_b)',
    'Escalation_Probability': '7-Day Escalation Risk',
    'Tone_7d_mean': '7-Day Diplomatic Tone',
    'Total_Events': 'Events Today'
})

st.dataframe(display_df.head(20), use_container_width=True)

st.divider()
st.subheader("Structural Driver Analysis (SHAP)")

selected_pair = st.selectbox("Select a country pair to analyze:", latest_data['Pair_ID'].head(10))

if selected_pair:
    pair_row = latest_data[latest_data['Pair_ID'] == selected_pair]
    X_single = pair_row.reindex(columns=feature_names, fill_value=0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)

    st.write(f"Analyzing acceleration drivers for **{selected_pair}**")

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_single.iloc[0],
        feature_names=feature_names
    ), show=False)

    st.pyplot(fig)# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import os

# st.set_page_config(page_title="Global Conflict Escalation Radar", layout="wide")

# @st.cache_resource
# def load_artifacts():
#     try:
#         model = joblib.load('models/best_xgb_model.pkl')
#         features = joblib.load('models/feature_names.pkl')
#         pair_stats = joblib.load('models/pair_stats.pkl')
#         df = pd.read_csv('data/processed/features_labels.csv')
#         df['SQLDATE'] = pd.to_datetime(df['SQLDATE'])
#         return model, features, pair_stats, df
#     except FileNotFoundError:
#         return None, None, None, None

# model, feature_names, pair_stats, df = load_artifacts()

# if model is None:
#     st.error("Artifacts missing! Please run scripts 01, 02, and 03 to generate the data and models.")
#     st.stop()

# st.title("🌍 Global Conflict Escalation Radar")
# st.markdown("A Hybrid Threat Matrix evaluating both **Current Tension Baseline** and **Probability of Future Shock**.")

# # Get Latest Data Snapshot
# latest_date = df['SQLDATE'].max()
# st.subheader(f"Risk Assessment as of: {latest_date.strftime('%Y-%m-%d')}")

# latest_data = df[df['SQLDATE'] == latest_date].copy()

# if latest_data.empty:
#     st.warning("No data available for the latest date.")
#     st.stop()

# # Reapply Pair Identity Metrics
# def map_dashboard_pair_stats(pid):
#     stats = pair_stats.get(pid, {'Pair_Historical_Freq': 0, 'Pair_Escalation_Rate': 0.0})
#     return pd.Series([stats['Pair_Historical_Freq'], stats['Pair_Escalation_Rate']])

# latest_data[['Pair_Historical_Freq', 'Pair_Escalation_Rate']] = latest_data['Pair_ID'].apply(map_dashboard_pair_stats)

# # Enforce strict column alignment for the model
# X_dash = latest_data.reindex(columns=feature_names, fill_value=0)

# # Generate True Probabilities
# probs = model.predict_proba(X_dash)[:, 1]
# latest_data['Escalation_Probability'] = probs

# # Sort by Risk Shock Probability
# latest_data = latest_data.sort_values(by='Escalation_Probability', ascending=False)

# # KPIs
# col1, col2, col3 = st.columns(3)
# col1.metric("Active Corridors Monitored", len(latest_data))
# col2.metric("High Risk Shock Alerts (>20%)", len(latest_data[latest_data['Escalation_Probability'] > 0.20]))
# col3.metric("Highest Shock Risk Pair", latest_data.iloc[0]['Pair_ID'])

# st.divider()

# # Interactive Hybrid Table
# st.subheader("High-Risk Watchlist")

# display_df = latest_data[['Pair_ID', 'Baseline_30d_mean', 'Escalation_Probability', 'Avg_Tone', 'Goldstein_3d_mean']].copy()
# display_df['Baseline_30d_mean'] = display_df['Baseline_30d_mean'].round(1).astype(str) + ' /day'
# display_df['Escalation_Probability'] = (display_df['Escalation_Probability'] * 100).round(1).astype(str) + '%'

# display_df = display_df.rename(columns={
#     'Baseline_30d_mean': 'Current Velocity (Baseline Events)',
#     'Escalation_Probability': '7-Day Shock Risk (Acceleration)',
#     'Avg_Tone': 'Current Tone Sentiment',
#     'Goldstein_3d_mean': 'Recent Goldstein Score'
# })

# st.dataframe(display_df.head(20), use_container_width=True)

# # SHAP EXPLANATION
# st.divider()
# st.subheader("Risk Driver Analysis (SHAP)")

# selected_pair = st.selectbox("Select a country pair to analyze:", latest_data['Pair_ID'].head(10))

# if selected_pair:
#     pair_row = latest_data[latest_data['Pair_ID'] == selected_pair]
#     X_single = pair_row.reindex(columns=feature_names, fill_value=0)

#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_single)

#     st.write(f"Analyzing acceleration drivers for **{selected_pair}**")

#     fig, ax = plt.subplots(figsize=(8, 4))
#     shap.waterfall_plot(shap.Explanation(
#         values=shap_values[0],
#         base_values=explainer.expected_value,
#         data=X_single.iloc[0],
#         feature_names=feature_names
#     ), show=False)

#     st.pyplot(fig)
