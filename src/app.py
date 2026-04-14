import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 { color: #fff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.65); margin: 0.4rem 0 0; font-size: 1rem; }

    .risk-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
    }
    .risk-card h2 { font-size: 2rem; margin: 0; }
    .risk-card p  { font-size: 1.1rem; opacity: 0.9; margin: 0.5rem 0 0; }

    .metric-box {
        background: #f8f9ff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-box .val { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .metric-box .lbl { font-size: 0.85rem; color: #64748b; margin-top: 0.2rem; }

    .stSlider > div > div > div { background: #0f3460 !important; }
    .sidebar-tip {
        background: #f0f4ff;
        border-left: 4px solid #0f3460;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #1e293b;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path  = 'model/model.pkl'
    scaler_path = 'model/scaler.pkl'
    if not os.path.exists(model_path):
        st.error("❌ model/model.pkl not found. Please run `python src/model_training.py` first.")
        st.stop()
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🩺 Smart Health Risk Predictor</h1>
  <p>Enter your health metrics below to assess your diabetes risk using a trained ML model.</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Your Health Metrics")
    st.markdown("Adjust the sliders to match your values.")

    pregnancies  = st.slider("Pregnancies",               0, 17,  1)
    glucose      = st.slider("Glucose (mg/dL)",           44, 200, 110)
    blood_press  = st.slider("Blood Pressure (mmHg)",     24, 122, 72)
    skin_thick   = st.slider("Skin Thickness (mm)",        7, 99,  23)
    insulin      = st.slider("Insulin (μU/mL)",           14, 850, 80)
    bmi          = st.slider("BMI",                       18.0, 67.0, 26.0, step=0.1)
    dpf          = st.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47, step=0.01)
    age          = st.slider("Age",                       21, 81, 30)

    st.markdown("""
    <div class="sidebar-tip">
    💡 <strong>Tip:</strong> Glucose and BMI are the two strongest predictors in this model.
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict My Risk", use_container_width=True, type="primary")

# ─── Prediction ───────────────────────────────────────────────────────────────
input_data = np.array([[pregnancies, glucose, blood_press, skin_thick,
                         insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)
prediction   = model.predict(input_scaled)[0]
probability  = model.predict_proba(input_scaled)[0][1]

col1, col2 = st.columns([1.2, 1])

with col1:
    if predict_btn or True:   # always show result
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-card risk-high">
              <h2>⚠️ High Risk</h2>
              <p>Estimated diabetes risk probability: <strong>{probability*100:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-card risk-low">
              <h2>✅ Low Risk</h2>
              <p>Estimated diabetes risk probability: <strong>{probability*100:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)

        # Risk meter
        st.markdown("#### Risk Probability Meter")
        fig_meter, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(['Risk'], [probability],       color='#e74c3c', height=0.5)
        ax.barh(['Risk'], [1 - probability],   color='#2ecc71', height=0.5,
                left=probability)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='#1a1a2e', linewidth=1.5, linestyle='--', alpha=0.6)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(left=False)
        patches = [mpatches.Patch(color='#e74c3c', label='High Risk'),
                   mpatches.Patch(color='#2ecc71', label='Low Risk')]
        ax.legend(handles=patches, loc='upper right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_meter)
        plt.close()

with col2:
    st.markdown("#### 📊 Your Input Summary")
    summary_df = pd.DataFrame({
        'Metric': FEATURES,
        'Your Value': [pregnancies, glucose, blood_press, skin_thick,
                       insulin, bmi, dpf, age]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─── Feature Importance ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 What Drives the Prediction?")

imp_col1, imp_col2 = st.columns(2)

with imp_col1:
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    features_sorted = [FEATURES[i] for i in indices]
    imp_sorted      = importances[indices]

    fig_imp, ax = plt.subplots(figsize=(7, 4))
    colors = ['#e74c3c' if i == (len(indices)-1) else
              '#f39c12' if i >= (len(indices)-3) else '#3498db'
              for i in range(len(indices))]
    ax.barh(features_sorted, imp_sorted, color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance (Random Forest)')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close()

with imp_col2:
    st.markdown("""
    **How to read this chart:**

    - 🔴 **Glucose** is almost always the most important feature — high blood sugar is the primary indicator of diabetes.
    - 🟠 **BMI** and **Age** are next — excess weight and older age significantly raise risk.
    - 🔵 **Other features** like pregnancies and insulin levels add supporting signals.

    ---
    **About the model:**
    - Algorithm: Random Forest (200 trees)
    - Dataset: Pima Indians Diabetes (768 samples)
    - Trained accuracy: ~78–82%
    - Evaluation metric: ROC-AUC

    ---
    > ⚠️ *This tool is for educational purposes only. Always consult a licensed physician for medical advice.*
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#94a3b8; font-size:0.85rem;'>"
    "Built with Python · scikit-learn · Streamlit &nbsp;|&nbsp; "
    "Dataset: Pima Indians Diabetes (UCI / Kaggle)"
    "</center>",
    unsafe_allow_html=True
)
