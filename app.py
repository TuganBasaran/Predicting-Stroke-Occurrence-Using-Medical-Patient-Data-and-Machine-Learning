import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import custom class for unpickling
from train_models import StrokeFeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model_bundle():
    """Load the trained model bundle"""
    try:
        bundle = joblib.load('best_stroke_model.pkl')
        return bundle
    except FileNotFoundError:
        st.error("⚠️ Model file 'best_stroke_model.pkl' not found! Please train the model first by running: python train_models.py")
        return None

def create_gauge_chart(probability, threshold):
    """Create a gauge chart for risk visualization"""
    
    # Determine risk level and color
    if probability < threshold:
        risk_level = "Low Risk"
        color = "green"
    elif probability < (threshold + (1-threshold)/2):
        risk_level = "Medium Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stroke Risk Probability"},
        delta = {'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [threshold*100, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    # Başlık ve margin ayarları ile taşmayı önle
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20), title_font_size=22)
    return fig, risk_level

def main():
    # Sidebar
    st.sidebar.image("image_icon.png", width=100)
    st.sidebar.title("Patient Data Input")
    st.sidebar.markdown("Enter patient details below:")
    # Daha pratik form: radio, slider, number_input
    gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
    age = st.sidebar.slider("Age", min_value=0, max_value=120, value=50)
    hypertension = st.sidebar.radio("Hypertension", ["No", "Yes"], horizontal=True)
    heart_disease = st.sidebar.radio("Heart Disease", ["No", "Yes"], horizontal=True)
    ever_married = st.sidebar.radio("Ever Married", ["No", "Yes"], horizontal=True)
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.radio("Residence Type", ["Urban", "Rural"], horizontal=True)
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submit_button = st.sidebar.button("Analyze Risk")

    # Main Content
    st.title("🧠 Stroke Prediction System")
    st.markdown("### Advanced Machine Learning Risk Assessment")
    bundle = load_model_bundle()
    if bundle is not None:
        model = bundle['model']
        threshold = bundle['threshold']
        # Kullanıcı dostu açıklama ve rehber
        st.info(
            """
            This tool estimates your risk of stroke based on the health information you provide.\n\n
            • If your result is high risk, please consult a healthcare professional.\n
            • Results are for informational purposes only and do not replace medical diagnosis.\n
            • Factors such as age, high blood pressure, heart disease, high blood sugar, and smoking can increase your risk.
            """
        )

        if submit_button:
            # Preprocess Input (Map to integers as per StrokeDataset)
            # Mappings from data_factory.py
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            married_map = {"No": 0, "Yes": 1}
            work_map = {"Never_worked": 0, "children": 1, "Self-employed": 2, "Private": 3, "Govt_job": 4}
            residence_map = {"Urban": 0, "Rural": 1}
            smoking_map = {"never smoked": 0, "Unknown": 1, "formerly smoked": 2, "smokes": 3}
            binary_map = {"No": 0, "Yes": 1}

            # Create DataFrame with correct column order
            # Order: ['gender','age','hypertension','heart_disease','ever_married','work_type',
            #         'Residence_type','avg_glucose_level','bmi','smoking_status']
            
            input_data = pd.DataFrame({
                'gender': [gender_map[gender]],
                'age': [age],
                'hypertension': [binary_map[hypertension]],
                'heart_disease': [binary_map[heart_disease]],
                'ever_married': [married_map[ever_married]],
                'work_type': [work_map[work_type]],
                'Residence_type': [residence_map[residence_type]],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_map[smoking_status]]
            })

            # Predict
            with st.spinner("Analyzing patient data..."):
                # Get probability
                probability = model.predict_proba(input_data)[:, 1][0]
                
                # Make decision based on optimized threshold
                prediction = 1 if probability >= threshold else 0
                
                # Visualizations
                col1, col2 = st.columns([2, 1])
                fig, risk_level = create_gauge_chart(probability, threshold)
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("### Analysis Result")
                    if prediction == 1:
                        st.error("**Prediction: STROKE RISK**")
                        st.markdown("The model predicts a high likelihood of stroke based on the provided data.")
                    else:
                        st.success("**Prediction: NO STROKE**")
                        st.markdown("The model predicts a low likelihood of stroke.")
                    st.markdown("---")
                    st.markdown(f"**Risk Level:** {risk_level}")
                    st.markdown(f"**Probability:** {probability:.1%}")

            # Feature Contribution (Simplified)
            st.markdown("### Key Risk Factors")
            st.info("Note: This analysis highlights factors contributing to the risk score.")
            
            # Simple heuristic for explanation (since we can't easily get SHAP values in real-time without heavy deps)
            factors = []
            if age > 60: factors.append("Age > 60")
            if hypertension == "Yes": factors.append("Hypertension")
            if heart_disease == "Yes": factors.append("Heart Disease")
            if avg_glucose_level > 140: factors.append("High Glucose Levels")
            if bmi > 30: factors.append("Obesity (BMI > 30)")
            if smoking_status in ["smokes", "formerly smoked"]: factors.append("Smoking History")
            
            if factors:
                st.write("The following factors contributed to increased risk:")
                for factor in factors:
                    st.markdown(f"- ⚠️ {factor}")
            else:
                st.write("No major high-risk factors identified in the input.")

if __name__ == "__main__":
    main()
