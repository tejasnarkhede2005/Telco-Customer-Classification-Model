import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# --- LOAD THE MODEL ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path):
    """Loads the pickled machine learning model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please make sure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model('AdaBoost_best_model.pkl')

# --- STYLING ---
def load_css():
    """Injects custom CSS for styling the app."""
    st.markdown("""
    <style>
        /* Main app styling */
        .stApp {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }

        /* Navbar styling */
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #16213e;
            padding: 1rem 2rem;
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            border-bottom: 2px solid #0f3460;
        }
        .navbar-brand {
            font-size: 1.5rem;
            font-weight: bold;
            color: #e94560;
            text-decoration: none;
        }
        .navbar-links a {
            color: #e0e0e0;
            margin-left: 20px;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        .navbar-links a:hover {
            color: #e94560;
        }

        /* Main content padding to avoid overlap with navbar */
        .main-content {
            padding-top: 80px;
        }

        /* Title styling */
        h1 {
            color: #e94560;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
        }
        
        h2, h3 {
            color: #e0e0e0;
        }

        /* Input widgets styling */
        .stSelectbox > div > div > div {
            background-color: #16213e;
        }
        .stNumberInput > div > div > input {
            background-color: #16213e;
            color: #e0e0e0;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #e94560;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }
        .stButton button:hover {
            background-color: #c73049;
            box-shadow: 0 6px 20px rgba(233, 69, 96, 0.5);
            transform: translateY(-2px);
        }

        /* Result display styling */
        .result-container {
            margin-top: 2rem;
            padding: 2rem;
            background-color: #16213e;
            border-radius: 15px;
            text-align: center;
            border: 1px solid #0f3460;
        }
        .result-text {
            font-size: 1.8rem;
            font-weight: bold;
        }
        .churn {
            color: #ff6b6b;
        }
        .no-churn {
            color: #4dff91;
        }
        
    </style>
    """, unsafe_allow_html=True)

# --- APP LAYOUT ---
load_css()

# Navbar
st.markdown("""
<div class="navbar">
    <a href="#" class="navbar-brand">🔮 ChurnPredict</a>
    <div class="navbar-links">
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content container
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.title("Customer Churn Prediction")
st.write("Fill in the customer details below to predict whether they will churn.")

# --- INPUT FORM ---
if model:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Has a Partner", ["No", "Yes"])
        Dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    with col2:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)

    with col3:
        st.subheader("Services Info")
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # --- PREDICTION LOGIC ---
    if st.button("Predict Churn"):
        # Mappings for categorical features
        # These need to match how the model was trained.
        gender_map = {'Male': 1, 'Female': 0}
        binary_map = {'No': 0, 'Yes': 1}
        multiline_map = {'No': 0, 'Yes': 1, 'No phone service': 2}
        online_service_map = {'No': 0, 'Yes': 1, 'No internet service': 2}
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        payment_map = {
            'Electronic check': 0, 
            'Mailed check': 1, 
            'Bank transfer (automatic)': 2, 
            'Credit card (automatic)': 3
        }

        # Create a dictionary of the input data
        input_data = {
            'gender': gender_map[gender],
            'SeniorCitizen': binary_map[SeniorCitizen],
            'Partner': binary_map[Partner],
            'Dependents': binary_map[Dependents],
            'tenure': tenure,
            'PhoneService': binary_map[PhoneService],
            'MultipleLines': multiline_map[MultipleLines],
            'OnlineSecurity': online_service_map[OnlineSecurity],
            'OnlineBackup': online_service_map[OnlineBackup],
            'DeviceProtection': online_service_map[DeviceProtection],
            'TechSupport': online_service_map[TechSupport],
            'StreamingTV': online_service_map[StreamingTV],
            'StreamingMovies': online_service_map[StreamingMovies],
            'Contract': contract_map[Contract],
            'PaperlessBilling': binary_map[PaperlessBilling],
            'PaymentMethod': payment_map[PaymentMethod],
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches the model's training data
        feature_order = model.feature_names_in_
        input_df = input_df[feature_order]

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Display result
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown('<p class="result-text churn">Prediction: This customer is likely to CHURN</p>', unsafe_allow_html=True)
            probability = prediction_proba[1] * 100
            st.write(f"**Probability of Churn:** {probability:.2f}%")
        else:
            st.markdown('<p class="result-text no-churn">Prediction: This customer is likely to STAY</p>', unsafe_allow_html=True)
            probability = prediction_proba[0] * 100
            st.write(f"**Probability of Staying:** {probability:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close main content container
