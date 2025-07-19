import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the pipeline
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load('pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Pipeline model file not found. Please ensure 'pipeline.pkl' is in the same directory.")
        return None

# Main app
def main():
    st.title("Customer Churn Prediction")
    st.markdown("---")
    
    # Load pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        st.stop()
    
    st.sidebar.header("Customer Information")
    
    column_names = ['Gender', 'Seniorcitizen', 'Partner', 'Dependents', 'Tenure',
                   'Phoneservice', 'Multiplelines', 'Internetservice', 'Onlinesecurity',
                   'Onlinebackup', 'Deviceprotection', 'Techsupport', 'Streamingtv',
                   'Streamingmovies', 'Contract', 'Paperlessbilling', 'Paymentmethod',
                   'Monthlycharges', 'Totalcharges']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 100, 12)
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col2:
        st.subheader("Security & Support")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
        st.subheader("Entertainment")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
    st.subheader("Contract & Payment")
    col3, col4 = st.columns(2)
    
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
        total_charges = monthly_charges * tenure
    
    st.markdown("---")
    
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        customer_data = pd.DataFrame([[
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service, online_security,
            online_backup, device_protection, tech_support, streaming_tv,
            streaming_movies, contract, paperless_billing, payment_method,
            monthly_charges, total_charges
        ]], columns=column_names)
        
        try:
            prediction = pipeline.predict(customer_data)
            prediction_proba = pipeline.predict_proba(customer_data)
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction[0] == 1:
                    st.error("⚠️ **HIGH RISK**: Customer likely to churn")
                    st.metric("Churn Probability", f"{prediction_proba[0][1]:.2%}")
                else:
                    st.success("✅ **LOW RISK**: Customer likely to stay")
                    st.metric("Retention Probability", f"{prediction_proba[0][0]:.2%}")
            
            with col_result2:
                prob_data = pd.DataFrame({
                    'Outcome': ['Stay', 'Churn'],
                    'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
                })
                st.bar_chart(prob_data.set_index('Outcome'))
            
            st.subheader("👤 Customer Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Monthly Charges", f"${monthly_charges:.2f}")
                st.metric("Total Charges", f"${total_charges:.2f}")
                
            with summary_col2:
                st.metric("Tenure", f"{tenure} months")
                st.metric("Contract Type", contract)
                
            with summary_col3:
                st.metric("Internet Service", internet_service)
                st.metric("Payment Method", payment_method)
            
            st.subheader("🔍 Risk Factors Analysis")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract increases churn risk")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment method is associated with higher churn")
            if tenure < 12:
                risk_factors.append("Low tenure (< 12 months) increases churn risk")
            if monthly_charges > 80:
                risk_factors.append("High monthly charges may lead to churn")
            if internet_service == "Fiber optic":
                risk_factors.append("Fiber optic customers sometimes have higher churn rates")
            
            if risk_factors:
                st.warning("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No significant risk factors identified.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Batch prediction section
    st.markdown("---")
    st.subheader("📁 Batch Prediction")
    st.write("Upload a CSV file with customer data for batch predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            # Display first few rows
            st.write("**Data Preview:**")
            st.dataframe(batch_data.head())
            
            if st.button("🔮 Predict Batch", type="secondary"):
                # Make batch predictions
                batch_predictions = pipeline.predict(batch_data)
                batch_probabilities = pipeline.predict_proba(batch_data)
                
                # Add predictions to dataframe
                results_df = batch_data.copy()
                results_df['Churn_Prediction'] = batch_predictions
                results_df['Churn_Probability'] = batch_probabilities[:, 1]
                results_df['Retention_Probability'] = batch_probabilities[:, 0]
                
                # Display results
                st.write("**Batch Prediction Results:**")
                st.dataframe(results_df)
                
                # Summary statistics
                churn_count = sum(batch_predictions)
                total_count = len(batch_predictions)
                
                st.metric("Predicted Churns", f"{churn_count} out of {total_count}")
                st.metric("Churn Rate", f"{(churn_count/total_count)*100:.1f}%")
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()