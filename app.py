# Core Streamlit and Data Handling
import streamlit as st # The main Streamlit library for building web apps
import pandas as pd     # For data manipulation and analysis
import numpy as np      # For numerical operations

# File System and Archiving (for downloading/unzipping data if using Kaggle API)
import os       # For interacting with the operating system (e.g., creating directories)
import zipfile  # For handling zip files (unzipping the Kaggle dataset)
import kaggle   # Kaggle API client to download datasets directly

# Machine Learning - Preprocessing
from sklearn.model_selection import train_test_split   # For splitting data into training/testing sets
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For scaling numerical features and encoding categorical features
from sklearn.compose import ColumnTransformer         # For applying different transformations to different columns
import joblib                                         # For saving and loading Python objects (models, preprocessors)

# Machine Learning - Supervised Models
from sklearn.linear_model import LogisticRegression   # For Logistic Regression model
from sklearn.tree import DecisionTreeClassifier       # For Decision Tree model
from sklearn.ensemble import RandomForestClassifier   # For Random Forest model
import xgboost as xgb                                 # For XGBoost Classifier
import lightgbm as lgb                                # For LightGBM Classifier

# Machine Learning - Unsupervised Models (Clustering)
from sklearn.cluster import KMeans                    # For K-Means clustering

# Machine Learning - Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,       # Overall accuracy
    precision_score,      # Precision for positive class
    recall_score,         # Recall for positive class
    f1_score,             # F1-score for positive class
    confusion_matrix,     # Confusion Matrix for detailed performance
    classification_report, # Comprehensive report of classification metrics
    silhouette_score      # For evaluating clustering quality
)

# Dimensionality Reduction (for Visualization)
from sklearn.decomposition import PCA                 # Principal Component Analysis for reducing dimensions for plotting

# Plotting and Visualization
import matplotlib.pyplot as plt # Matplotlib for basic static plots (used for confusion matrix, but Plotly preferred)
import seaborn as sns           # Seaborn for enhanced statistical plots (built on Matplotlib)
import plotly.express as px     # Plotly Express for interactive, high-level plots
import plotly.graph_objects as go # Plotly Graph Objects for more granular control over plots


# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Customer Churn & Sales Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

st.title("📊 Customer Churn Prediction and Sales Dashboard")
st.markdown("---")


# --- Utility Functions for Data Loading and Preprocessing (Cached for Performance) ---

@st.cache_resource # Cache the function itself, its return value, and resources
def download_and_load_data():
    """
    Downloads the Telco Customer Churn dataset from Kaggle and loads it.
    This runs only once, then caches the result.
    """
    dataset_name = 'blastchar/telco-customer-churn'
    download_path = './kaggle_data_temp/'
    os.makedirs(download_path, exist_ok=True)

    csv_file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    extracted_csv_path = os.path.join(download_path, csv_file_name)

    if os.path.exists(extracted_csv_path):
        pass # Keep silent if already loaded
    else:
        try:
            zip_file_path = os.path.join(download_path, f"{dataset_name.split('/')[-1]}.zip")
            if not os.path.exists(zip_file_path):
                kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=False)
            else:
                pass # Keep silent if already downloaded

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)

        except Exception as e:
            st.error(f"Error downloading or unzipping data: {e}. Please ensure your Kaggle API key is set up correctly.")
            st.info("You can set up your Kaggle API key by placing `kaggle.json` in `~/.kaggle/` (your user home directory).")
            st.stop() # Stop execution if data cannot be loaded

    try:
        df = pd.read_csv(extracted_csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}. Please check the file path and integrity. ❌")
        st.stop()


@st.cache_data # Cache the output of this function based on its inputs
def preprocess_data(df_raw):
    """
    Performs ETL operations, handles missing values, encodes categorical variables,
    and scales numerical features. Returns processed X, y, and the updated original-like df.
    """
    df = df_raw.copy()

    # Handle 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # Drop Customer ID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Encode 'Churn' target variable
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        st.warning("Churn column not found. Dashboard features will be limited.")
        df['Churn'] = 0 # Dummy churn column for dashboard if not present

    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1, errors='ignore')
    y = df['Churn']

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor and transform X
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding for model interpretation/reconstruction
    new_column_names = numerical_cols + \
                       list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

    X_processed_df = pd.DataFrame(X_processed, columns=new_column_names, index=X.index)

    return X_processed_df, y, df, preprocessor # Return preprocessor too


@st.cache_resource
def train_churn_model(X_processed, y, model_name="LightGBM"):
    """
    Trains a selected churn prediction model and returns the model, metrics, and test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    model = None
    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, solver='liblinear')
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    elif model_name == "LightGBM":
        model = lgb.LGBMClassifier(objective='binary', random_state=42)
    else:
        st.error("Invalid model selected.")
        return None, None, None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }
    return model, metrics, X_test, y_test


@st.cache_data
def run_kmeans_clustering(X_processed, n_clusters=3):
    """
    Performs K-Means clustering and returns cluster labels.
    """
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(X_processed)
    silhouette_avg = silhouette_score(X_processed, clusters) if n_clusters > 1 else 0
    return clusters, silhouette_avg


# --- Main Application Logic ---

# 1. Load and Preprocess Data
with st.spinner("Loading and preprocessing data... ⏳"):
    df_raw = download_and_load_data()
    X_processed, y, df, preprocessor = preprocess_data(df_raw.copy())

# Add a 'Cluster' column to the main df for dashboard use, if not already present
if 'Cluster' not in df.columns:
    with st.spinner("Running initial K-Means clustering... 💡"):
        initial_clusters, _ = run_kmeans_clustering(X_processed, n_clusters=3)
        df['Cluster'] = initial_clusters


# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Churn Prediction", "Sales Analysis", "Customer Segmentation", "Predict New Customer"])
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("This dashboard leverages Data Analytics and Machine Learning for Customer Churn Prediction and Sales Insights.")

# --- Page: Home ---
if page == "Home":
    st.header("Welcome to the Customer Churn & Sales Dashboard! 👋")
    st.markdown("""
    This interactive dashboard provides a comprehensive view of customer behavior, sales trends,
    and churn probability using the **Telco Customer Churn dataset**.
    Leverage machine learning models and data analytics to enhance customer retention and optimize sales strategies.
    """)

    st.markdown("---")
    st.subheader("Dataset Overview 📊")
    col_ov1, col_ov2, col_ov3 = st.columns(3)
    with col_ov1:
        st.metric("Total Customers", df_raw.shape[0], help="Total number of customer records in the dataset.")
    with col_ov2:
        st.metric("Number of Features", df_raw.shape[1], help="Total number of attributes describing each customer.")
    with col_ov3:
        churn_rate = df_raw['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%", help="Percentage of customers who have churned.")

    st.markdown("---")
    st.subheader("Quick Actions & Insights ✨")
    st.markdown("""
    - **Churn Prediction:** Dive into model performance and evaluate different algorithms.
    - **Sales Analysis:** Understand revenue patterns and service popularity.
    - **Customer Segmentation:** Discover unique customer groups for targeted strategies.
    - **Predict New Customer:** Test the churn prediction model with custom inputs.
    """)

    st.subheader("Raw Data Snapshot 📋")
    st.dataframe(df_raw.head(10)) # Show a few more rows

    st.markdown("---")
    st.subheader("Overall Churn Distribution 📉")
    churn_counts_home = df['Churn'].value_counts()
    fig_pie_home = px.pie(names=churn_counts_home.index.map({0: 'No Churn', 1: 'Churn'}),
                          values=churn_counts_home.values,
                          title='Distribution of Churn Status',
                          color_discrete_sequence=px.colors.qualitative.D3,
                          hole=0.3) # Add a hole for donut chart effect
    fig_pie_home.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie_home, use_container_width=True)

# --- Page: Churn Prediction ---
elif page == "Churn Prediction":
    st.header("Churn Prediction Models 🤖")
    st.write("Train and evaluate different supervised machine learning models to predict customer churn. Compare their performance to find the best fit for your business needs.")

    model_choice = st.selectbox(
        "Select Model for Training and Evaluation:",
        ("LightGBM", "Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"),
        key="model_selection_predict"
    )

    if y is None:
        st.warning("Churn column not found in data for prediction. Cannot train models.")
    else:
        with st.spinner(f"Training and evaluating {model_choice} model..."):
            model, metrics, X_test_eval, y_test_eval = train_churn_model(X_processed, y, model_choice)

        if model:
            st.subheader(f"{model_choice} Model Performance Overview")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
            col2.metric("Precision", f"{metrics['Precision']:.2f}")
            col3.metric("Recall", f"{metrics['Recall']:.2f}")
            col4.metric("F1-Score", f"{metrics['F1-Score']:.2f}")

            st.markdown("---")
            st.subheader("Detailed Model Evaluation")

            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.markdown("#### Confusion Matrix 📈")
                # Using Plotly for Confusion Matrix for better interactivity
                cm_data = metrics['Confusion Matrix']
                fig_cm = px.imshow(cm_data,
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['No Churn', 'Churn'],
                                   y=['No Churn', 'Churn'],
                                   color_continuous_scale='Blues',
                                   text_auto=True) # Automatically show text on heatmap
                fig_cm.update_layout(title_text='<b>Confusion Matrix</b>', title_x=0.5)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_eval2:
                st.markdown("#### Classification Report 📊")
                # Display classification report in a more readable format
                report_df = pd.DataFrame(metrics['Classification Report']).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))

            # Feature Importance/Coefficients in an expander
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                with st.expander("View Feature Importance/Coefficients"):
                    feature_names = X_processed.columns.tolist()
                    if hasattr(model, 'feature_importances_'): # Tree-based models
                        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
                        fig_fi = px.bar(importances.head(15), x=importances.head(15).values, y=importances.head(15).index,
                                        orientation='h', title='Top 15 Feature Importances',
                                        labels={'x': 'Importance', 'y': 'Feature'},
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_fi, use_container_width=True)
                    elif hasattr(model, 'coef_'): # Linear models
                        coef_values = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                        coefficients = pd.Series(coef_values, index=feature_names).sort_values(key=abs, ascending=False)
                        fig_coef = px.bar(coefficients.head(15), x=coefficients.head(15).values, y=coefficients.head(15).index,
                                        orientation='h', title='Top 15 Feature Coefficients (Magnitude)',
                                        labels={'x': 'Coefficient Value', 'y': 'Feature'},
                                        color_discrete_sequence=px.colors.qualitative.Plotly)
                        st.plotly_chart(fig_coef, use_container_width=True)

# --- Page: Sales Analysis ---
elif page == "Sales Analysis":
    st.header("Sales & Revenue Trend Analysis 💰")
    st.write("Explore patterns in customer charges, service subscriptions, and their correlation with churn to optimize revenue strategies.")

    st.subheader("Monthly and Total Charges Distribution �")
    col_charge1, col_charge2 = st.columns(2)
    with col_charge1:
        fig_mc = px.histogram(df, x='MonthlyCharges', nbins=50, title='Distribution of Monthly Charges',
                              color_discrete_sequence=['#636EFA']) # Plotly default blue
        st.plotly_chart(fig_mc, use_container_width=True)
    with col_charge2:
        fig_tc = px.histogram(df, x='TotalCharges', nbins=50, title='Distribution of Total Charges',
                              color_discrete_sequence=['#EF553B']) # Plotly default red
        st.plotly_chart(fig_tc, use_container_width=True)

    st.markdown("---")
    st.subheader("Churn Rate by Key Service Attributes 📉")
    col_contract, col_internet = st.columns(2)
    with col_contract:
        churn_by_contract = df.groupby('Contract')['Churn'].mean().reset_index()
        fig_contract = px.bar(churn_by_contract, x='Contract', y='Churn',
                              title='Churn Rate by Contract Type',
                              labels={'Churn': 'Churn Rate (Proportion)'},
                              color='Churn', color_continuous_scale=px.colors.sequential.Plasma_r) # Reverse plasma for churn
        st.plotly_chart(fig_contract, use_container_width=True)
    with col_internet:
        churn_by_internet = df.groupby('InternetService')['Churn'].mean().reset_index()
        fig_internet = px.bar(churn_by_internet, x='InternetService', y='Churn',
                              title='Churn Rate by Internet Service Type',
                              labels={'Churn': 'Churn Rate (Proportion)'},
                              color='Churn', color_continuous_scale=px.colors.sequential.Plasma_r)
        st.plotly_chart(fig_internet, use_container_width=True)

    st.markdown("---")
    st.subheader("Service Popularity & Revenue Impact 🌟")
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies']
    service_counts = {}
    for col in service_cols:
        if col == 'InternetService':
            service_counts[col] = df[df[col].isin(['DSL', 'Fiber optic'])].shape[0]
        else:
            service_counts[col] = df[df[col] == 'Yes'].shape[0]
    service_popularity = pd.Series(service_counts).sort_values(ascending=False)

    fig_svc_pop = px.bar(x=service_popularity.index, y=service_popularity.values,
                         title='Popularity of Services (Number of Customers Subscribed)',
                         labels={'x': 'Service', 'y': 'Number of Customers'},
                         color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_svc_pop, use_container_width=True)

    st.markdown("---")
    st.subheader("Tenure vs. Monthly Charges by Churn Status 🤝")
    fig_tenure_charges = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                                    title='Tenure vs. Monthly Charges by Churn Status',
                                    labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)'},
                                    color_discrete_map={0: 'green', 1: 'red'},
                                    hover_data=['Contract', 'PaymentMethod'])
    st.plotly_chart(fig_tenure_charges, use_container_width=True)


# --- Page: Customer Segmentation ---
elif page == "Customer Segmentation":
    st.header("Customer Segmentation Insights 👥")
    st.write("Understand your customer base by identifying distinct segments using K-Means clustering. This helps in tailoring marketing and retention strategies.")

    st.sidebar.subheader("Segmentation Settings")
    k_clusters = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=8, value=3, key="k_slider")

    with st.spinner(f"Running K-Means clustering with K={k_clusters}..."):
        clusters, silhouette_avg = run_kmeans_clustering(X_processed, k_clusters)
        df['Cluster'] = clusters # Update the main df with new cluster assignments

    st.subheader(f"Customer Segments (K={k_clusters}) Overview")
    st.info(f"The Silhouette Score for this clustering is: **{silhouette_avg:.2f}**. A higher score indicates better-defined and more separated clusters.")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)
    df_pca['Cluster'] = df['Cluster']
    df_pca['Churn'] = df['Churn'].map({0: 'No Churn', 1: 'Churn'}) # For better legend

    # Add original numerical columns to df_pca for hover data
    df_pca['MonthlyCharges'] = df['MonthlyCharges']
    df_pca['TotalCharges'] = df['TotalCharges']
    df_pca['tenure'] = df['tenure']

    fig_clusters = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                              title=f'Customer Clusters (K-Means, K={k_clusters}) - PCA Reduced Dimensions',
                              hover_data={'Cluster': True, 'Churn': True, 'MonthlyCharges': True, 'TotalCharges': True, 'tenure': True},
                              color_continuous_scale=px.colors.qualitative.Safe)
    st.plotly_chart(fig_clusters, use_container_width=True)

    st.markdown("---")
    st.subheader("Cluster Characteristics & Churn Rates 🔍")
    st.write("Analyze the average metrics and distribution of key features within each cluster to understand their unique profiles.")

    cluster_churn_rate = df.groupby('Cluster')['Churn'].mean().reset_index()
    fig_cluster_churn = px.bar(cluster_churn_rate, x='Cluster', y='Churn',
                                title='Churn Rate by Customer Segment',
                                labels={'Churn': 'Churn Rate (Proportion)'},
                                color='Churn', color_continuous_scale=px.colors.sequential.Sunset_r) # Reverse for churn
    st.plotly_chart(fig_cluster_churn, use_container_width=True)

    st.markdown("##### Average Financial & Tenure Metrics per Cluster")
    cluster_financial_summary = df.groupby('Cluster')[['MonthlyCharges', 'TotalCharges', 'tenure']].mean()
    st.dataframe(cluster_financial_summary.style.format("{:.2f}"))

    st.markdown("##### Contract Type Distribution per Cluster")
    contract_cluster_dist = df.groupby('Cluster')['Contract'].value_counts(normalize=True).unstack(fill_value=0)
    st.dataframe(contract_cluster_dist.style.format("{:.2%}"))

    st.markdown("##### Internet Service Distribution per Cluster")
    internet_cluster_dist = df.groupby('Cluster')['InternetService'].value_counts(normalize=True).unstack(fill_value=0)
    st.dataframe(internet_cluster_dist.style.format("{:.2%}"))

    st.info("""
    **💡 Interpreting Clusters for Targeted Marketing:**
    * **High Churn Clusters:** Identify segments with higher average churn rates. These are your priority for retention campaigns.
    * **Value Segments:** Understand which segments contribute most to revenue (e.g., high Total/Monthly Charges).
    * **Behavioral Differences:** Observe distinct patterns in contract types, internet services, etc., to tailor marketing messages and offers.
    * For example, a cluster with high churn, short tenure, and month-to-month contracts might be targeted with long-term contract incentives.
    """)


# --- Page: Predict New Customer ---
elif page == "Predict New Customer":
    st.header("Predict Churn for a New Customer 🔮")
    st.write("Enter details for a hypothetical customer to get their churn prediction based on our trained models.")

    st.markdown("---")
    st.subheader("Customer Information Input")

    # Create input fields for each feature
    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        gender = st.selectbox('Gender', ['Male', 'Female'], key='gender_input')
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'], key='senior_citizen_input')
        partner = st.selectbox('Partner', ['Yes', 'No'], key='partner_input')
        dependents = st.selectbox('Dependents', ['Yes', 'No'], key='dependents_input')
        tenure = st.slider('Tenure (months)', 0, 72, 1, key='tenure_input')
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'], key='phone_service_input')
        multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'], key='multiple_lines_input')

    with col_input2:
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'], key='internet_service_input')
        online_security = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'], key='online_security_input')
        online_backup = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'], key='online_backup_input')
        device_protection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'], key='device_protection_input')
        tech_support = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'], key='tech_support_input')
        streaming_tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'], key='streaming_tv_input')

    with col_input3:
        streaming_movies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'], key='streaming_movies_input')
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'], key='contract_input')
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'], key='paperless_billing_input')
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], key='payment_method_input')
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0, key='monthly_charges_input')
        total_charges = st.number_input('Total Charges', min_value=0.0, value=50.0, key='total_charges_input')

    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([customer_data])

    st.markdown("---")
    st.subheader("Prediction Result")
    # Using the LightGBM model as default for prediction
    default_model_name = "LightGBM"
    current_model, _, _, _ = train_churn_model(X_processed, y, default_model_name)

    if st.button("Predict Churn", key='predict_button'):
        if current_model is None:
            st.error(f"Churn prediction model ({default_model_name}) is not trained yet. Please go to 'Churn Prediction' page and train a model first.")
        else:
            try:
                # Ensure the order of columns in input_df matches the training data's original columns
                original_feature_columns = [col for col in df_raw.columns if col not in ['customerID', 'Churn']]
                input_df_reordered = input_df[original_feature_columns]

                input_processed = preprocessor.transform(input_df_reordered)

                prediction = current_model.predict(input_processed)[0]
                prediction_proba = current_model.predict_proba(input_processed)[0][1]

                if prediction == 1:
                    st.error(f"**Prediction: CHURN (High Risk) 🔴**")
                    st.markdown(f"**Probability of Churn:** `{prediction_proba:.2f}`")
                    st.warning("This customer is highly likely to churn. Consider proactive retention strategies!")
                else:
                    st.success(f"**Prediction: NO CHURN (Low Risk) 🟢**")
                    st.markdown(f"**Probability of Churn:** `{prediction_proba:.2f}`")
                    st.info("This customer is likely to remain. Continue to monitor and ensure satisfaction.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}. Please verify the input values.")
                st.warning("Ensure all necessary features are provided and match the expected format used during model training.")
