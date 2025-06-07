import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import time

# --- Streamlit Page Configuration ---
# Sets up the basic configuration of the Streamlit page.
# page_title: Title that appears in the browser tab.
# layout: "wide" uses the full width of the browser window.
st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("💳 Fraudulent Transaction Detector")

# --- Data Loading Function ---
@st.cache_data(show_spinner=False)
# Caches the data loading operation.
# This prevents the app from reloading the CSV every time an interaction occurs,
# making it much faster for large files.
# show_spinner=False: hides the default Streamlit spinner, as we'll use a custom one.
def load_data(file):
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        file: The uploaded CSV file object from Streamlit's file_uploader.

    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file)

# --- Model Auto-Selection Logic ---
def auto_select_model(X, y):
    """
    Automatically selects a suitable machine learning model based on dataset characteristics.
    The selection criteria are heuristic and designed to provide a reasonable default.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series (Class column).

    Returns:
        str: The name of the selected model ("XGBoost", "Random Forest", or "Logistic Regression").
    """
    imbalance_ratio = y.sum() / len(y)  # Fraction of fraud samples (Class = 1)
    num_features = X.shape[1]
    num_rows = len(X)

    # If the imbalance is very high (less than 1% fraud), prioritize XGBoost.
    # XGBoost is generally very robust for highly imbalanced datasets.
    if imbalance_ratio < 0.01:
        return "XGBoost"
    # For datasets with very few features, Logistic Regression is simple and interpretable.
    elif num_features <= 10:
        return "Logistic Regression"
    # For smaller datasets (fewer than 10,000 rows), Random Forest is a good balance
    # of performance and robustness without being overly complex.
    elif num_rows < 10000:
        return "Random Forest"
    # For larger datasets or other cases, default to XGBoost for its general high performance.
    else:
        return "XGBoost"

# --- Model Training Function ---
@st.cache_resource(show_spinner=False)
# Caches the trained model and associated metrics.
# This prevents retraining the model on every interaction once the 'Run Model' button is pressed.
def train_model(X, y, model_name):
    """
    Trains the selected machine learning model, handles data imbalance,
    and calculates evaluation metrics.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series (Class column).
        model_name (str): The name of the model to train ("Random Forest",
                          "Logistic Regression", or "XGBoost").

    Returns:
        tuple: A tuple containing:
            - model: The trained machine learning model.
            - report (dict): Classification report.
            - matrix (np.array): Confusion matrix.
            - fpr (np.array): False positive rates for ROC curve.
            - tpr (np.array): True positive rates for ROC curve.
            - roc_auc (float): Area under the ROC curve.
            - X_full (pd.DataFrame): Original full feature set (unresampled).
            - y_full (pd.Series): Original full target set (unresampled).
            - X_test (pd.DataFrame): Test set features (from resampled data).
            - y_test (pd.Series): Test set targets (from resampled data).
            - y_pred (np.array): Predicted labels for the test set.
    """
    # Initialize SMOTEENN for handling class imbalance.
    # Combines SMOTE (Synthetic Minority Over-sampling Technique) with Edited Nearest Neighbours (ENN).
    # SMOTE oversamples the minority class, and ENN cleans up noise and overlaps.
    sm = SMOTEENN(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Split the resampled data into training and testing sets.
    # stratify=y_resampled ensures that the proportion of classes is the same in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )

    # Initialize the chosen model with appropriate parameters.
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,     # Number of trees in the forest.
            max_depth=10,         # Maximum depth of the tree.
            class_weight='balanced', # Automatically adjusts weights inversely proportional to class frequencies.
            random_state=42       # For reproducibility.
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            class_weight='balanced', # Handles imbalance.
            max_iter=1000         # Maximum number of iterations for the solver to converge.
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            use_label_encoder=False, # Suppresses a common deprecation warning.
            eval_metric='logloss',   # Evaluation metric used for validation.
            # scale_pos_weight: Helps handle imbalanced datasets by scaling the positive class weight.
            # A value of 5 gives more importance to the minority class (fraud).
            scale_pos_weight=5
        )

    # Train the model on the resampled training data.
    model.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = model.predict(X_test)
    # Get probability estimates for the positive class (fraud).
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics.
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_probs) # False Positive Rate, True Positive Rate for ROC curve
    roc_auc = auc(fpr, tpr) # Area Under the Receiver Operating Characteristic Curve

    return model, report, matrix, fpr, tpr, roc_auc, X, y, X_test, y_test, y_pred

# --- Streamlit UI: File Uploader ---
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

# --- Main Logic when a File is Uploaded ---
if uploaded_file:
    try:
        # Load the uploaded data
        df = load_data(uploaded_file)

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head(10))

        # Check for the mandatory 'Class' column
        if "Class" not in df.columns:
            st.error("CSV must contain a 'Class' column (0 = legit, 1 = fraud)")
        else:
            # --- Data Preprocessing ---
            # Scale the 'Amount' column using RobustScaler.
            # RobustScaler is robust to outliers, which is common for financial amounts.
            df["Amount"] = RobustScaler().fit_transform(df[["Amount"]])
            # Drop the 'Time' column if it exists, as raw time isn't usually a direct feature.
            if "Time" in df.columns:
                df = df.drop(columns=["Time"])

            # Define features (X) and target (y).
            X = df.drop("Class", axis=1)
            y = df["Class"]

            # --- Model Selection UI ---
            # Allows the user to choose between auto-selecting a model or manual selection.
            model_mode = st.radio("Model Selection", ["Auto Select Best Model", "Manual Model Selection"])

            if model_mode == "Manual Model Selection":
                # Dropdown for manual model selection.
                selected_model_name = st.selectbox(
                    "Choose Model",
                    ["Random Forest", "Logistic Regression", "XGBoost"],
                    help="Pick a model manually. Random Forest works well in most cases, XGBoost is powerful for imbalanced data, and Logistic Regression is best for simpler data."
                )
            else:
                # Auto-select model based on data characteristics.
                selected_model_name = auto_select_model(X, y)
                st.info(f"🤖 Auto-selected model: **{selected_model_name}**")

            # Button to trigger model training and evaluation.
            run_button = st.button("▶️ Run Model")
            if run_button:
                # --- Model Training and Evaluation Execution ---
                with st.spinner("⏳ Processing and training model..."):
                    start_time = time.time()
                    # Call the cached training function.
                    model, report, matrix, fpr, tpr, roc_auc, X_full, y_full, X_test, y_test, y_pred = train_model(X, y, selected_model_name)
                    end_time = time.time()
                    time_taken_seconds = end_time - start_time

                    # Format training time for display.
                    hours = int(time_taken_seconds // 3600)
                    minutes = int((time_taken_seconds % 3600) // 60)
                    seconds = int(time_taken_seconds % 60)
                    time_taken_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

                st.success("✅ Model trained successfully!")

                # --- Display Evaluation Metrics ---
                st.subheader("📈 Evaluation Metrics")

                # Extract overall accuracy and class-specific metrics from the classification report.
                accuracy = report['accuracy']
                non_fraud_metrics = report.get('0', {})
                fraud_metrics = report.get('1', {})

                # Create a DataFrame for better display of metrics.
                evaluation_metrics_data = {
                    "Term": ["Precision", "Recall", "F1-Score"],
                    "Non-fraud": [
                        f"{non_fraud_metrics.get('precision', np.nan):.4f}", # Precision for class 0
                        f"{non_fraud_metrics.get('recall', np.nan):.4f}",    # Recall for class 0
                        f"{non_fraud_metrics.get('f1-score', np.nan):.4f}" # F1-Score for class 0
                    ],
                    "Fraud": [
                        f"{fraud_metrics.get('precision', np.nan):.4f}",  # Precision for class 1
                        f"{fraud_metrics.get('recall', np.nan):.4f}",     # Recall for class 1
                        f"{fraud_metrics.get('f1-score', np.nan):.4f}"  # F1-Score for class 1
                    ]
                }

                evaluation_metrics_df = pd.DataFrame(evaluation_metrics_data)
                st.table(evaluation_metrics_df)

                # --- Display Confusion Matrix ---
                st.subheader("🧩 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
                sns.heatmap(matrix, annot=True, fmt='d', cmap="Blues", ax=ax_cm,
                            xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                            yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")

                # Center the plot using columns.
                col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
                with col_cm2:
                    st.pyplot(fig_cm)
                plt.close(fig_cm) # Close the figure to prevent display issues in Streamlit

                # --- Display ROC Curve ---
                st.subheader("📊 ROC Curve")
                fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
                ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], 'k--') # Dashed diagonal line for reference (random classifier)
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"Receiver Operating Characteristic (ROC) Curve - {selected_model_name}")
                ax_roc.legend(loc="lower right")

                # Center the plot using columns.
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.pyplot(fig_roc)
                plt.close(fig_roc) # Close the figure to prevent display issues in Streamlit

                # --- Predict on the full uploaded dataset ---
                with st.spinner("Predicting on all uploaded data..."):
                    # Predict fraud for all rows in the original (unresampled) DataFrame.
                    df["Predicted"] = model.predict(X)

                # Select relevant columns for display (first 3 'V' columns, 'Amount', 'Class', 'Predicted').
                v_cols = [col for col in df.columns if col.startswith("V")][:3]
                display_cols = ["Amount"] + v_cols + ["Class", "Predicted"]

                # Filter for transactions predicted as fraudulent.
                fraud_transactions = df[df["Predicted"] == 1]

                # Display detected fraudulent transactions.
                if not fraud_transactions.empty:
                    st.subheader("🔎 Fraudulent Transactions List")
                    st.dataframe(fraud_transactions[display_cols].reset_index(drop=True))
                else:
                    st.info("No fraudulent transactions detected.")

                # Display the total count of predicted fraudulent transactions.
                fraud_count = df["Predicted"].sum()
                st.markdown(f"### 🚨 Detected {int(fraud_count)} Fraudulent Transactions")

                # --- Summary Section ---
                st.subheader("📝 Summary of Analysis")
                summary_col1, summary_col2 = st.columns(2)

                with summary_col1:
                    st.markdown("### Fraud Detection Summary")
                    st.markdown(f"**Detected Fraudulent Transactions:** {int(fraud_count)}")
                    st.markdown(f"**Time Taken for Training & Prediction:** {time_taken_str}")
                    st.markdown(f"**Selected Model:** `{selected_model_name}`")

                    # Provide explanation for auto-selected model.
                    if model_mode == "Auto Select Best Model":
                        explanation = {
                            "XGBoost": "Selected due to highly imbalanced data or large dataset size.",
                            "Random Forest": "Selected for small-medium dataset size with moderate imbalance.",
                            "Logistic Regression": "Selected for small feature count and simple data pattern."
                        }
                        st.caption(f"ℹ️ *Model auto-selected based on dataset characteristics:* {explanation.get(selected_model_name, '')}")

                with summary_col2:
                    # Display key overall and class-specific metrics.
                    accuracy = report['accuracy'] * 100
                    non_fraud_precision = report['0']['precision'] if '0' in report and 'precision' in report['0'] else np.nan
                    non_fraud_recall = report['0']['recall'] if '0' in report and 'recall' in report['0'] else np.nan
                    non_fraud_f1 = report['0']['f1-score'] if '0' in report and 'f1-score' in report['0'] else np.nan

                    fraud_precision = report['1']['precision'] if '1' in report and 'precision' in report['1'] else np.nan
                    fraud_recall = report['1']['recall'] if '1' in report and 'recall' in report['1'] else np.nan
                    fraud_f1 = report['1']['f1-score'] if '1' in report and 'f1-score' in report['1'] else np.nan

                    summary_data = {
                        "Metric": ["Accuracy", "Non-Fraudulent (0) Precision", "Non-Fraudulent (0) Recall", "Non-Fraudulent (0) F1-Score",
                                   "Fraudulent (1) Precision", "Fraudulent (1) Recall", "Fraudulent (1) F1-Score"],
                        "Value": [f"{accuracy:.2f}%", f"{non_fraud_precision:.2f}", f"{non_fraud_recall:.2f}", f"{non_fraud_f1:.2f}",
                                  f"{fraud_precision:.2f}", f"{fraud_recall:.2f}", f"{fraud_f1:.2f}"]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.table(summary_df)

                    # Display percentage distribution of predicted classes.
                    total_predictions = len(y_pred)
                    predicted_non_fraud_count = np.sum(y_pred == 0)
                    predicted_fraud_count = np.sum(y_pred == 1)

                    percent_non_fraud = (predicted_non_fraud_count / total_predictions) * 100
                    percent_fraud = (predicted_fraud_count / total_predictions) * 100

                    st.markdown(f"**Predicted Non-Fraudulent (0):** {percent_non_fraud:.2f}%")
                    st.markdown(f"**Predicted Fraudulent (1):** {percent_fraud:.2f}%")

                    # --- Download Results Button ---
                    # Allows user to download the original data with a new 'Predicted' column.
                    st.download_button(
                        "⬇️ Download Results as CSV",
                        df.to_csv(index=False),
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        # General error handling for file processing issues.
        st.error(f"❌ Error processing file: {e}")
        st.exception(e) # Displays the full traceback for debugging.
else:
    # Initial message when no file is uploaded.
    st.info("Please upload a CSV file to get started.")
