# 💳 Fraudulent Transaction Detector

&lt;p align="center">
&lt;img src="https://img.shields.io/badge/FRAUDULENT--TRANSACTIONAL--DATA--DETECTOR--APPLICATION-404040?style=for-the-badge&amp;logo=&amp;logoColor=white&amp;label=title" alt="FRAUDULENT-TRANSACTIONAL-DATA-DETECTOR-APPLICATION" width="100%"/>
&lt;/p>

&lt;h3 align="center">&lt;i>Detect fraud swiftly, secure transactions confidently.&lt;/i>&lt;/h3>

&lt;p align="center">
&lt;img src="https://img.shields.io/badge/last%20commit-today-grey?style=for-the-badge" alt="Last Commit">
&lt;img src="https://img.shields.io/badge/python-100.0%25-blue?style=for-the-badge&amp;logo=python" alt="Python Percentage">
&lt;img src="https://img.shields.io/badge/languages-1-blue?style=for-the-badge" alt="Languages">
&lt;/p>

&lt;h3 align="center">Built with the tools and technologies:&lt;/h3>

&lt;p align="center">
&lt;img src="https://img.shields.io/badge/Markdown-000000?style=for-the-badge&amp;logo=markdown&amp;logoColor=white" alt="Markdown">
&lt;img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&amp;logo=streamlit&amp;logoColor=white" alt="Streamlit">
&lt;img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&amp;logo=scikit-learn&amp;logoColor=white" alt="scikit-learn">
&lt;img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&amp;logo=numpy&amp;logoColor=white" alt="NumPy">
&lt;img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python">
&lt;img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&amp;logo=pandas&amp;logoColor=white" alt="pandas">
&lt;/p>

A Streamlit-powered web app that automatically detects fraudulent transactions using machine learning models such as XGBoost, Random Forest, and Logistic Regression. It supports both auto and manual model selection modes.

## 🚀 Features

- Upload your own CSV transaction file
- Auto or manual model selection
- Handles imbalanced data using SMOTEENN
- Performance metrics (Precision, Recall, F1-Score)
- Confusion Matrix & ROC Curve
- Download fraud predictions as CSV
- Smart model recommendation based on data characteristics

## 📂 Dataset Format

The uploaded CSV file **must** contain:
- A `Class` column (`0 = legit`, `1 = fraud`)
- A numerical `Amount` column
- Optional `Time` column (it will be removed)

- 

## 🧠 Models Used

- **XGBoost**: For large/imbalanced datasets
- **Random Forest**: Balanced accuracy, handles noise
- **Logistic Regression**: For simpler data with few features

## 📊 Metrics Visualized

- Classification Report
- Confusion Matrix
- ROC Curve (AUC Score)
- Fraudulent transaction preview

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Backend/ML**: Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization**: Seaborn, Matplotlib
- **Resampling**: SMOTEENN

Here is a preview of the Fraudulent Transaction Detector app in action:

### Data Upload Screen  
![Data Upload](images/upload_screen.png)

### Model Training & Results  
![Model Results](images/model_results.png)


## ▶️ How to Run


```bash
# 1. Clone this repo
git clone https://github.com/Dharnu04/fraud-detector-app.git
cd fraud-detector-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
