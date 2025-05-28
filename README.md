# Income Predictation

Predict whether an individual's income exceeds $50K/year based on census data using a machine learning model.

🎯 Objective
This project builds a binary classification model to predict income category (>50K or <=50K) using the Adult Census Income dataset. It includes:

Exploratory Data Analysis (EDA)

Data preprocessing

Model training and evaluation

Deployment with Flask and Streamlit

📦 Dataset Information
Dataset: Adult Census Income

Source: OpenML

Target: income

Task: Binary Classification

python
Copy
Edit
from sklearn.datasets import fetch_openml
data = fetch_openml("adult", version=2, as_frame=True)
🧪 Methodology
1. Exploratory Data Analysis (EDA)
Explored distributions of categorical and numerical features

Handled missing values and outliers

Visualized key relationships with the target variable

2. Data Preprocessing
Label encoding for binary categorical features

One-hot encoding for multi-class categorical features

Normalization of numerical columns

Split data into training and test sets

3. Model Building
Tried multiple models: Logistic Regression, Random Forest, etc.

Selected the best-performing model based on:

Accuracy

Precision

Recall

F1-score

4. Model Saving
Serialized the trained model using joblib and saved as model.pkl

🚀 Deployment
✅ Flask Web App
flask_app.py creates a form for user input

Predicts income category based on form data

HTML form rendered via Jinja2 (templates/index.html)

![image](https://github.com/user-attachments/assets/21439b39-4ce1-4ffe-a523-1578feb415f0)


✅ Streamlit App
streamlit_app.py offers a user-friendly UI with widgets

Displays prediction and model confidence interactively

![image](https://github.com/user-attachments/assets/ba473d58-0883-4d72-9b27-c62c75a0f594)


📂 Folder Structure
cpp

Copy
Edit

income-predictor/

├── flask_app.py

├── streamlit_app.py

├── model/

│   └── model.pkl

├── templates/

│   └── index.html

├── static/

├── notebook/


│   └── eda_model.ipynb

├── requirements.txt

└── README.md
