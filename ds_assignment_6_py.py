# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

# Load training dataset
df = pd.read_csv('Titanic_train.csv')

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary column
df.drop('PassengerId', axis=1, inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
if 'Sex' not in categorical_cols:
    categorical_cols = categorical_cols.tolist() + ['Sex']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features/target
y = df_encoded['Survived']
X = df_encoded.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save model and training columns
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(X.columns.tolist(), 'training_columns.pkl')

print("Model and training columns saved successfully!")







import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
model = joblib.load('logistic_regression_model.pkl')
training_columns = joblib.load('training_columns.pkl')

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival using Logistic Regression.")

# Sidebar inputs
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["female", "male"])
age = st.sidebar.slider("Age", 0, 80, 29)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 512.33, 32.2)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
cabin = st.sidebar.text_input("Cabin (Optional)", "B96 B98")

# Prediction button
if st.button("Predict Survival"):
    user_input = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'Cabin': cabin
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables
    categorical_cols = ['Sex', 'Embarked', 'Cabin']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align with training columns
    input_aligned = input_encoded.reindex(columns=training_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_aligned)

    if prediction[0] == 1:
        st.success("✅ The passenger is predicted to SURVIVE!")
    else:
        st.error("❌ The passenger is predicted NOT to survive.")
