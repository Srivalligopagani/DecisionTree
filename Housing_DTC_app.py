import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Decision Tree Classifier", layout="centered")

st.title("🌳 Decision Tree Classifier for Housing Dataset")

# ===============================
# STEP 1: Load Dataset
# ===============================
st.subheader("Step 1: Load Dataset")
df = pd.read_csv("housing.csv")
st.success("Dataset loaded successfully")
st.dataframe(df.head())

# ===============================
# STEP 2: Create Classes
# ===============================
st.subheader("Step 2: Create Price Categories")

df["price_category"] = pd.qcut(
    df["median_house_value"],
    q=3,
    labels=["Low", "Medium", "High"]
)

st.write(df["price_category"].value_counts())

# ===============================
# STEP 3: Feature & Target Split
# ===============================
st.subheader("Step 3: Feature & Target Selection")

X = df.drop(
    ["median_house_value", "price_category", "ocean_proximity"],
    axis=1
)
y = df["price_category"]

st.write("Selected Features:")
st.write(list(X.columns))

# ===============================
# STEP 4: Handle Missing Values
# ===============================
st.subheader("Step 4: Handle Missing Values")
X["total_bedrooms"].fillna(X["total_bedrooms"].median(), inplace=True)
st.write("Missing values filled using median")

# ===============================
# Sidebar: Model Settings
# ===============================
st.sidebar.header("⚙️ Model Settings")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 4)

# ===============================
# STEP 5: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===============================
# STEP 6: Train Model
# ===============================
st.subheader("Step 6: Train Decision Tree Classifier")

model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
st.success("Model trained successfully")

# ===============================
# Sidebar: User Input
# ===============================
st.sidebar.header("📌 Enter House Details")

user_input = []
for col in X.columns:
    value = st.sidebar.number_input(
        f"{col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    user_input.append(value)

input_data = np.array([user_input])

# ===============================
# STEP 7: Prediction
# ===============================
st.subheader("Step 7: Prediction")

prediction = model.predict(input_data)[0]
st.success(f"🏠 Predicted Price Category: **{prediction}**")

# ===============================
# STEP 8: Model Performance (BIG & CLEAR)
# ===============================
st.markdown("## 📊 Model Performance")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Total Test Samples", len(y_test))

# ===============================
# STEP 9: Detailed Report
# ===============================
with st.expander("📄 Classification Report"):
    st.text(classification_report(y_test, y_pred))

with st.expander("🧮 Confusion Matrix"):
    st.write(confusion_matrix(y_test, y_pred))

# ===============================
# Dataset Viewer
# ===============================
with st.expander("📂 View Full Dataset"):
    st.dataframe(df)