import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Decision Tree Regressor", layout="centered")

st.title("🌳 Decision Tree Regressor for Housing Dataset")

# ===============================
# STEP 1: Load Dataset
# ===============================
st.subheader("Step 1: Loading Dataset")
df = pd.read_csv("housing.csv")
st.write("Dataset loaded successfully")
st.dataframe(df.head())

# ===============================
# STEP 2: Feature & Target Split
# ===============================
st.subheader("Step 2: Splitting Features & Target")

X = df.drop(["median_house_value", "ocean_proximity"], axis=1)
y = df["median_house_value"]

st.write("Features used:")
st.write(list(X.columns))

# ===============================
# STEP 3: Handle Missing Values
# ===============================
st.subheader("Step 3: Handling Missing Values")
X["total_bedrooms"].fillna(X["total_bedrooms"].median(), inplace=True)
st.write("Missing values filled using median")

# ===============================
# STEP 4: Train-Test Split
# ===============================
st.subheader("Step 4: Train-Test Split")
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
st.write(f"Training samples: {x_train.shape[0]}")
st.write(f"Testing samples: {x_test.shape[0]}")

# ===============================
# Sidebar: Model Settings
# ===============================
st.sidebar.header("⚙️ Model Settings")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

# ===============================
# STEP 5: Train Model
# ===============================
st.subheader("Step 5: Training Decision Tree Regressor")

model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
model.fit(x_train, y_train)
st.success("Model trained successfully")

# ===============================
# Sidebar: User Inputs
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
# STEP 6: Prediction
# ===============================
st.subheader("Step 6: Prediction")

prediction = model.predict(input_data)[0]
st.success(f"🏠 Predicted House Value: ₹ {prediction:,.2f}")

# ===============================
# STEP 7: Model Performance
# ===============================
st.markdown("## 📊 Model Performance")

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="R² Score",
        value=f"{r2:.3f}"
    )

with col2:
    st.metric(
        label="Mean Squared Error",
        value=f"{mse:,.0f}"
    )

# ===============================
# Dataset Viewer
# ===============================
with st.expander("📂 View Full Dataset"):
    st.dataframe(df)