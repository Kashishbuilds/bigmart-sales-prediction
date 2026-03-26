import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Big Mart AI", layout="wide")

# ------------------ CLEAN UI ------------------
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #000000;
}
h1, h2, h3 {
    color: #2E7D32;
}
.stButton>button {
    background-color: #2E7D32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model_path = os.path.join(os.path.dirname(__file__), 'bigmart_model.sav')
model = pickle.load(open(model_path, 'rb'))

# ------------------ SIDEBAR ------------------
menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "🔮 Predict", "📊 Insights"])

# ------------------ HOME ------------------
if menu == "🏠 Home":
    st.title("🛒 Big Mart Sales Prediction System")

    st.markdown("""
    ### 💼 About
    Predict retail sales using Machine Learning (XGBoost).

    ### 🚀 Features
    - 🔮 Single Prediction + Chart  
    - 📊 Insights Dashboard  
    """)

# ------------------ PREDICT ------------------
elif menu == "🔮 Predict":

    st.title("🔮 Predict Sales")

    col1, col2 = st.columns(2)

    with col1:
        Item_Identifier = st.number_input("Item Identifier", min_value=0)
        Item_Weight = st.number_input("Item Weight", min_value=0.0)
        Item_Fat_Content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
        Item_Visibility = st.number_input("Item Visibility", min_value=0.0)

    with col2:
        Item_Type = st.number_input("Item Type", min_value=0)
        Item_MRP = st.number_input("Item MRP", min_value=0.0)
        Outlet_Identifier = st.number_input("Outlet Identifier", min_value=0)
        Outlet_Establishment_Year = st.number_input("Year", min_value=1985, max_value=2025)

    Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    Outlet_Location_Type = st.selectbox("Location", ["Tier 1", "Tier 2", "Tier 3"])
    Outlet_Type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

    # Encoding
    fat_map = {"Low Fat": 0, "Regular": 1}
    size_map = {"Small": 0, "Medium": 1, "High": 2}
    location_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    type_map = {"Grocery Store": 0, "Supermarket Type1": 1, "Supermarket Type2": 2, "Supermarket Type3": 3}

    if st.button("Predict"):

        if Item_Weight <= 0:
            st.warning("⚠️ Weight must be positive")
        else:
            input_data = np.array([[Item_Identifier, Item_Weight, fat_map[Item_Fat_Content],
                                    Item_Visibility, Item_Type, Item_MRP,
                                    Outlet_Identifier, Outlet_Establishment_Year,
                                    size_map[Outlet_Size], location_map[Outlet_Location_Type],
                                    type_map[Outlet_Type]]])

            with st.spinner("Predicting..."):
                prediction = model.predict(input_data)[0]

            st.success(f"💰 Predicted Sales: ₹ {prediction:,.2f}")

            # Chart
            st.subheader("📊 Sales Visualization")
            chart_df = pd.DataFrame({"Predicted Sales": [prediction]})
            st.bar_chart(chart_df)

            # Insight
            if prediction > 5000:
                st.info("📈 High sales expected")
            else:
                st.info("📉 Moderate sales expected")

# ------------------ INSIGHTS ------------------
elif menu == "📊 Insights":

    st.title("📊 Insights Dashboard")

    file = st.file_uploader("Upload Dataset (optional)", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
    else:
        try:
            data = pd.read_csv("big_mart_data.csv")
        except:
            st.warning("⚠️ Upload dataset to view insights")
            st.stop()

    st.subheader("📌 Data Preview")
    st.write(data.head())

    # KPIs
    st.subheader("📊 Key Metrics")
    c1, c2, c3 = st.columns(3)

    c1.metric("Avg Sales", f"₹ {data['Item_Outlet_Sales'].mean():.2f}")
    c2.metric("Max Sales", f"₹ {data['Item_Outlet_Sales'].max():.2f}")
    c3.metric("Min Sales", f"₹ {data['Item_Outlet_Sales'].min():.2f}")

    # Charts
    st.subheader("📈 Sales Distribution")
    st.bar_chart(data['Item_Outlet_Sales'])

    st.subheader("🏪 Outlet Type Distribution")
    st.bar_chart(data['Outlet_Type'].value_counts())

    st.subheader("💰 MRP vs Sales")
    st.line_chart(data[['Item_MRP', 'Item_Outlet_Sales']])

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>🚀 Big Mart ML Project | Streamlit</p>", unsafe_allow_html=True)