import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="BizInsight AI", layout="wide")
st.title("📊 BizInsight AI – Business Dashboard")

uploaded_file = st.file_uploader("Upload your Business Data (CSV only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.success("File uploaded successfully!")

    # Clean up date column
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df.dropna(subset=['Order Date'], inplace=True)

        # Show raw data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # Revenue Over Time
        st.subheader("📈 Revenue Trend Over Time")
        revenue_data = df.groupby('Order Date')['Sales'].sum().reset_index()
        fig1 = px.line(revenue_data, x='Order Date', y='Sales', title='Total Sales Over Time')
        st.plotly_chart(fig1, use_container_width=True)

        # Top 10 Products
        st.subheader("🏆 Top 10 Best-Selling Products")
        top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top Products')
        st.plotly_chart(fig2, use_container_width=True)

        # Sales by Region
        if 'Region' in df.columns:
            st.subheader("🌍 Sales by Region")
            region_sales = df.groupby('Region')['Sales'].sum().reset_index()
            fig3 = px.pie(region_sales, values='Sales', names='Region', title='Sales by Region')
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("The dataset must contain an 'Order Date' column to continue.")
