import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="BizInsight AI", layout="wide")
st.title("📊 BizInsight AI – Business Dashboard")

uploaded_file = st.file_uploader("Upload your business data (CSV only)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.success("File uploaded successfully!")

    # Ensure 'Order Date' is datetime
    if 'Order Date' in df.columns and 'Sales' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df = df.dropna(subset=['Order Date', 'Sales'])

        # Revenue trend
        st.subheader("📈 Revenue Over Time")
        monthly_sales = df.set_index('Order Date').resample('M')['Sales'].sum().reset_index()
        fig1 = px.line(monthly_sales, x='Order Date', y='Sales', title='Monthly Sales Trend')
        st.plotly_chart(fig1, use_container_width=True)

        # Top 10 products
        if 'Product Name' in df.columns:
            st.subheader("🏆 Top 10 Products by Sales")
            top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
            fig2 = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top Products')
            st.plotly_chart(fig2, use_container_width=True)

        # Forecasting section
        st.subheader("🔮 Sales Forecast (Next 6 Months)")
        try:
            prophet_df = monthly_sales.rename(columns={"Order Date": "ds", "Sales": "y"})
            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future)

            forecast_fig = px.line(forecast, x='ds', y='yhat', title='Sales Forecast')
            st.plotly_chart(forecast_fig, use_container_width=True)

            st.write("Forecasted sales for next 6 months:")
            st.dataframe(
                forecast[['ds', 'yhat']].tail(6)
                .rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'})
                .round(2)
            )
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
    else:
        st.error("CSV must contain at least 'Order Date' and 'Sales' columns.")
from textblob import TextBlob

st.header("🗣️ Customer Review Analysis")

review_file = st.file_uploader("Upload customer reviews (CSV with a 'Review' column)", type=["csv"], key="reviews")
if review_file:
    reviews_df = pd.read_csv(review_file)

    if 'Review' not in reviews_df.columns:
        st.error("CSV must contain a 'Review' column.")
    else:
        st.success("Review file uploaded!")

        # Sentiment Analysis
        def get_sentiment(text):
            return TextBlob(str(text)).sentiment.polarity

        reviews_df['Sentiment Score'] = reviews_df['Review'].apply(get_sentiment)
        reviews_df['Sentiment Label'] = reviews_df['Sentiment Score'].apply(
            lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

        sentiment_counts = reviews_df['Sentiment Label'].value_counts()

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(sentiment_counts)

        st.subheader("🔍 Sample Reviews with Sentiment")
        st.dataframe(reviews_df[['Review', 'Sentiment Label', 'Sentiment Score']].head(10))

        st.subheader("📈 Review Stats")
        reviews_df['Length'] = reviews_df['Review'].apply(lambda x: len(str(x).split()))
        st.metric("Average Words per Review", round(reviews_df['Length'].mean(), 2))
