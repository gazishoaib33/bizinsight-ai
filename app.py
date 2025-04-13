import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import difflib

st.set_page_config(page_title="BizInsight AI", layout="wide")
st.title("📊 BizInsight AI – Business Intelligence Dashboard")

# =============================
# BUSINESS DATA UPLOAD & ANALYSIS
# =============================
st.header("📁 Upload Business Data")

uploaded_file = st.file_uploader("Upload your business data (CSV only)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success("✅ File uploaded successfully!")

        def auto_match_column(possible_names, df_cols):
            for name in possible_names:
                match = difflib.get_close_matches(name, df_cols, n=1, cutoff=0.7)
                if match:
                    return match[0]
            return None

        order_date_col = auto_match_column(["Order Date", "Date", "order_date"], df.columns)
        sales_col = auto_match_column(["Sales", "Revenue", "Amount", "Total"], df.columns)
        product_col = auto_match_column(["Product Name", "Item", "Product"], df.columns)

        st.subheader("🧠 Auto Column Mapping")
        order_date_col = st.selectbox("Date Column", df.columns, index=df.columns.get_loc(order_date_col) if order_date_col else 0)
        sales_col = st.selectbox("Sales Column", df.columns, index=df.columns.get_loc(sales_col) if sales_col else 0)
        product_col = st.selectbox("Product Column", df.columns, index=df.columns.get_loc(product_col) if product_col else 0)

        df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
        df = df.dropna(subset=[order_date_col, sales_col])

        # 📋 Data Preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))

        # 📈 Revenue Over Time
        st.subheader("📈 Revenue Over Time")
        monthly_sales = df.set_index(order_date_col).resample('M')[sales_col].sum().reset_index()
        fig1 = px.line(monthly_sales, x=order_date_col, y=sales_col, title="Monthly Sales Trend")
        st.plotly_chart(fig1, use_container_width=True)

        # 🏆 Top Products
        if product_col in df.columns:
            st.subheader("🏆 Top 10 Products by Sales")
            top_products = df.groupby(product_col)[sales_col].sum().nlargest(10).reset_index()
            fig2 = px.bar(top_products, x=sales_col, y=product_col, orientation='h', title="Top Products")
            st.plotly_chart(fig2, use_container_width=True)

        # 🔮 Sales Forecast
        st.subheader("🔮 Sales Forecast (Next 6 Months)")
        try:
            prophet_df = monthly_sales.rename(columns={order_date_col: "ds", sales_col: "y"}).dropna()
            if prophet_df.shape[0] < 2:
                st.error("📉 Not enough data to forecast.")
            else:
                model = Prophet()
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=6, freq='M')
                forecast = model.predict(future)

                fig3 = px.line(forecast, x='ds', y='yhat', title="Forecasted Sales")
                st.plotly_chart(fig3, use_container_width=True)

                st.write("📅 Forecast for next 6 months:")
                st.dataframe(
                    forecast[['ds', 'yhat']].tail(6)
                    .rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'})
                    .round(2)
                )
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")

        # 📋 Business Summary
        st.subheader("📋 Business Summary")
        if not monthly_sales[order_date_col].isnull().all():
            latest_month = monthly_sales[order_date_col].max()
            latest_month_str = latest_month.strftime('%B %Y') if pd.notnull(latest_month) else "Unknown"
        else:
            latest_month_str = "Unknown"

        last_sales = monthly_sales[sales_col].iloc[-1]
        prev_sales = monthly_sales[sales_col].iloc[-2] if len(monthly_sales) > 1 else None

        if prev_sales:
            diff = last_sales - prev_sales
            perc = (diff / prev_sales) * 100 if prev_sales != 0 else 0
            trend = "up 📈" if diff > 0 else "down 📉"
            st.markdown(f"**Last month’s sales ({latest_month_str})** were **${last_sales:,.2f}**, {trend} by **{abs(perc):.1f}%** from previous month.")

    except Exception as e:
        st.error(f"Failed to read business file: {e}")

# =============================
# CUSTOMER REVIEW ANALYSIS
# =============================
st.header("🗣️ Customer Review Analysis")

review_file = st.file_uploader("Upload customer reviews (CSV with 'Review' column)", type=["csv"], key="reviews")
if review_file:
    try:
        reviews_df = pd.read_csv(review_file)

        possible_review_cols = ['Review', 'review', 'reviews', 'review_body', 'ReviewText']
        for col in reviews_df.columns:
            if col.strip() in possible_review_cols:
                reviews_df.rename(columns={col: 'Review'}, inplace=True)
                break

        if 'Review' not in reviews_df.columns:
            st.error("CSV must contain a 'Review' column.")
        else:
            st.success("Review file uploaded!")

            def get_sentiment(text):
                return TextBlob(str(text)).sentiment.polarity

            reviews_df['Sentiment Score'] = reviews_df['Review'].apply(get_sentiment)
            reviews_df['Sentiment Label'] = reviews_df['Sentiment Score'].apply(
                lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

            st.subheader("📊 Sentiment Distribution")
            st.bar_chart(reviews_df['Sentiment Label'].value_counts())

            st.subheader("🔍 Sample Reviews")
            st.dataframe(reviews_df[['Review', 'Sentiment Label', 'Sentiment Score']].head(10))

            reviews_df['Length'] = reviews_df['Review'].apply(lambda x: len(str(x).split()))
            st.metric("Average Words per Review", round(reviews_df['Length'].mean(), 2))

            st.subheader("☁️ Word Cloud")
            text = ' '.join(str(r) for r in reviews_df['Review'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Review file error: {e}")

# =============================
# GOOGLE TRENDS INTEGRATION
# =============================
st.header("🌐 Google Trends Insights")

pytrends = TrendReq(hl='en-US', tz=360)
keyword = st.text_input("🔎 Enter a keyword to analyze on Google Trends", value="AI")

if keyword:
    try:
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
        trends_data = pytrends.interest_over_time()

        if not trends_data.empty:
            st.subheader(f"📈 Interest Over Time for '{keyword}'")
            st.line_chart(trends_data[keyword])

            st.subheader("🔍 Related Queries")
            related = pytrends.related_queries()
            top_queries = related.get(keyword, {}).get('top')

            if top_queries is not None and not top_queries.empty:
                st.dataframe(top_queries)
            else:
                st.info("No related queries found.")
        else:
            st.warning("No data returned. Try another keyword.")
    except Exception as e:
        st.error(f"Failed to fetch Google Trends: {e}")
# =============================
# 🤖 GPT AI Assistant (OpenAI v1.0+)
# =============================

from openai import OpenAI

st.header("🤖 GPT AI Assistant")

# Load your API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # or use: OpenAI(api_key="sk-...")

user_input = st.chat_input("Ask me anything about your uploaded business data...")

if user_input and uploaded_file:
    with st.spinner("🤖 Thinking..."):

        # Build context from your uploaded data
        try:
            total_sales = df[sales_col].sum()
            top_product = df.groupby(product_col)[sales_col].sum().idxmax()
            total_orders = len(df)
            date_min = df[order_date_col].min().date()
            date_max = df[order_date_col].max().date()

            context = f"""
            You are a helpful AI assistant for business analysis.
            You are helping the user explore their uploaded sales data.

            - Total Sales: ${total_sales:,.2f}
            - Top Product: {top_product}
            - Total Orders: {total_orders}
            - Date Range: {date_min} to {date_max}

            Answer questions using the uploaded dataset. If unsure, say so.
            """

            # Call OpenAI Chat Completion (v1.0+)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input}
                ]
            )

            reply = response.choices[0].message.content
            st.chat_message("assistant").markdown(reply)

        except Exception as e:
            st.error(f"💥 GPT Assistant Error: {e}")
