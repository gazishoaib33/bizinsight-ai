import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import difflib
from newsapi import NewsApiClient
from pytrends.request import TrendReq

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

        # Define column matching function
        def auto_match_column(possible_names, df_cols):
            for name in possible_names:
                match = difflib.get_close_matches(name, df_cols, n=1, cutoff=0.7)
                if match:
                    return match[0]
            return None

        # Clean and convert sales column
        sales_col_raw = auto_match_column(["Sales", "Revenue", "Amount", "Total"], df.columns)
        if sales_col_raw:
            df[sales_col_raw] = pd.to_numeric(
                df[sales_col_raw].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')

        st.success("✅ File uploaded successfully!")

        order_date_col = auto_match_column(["Order Date", "Date", "order_date"], df.columns)
        sales_col = auto_match_column(["Sales", "Revenue", "Amount", "Total"], df.columns)
        product_col = auto_match_column(["Product Name", "Item", "Product"], df.columns)

        if not order_date_col:
            order_date_col = st.selectbox("Select Date Column", df.columns)
        if not sales_col:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                sales_col = st.selectbox("Select Sales Column", numeric_cols)
            else:
                st.error("❌ No numeric columns found for sales.")
                st.stop()
        if not product_col:
            product_col = st.selectbox("Select Product Column", df.columns)

        try:
            df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce', infer_datetime_format=True)
        except Exception as e:
            st.error(f"❌ Failed to parse dates. Error: {e}")
            st.stop()

        if df[sales_col].isna().all():
            st.error("❌ Sales column could not be converted to numbers. Please check your file format.")
            st.stop()

        df = df.dropna(subset=[order_date_col, sales_col])
        if df.empty:
            st.error("⚠️ No valid rows after cleaning. Please check your data.")
            st.stop()

        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))

        st.subheader("📈 Revenue Over Time")
        monthly_sales = df.set_index(order_date_col).resample('M')[sales_col].sum().reset_index()
        fig1 = px.line(monthly_sales, x=order_date_col, y=sales_col, title="Monthly Sales Trend")
        st.plotly_chart(fig1, use_container_width=True)

        if product_col in df.columns:
            st.subheader("🏆 Top 10 Products by Sales")
            top_products = df.groupby(product_col)[sales_col].sum().nlargest(10).reset_index()
            fig2 = px.bar(top_products, x=sales_col, y=product_col, orientation='h', title="Top Products")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🔮 Sales Forecast (Next 6 Months)")
        try:
            prophet_df = monthly_sales.rename(columns={order_date_col: "ds", sales_col: "y"}).dropna()
            if prophet_df.shape[0] < 2:
                st.warning("⚠️ Not enough data to build a forecast.")
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

        st.subheader("📋 Business Summary")
        latest_month = monthly_sales[order_date_col].max()
        latest_month_str = latest_month.strftime('%B %Y') if pd.notnull(latest_month) else "Unknown"
        last_sales = monthly_sales[sales_col].iloc[-1]
        prev_sales = monthly_sales[sales_col].iloc[-2] if len(monthly_sales) > 1 else None

        if prev_sales:
            diff = last_sales - prev_sales
            perc = (diff / prev_sales) * 100 if prev_sales != 0 else 0
            trend = "up 📈" if diff > 0 else "down 📉"
            st.markdown(f"**Last month’s sales ({latest_month_str})** were **${last_sales:,.2f}**, {trend} by **{abs(perc):.1f}%** from previous month.")

        # =============================
        # 📈 Product Growth Advisor
        # =============================
        st.header("📈 Product Growth Advisor")
        top_n = 3
        top_products = df.groupby(product_col)[sales_col].sum().nlargest(top_n).reset_index()

        pytrends = TrendReq(hl='en-US', tz=360)
        newsapi = NewsApiClient(api_key="8b3e20485b4943d190d06f67eee4df6b")  # Replace with your API key

        for i, row in top_products.iterrows():
            product = row[product_col]
            product_sales = row[sales_col]

            st.subheader(f"🧪 Product: **{product}**")
            st.markdown(f"- 💰 **Current Sales:** ${product_sales:,.2f}")

            try:
                pytrends.build_payload([product], cat=0, timeframe='today 3-m', geo='', gprop='')
                trend_df = pytrends.interest_over_time()
                trend_score = trend_df[product].mean() if not trend_df.empty else 0
                trend_change = trend_df[product].pct_change().mean() * 100 if not trend_df.empty else 0

                st.markdown(f"- 🌍 **Google Trend:** Avg Score: {trend_score:.1f}, Change: {trend_change:+.1f}%")

                related = pytrends.related_queries()
                related_top = related.get(product, {}).get('top')
                if related_top is not None and not related_top.empty:
                    top_keywords = ', '.join(related_top['query'].head(3))
                    st.markdown(f"- 🔁 **Related Searches:** {top_keywords}")
                else:
                    st.markdown("- 🔁 **Related Searches:** Not found")

            except Exception as e:
                st.warning(f"Google Trends failed: {e}")

            try:
                news = newsapi.get_everything(q=product, language='en', sort_by='relevancy', page_size=5)
                if news['articles']:
                    headlines = [article['title'] for article in news['articles']]
                    sentiments = [TextBlob(title).sentiment.polarity for title in headlines]
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_label = 'Positive ✅' if avg_sentiment > 0 else 'Negative ❌' if avg_sentiment < 0 else 'Neutral ⚪'
                    st.markdown(f"- 📰 **News Sentiment:** {sentiment_label} ({avg_sentiment:.2f})")
                else:
                    avg_sentiment = 0
                    st.markdown("- 📰 **News Sentiment:** No recent news found")

            except Exception as e:
                avg_sentiment = 0
                st.warning(f"News API failed: {e}")

            if trend_change > 10 and avg_sentiment > 0:
                suggestion = f"📢 Consider launching a **promotion or ad campaign** — interest is growing and sentiment is positive!"
            elif trend_change < -10:
                suggestion = f"📉 Trend is declining. Consider repositioning or bundling this product with a high-performer."
            elif avg_sentiment < 0:
                suggestion = f"⚠️ Reviews/news are negative. Investigate customer concerns and improve perception."
            else:
                suggestion = f"📊 Stable trend — explore content marketing using keywords like **{top_keywords if 'top_keywords' in locals() else product}**."

            st.success(f"🧠 Suggestion: {suggestion}")
            st.markdown("---")

    except Exception as e:
        st.error(f"Failed to read business file: {e}")
