# BizInsight AI – A Smart Business Dashboard 📊💡

**BizInsight AI** is a project I created to help businesses make smarter decisions by combining data analysis, forecasting, customer feedback, and real-time market signals — all in one place.

Whether you're running a small business or just exploring your sales data, this app lets you:

---

## 🔍 What It Can Do

### 🧾 Upload Your Sales Data (CSV)
Just drag and drop your file — the app figures out columns like dates, sales, and products automatically.

### 📈 See Sales Trends
- Monthly revenue trend (visualized)
- Top 10 best-selling products
- Summary of how your sales changed over time

### 🔮 Get Future Predictions
- 6-month sales forecast using Facebook Prophet
- Easily see what’s coming up next in your business

### 💬 Understand Customer Sentiment
- Upload reviews → get instant sentiment scores
- Breakdown: Positive, Neutral, Negative
- Word cloud to see what customers talk about most

### 🌍 Watch the Market in Real-Time
- Track Google Trends for your products
- See what related terms people are searching for
- News sentiment analysis from real headlines

### 🧠 Product Growth Advisor
This is the fun part. For your top products, BizInsight:
- Checks their trend score
- Analyzes live news
- And gives smart suggestions like:
  > "Run a promo — trends are rising and news is positive!"

---

## 💻 Built With
- Python & Streamlit
- Prophet for forecasting
- PyTrends for trend tracking
- NewsAPI for real-time headlines
- TextBlob for sentiment analysis

---

## 🚀 How to Run It

1. **Install required packages:**
```bash
pip install -r requirements.txt
```

2. **Add your NewsAPI key to secrets:**
```
📁 .streamlit/secrets.toml
```
```toml
NEWSAPI_KEY = "your_api_key_here"
```

3. **Run the app:**
```bash
streamlit run app.py
```

---

## 🙋‍♂️ Why I Built This
I wanted to build something that brings data science and real-world business together. Something that’s useful, visual, and feels alive with data.

If you're into data, BI, or just love smart dashboards — this one’s for you.

---

## 👋 Let's Connect
Built by **Gazi Shoaib**

- GitHub: [@gazishoaib33](https://github.com/gazishoaib33)
- LinkedIn: [Gazi Shoaib](https://www.linkedin.com/in/gazi-shoaib-1291531a4/)

Feel free to fork, use, or suggest ideas!
