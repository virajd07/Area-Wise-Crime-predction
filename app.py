import streamlit as st
import pandas as pd
import pickle
import joblib
from utils import crime_trend_plot
from news_scraper import fetch_crime_articles

# ----------------- Page Setup -----------------
st.set_page_config(page_title="🔍 AI Crime Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🔍 AI-Powered Area-Wise Crime Predictor</h1>
    <p style='text-align: center; font-size:18px;'>Analyze trends, predict crimes, and classify live news using NLP</p>
    <hr style='border: 2px solid #f63366;'>
""", unsafe_allow_html=True)

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    model = pickle.load(open("crime_model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    clf = joblib.load("crime_news_model.pkl")
    vec = joblib.load("crime_vectorizer.pkl")
    return model, le, clf, vec

# ----------------- Load Data -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_crime_data.csv")
    if "Major_Crime" not in df.columns:
        df["Major_Crime"] = df.iloc[:, 2:].idxmax(axis=1)
    return df

def classify_news(news_df, clf, vec):
    labels = ["Murder", "Robbery", "Kidnapping", "Assault", "Cyber Crime"]
    if "text" not in news_df.columns or news_df.empty:
        return pd.DataFrame()
    preds = clf.predict(vec.transform(news_df["text"]))
    news_df["Predicted_Crime"] = [labels[p] for p in preds]
    return news_df

# ----------------- Load Resources -----------------
model, le, clf, vec = load_models()
df = load_data()

# ----------------- Sidebar Filters -----------------
st.sidebar.header("⚙️ Filter Options")
state = st.sidebar.selectbox("📍 Select State", df["STATE/UT"].unique())
year = st.sidebar.selectbox("📅 Select Year", sorted(df["Year"].unique()))

# ----------------- Crime Prediction -----------------
st.header("📌 Predicted Major Crime")

row = df[(df["STATE/UT"] == state) & (df["Year"] == year)]
if not row.empty:
    features = row.iloc[0, 2:-1].values.reshape(1, -1)
    pred = le.inverse_transform(model.predict(features))

    st.markdown(f"""
    <div style="background-color:#dff0d8; padding:15px; border-radius:10px; font-size:18px;">
        🔐 <b>Predicted Major Crime in {state} ({year}):</b>
        <span style="color: #f63366;"><b>{pred[0]}</b></span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No data found for selected combination.")

# ----------------- Crime Trend Plot -----------------
st.header("📈 Crime Trend Over Years")
st.plotly_chart(crime_trend_plot(df, state), use_container_width=True)

# ----------------- Real-Time News Classification -----------------
st.header("📰 Real-Time Crime News Classification")
news_df = fetch_crime_articles()

if news_df.empty or "text" not in news_df.columns:
    st.error("⚠️ Could not fetch real-time news articles.")
else:
    classified_df = classify_news(news_df, clf, vec)
    st.success(f"✅ Classified {len(classified_df)} articles")

    st.dataframe(
        classified_df[["title", "Predicted_Crime", "link"]].rename(columns={
            "title": "📰 Headline",
            "Predicted_Crime": "🔎 Crime Type",
            "link": "🔗 Source"
        }),
        use_container_width=True
    )

# ----------------- Crime Count Lookup Section -----------------
st.header("🔍 Check Crime Count by State, Year and Crime Type")

selected_state = st.selectbox("📍 Choose State", df["STATE/UT"].unique(), key="crime_input_state")
selected_year = st.selectbox("📅 Choose Year", sorted(df["Year"].unique()), key="crime_input_year")
selected_crime = st.selectbox("🔎 Choose Crime Type", 
                              [col for col in df.columns if col not in ["STATE/UT", "Year", "Major_Crime"]],
                              key="crime_input_type")

crime_row = df[(df["STATE/UT"] == selected_state) & (df["Year"] == selected_year)]

if not crime_row.empty:
    count = int(crime_row[selected_crime].values[0])
    st.markdown(f"""
    <div style="background-color:#000000; padding:15px; border-radius:10px; font-size:18px;">
        🔎 <b>Reported '{selected_crime}' cases in {selected_state} ({selected_year}):</b>
        <span style="color: white;"><b>{count}</b></span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No data available for this combination.")

# ----------------- Footer -----------------
st.markdown("""
<hr>
<p style='text-align: center; color: grey; font-size: 14px;'>
🔐 Developed by <b>Viraj Deore</b> | AI + NLP Project | <a href="https://github.com/yourgithub" target="_blank">GitHub</a>
</p>
""", unsafe_allow_html=True)
