import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="AI Helpdesk Classifier", page_icon="🛠️", layout="centered")

# Small, transparent demo dataset. Expand and evaluate before production use.
DATA = [
    ("wifi is not working", "Network"),
    ("internet connection is slow", "Network"),
    ("network keeps disconnecting", "Network"),
    ("cannot connect to office network", "Network"),
    ("router connection has failed", "Network"),
    ("network access is unavailable", "Network"),
    ("VPN is not connecting", "Network"),
    ("website cannot be reached because of network", "Network"),
    ("office internet is down", "Network"),
    ("wireless connection keeps dropping", "Network"),
    ("laptop screen is broken", "Hardware"),
    ("keyboard is not working", "Hardware"),
    ("computer will not start", "Hardware"),
    ("battery is not charging", "Hardware"),
    ("mouse is not responding", "Hardware"),
    ("monitor display is damaged", "Hardware"),
    ("printer has a paper jam", "Hardware"),
    ("laptop is overheating", "Hardware"),
    ("headphones are not detected", "Hardware"),
    ("computer fan is making noise", "Hardware"),
    ("application keeps crashing", "Software"),
    ("software installation failed", "Software"),
    ("login error in the application", "Software"),
    ("program is not opening", "Software"),
    ("software update failed", "Software"),
    ("application shows an error", "Software"),
    ("cannot open the spreadsheet program", "Software"),
    ("password reset page has an error", "Software"),
    ("email app is freezing", "Software"),
    ("document editor is not responding", "Software"),
    ("wifi connection has stopped", "Network"),
    ("ethernet cable connection is unavailable", "Network"),
    ("computer screen is flickering", "Hardware"),
    ("laptop charger is damaged", "Hardware"),
    ("app crashes after the latest update", "Software"),
    ("cannot install the required application", "Software"),
    ("network speed is very poor", "Network"),
    ("keyboard keys are damaged", "Hardware"),
    ("application login is failing", "Software"),
    ("monitor is not turning on", "Hardware"),
    ("cannot access shared network drive", "Network"),
    ("USB port is not detecting devices", "Hardware"),
    ("browser keeps showing an application error", "Software"),
    ("internet access is unavailable", "Network"),
    ("laptop hinge is damaged", "Hardware"),
    ("software license activation failed", "Software"),
    ("office network is very slow", "Network"),
    ("touchpad is not responding", "Hardware"),
    ("program freezes when I open it", "Software"),
    ("email software will not launch", "Software"),
]

@st.cache_resource
def train_model():
    df = pd.DataFrame(DATA, columns=["ticket", "category"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(df["ticket"], df["category"])
    return model, df

model, training_data = train_model()

st.title("🛠️ AI-Powered IT Helpdesk Ticket Classifier")
st.write("A small NLP + machine-learning prototype that suggests a support-ticket category.")
st.info("Demo limitation: the model uses a small hand-written dataset. Predictions may be wrong, especially for unfamiliar wording.")

with st.expander("How the model works"):
    st.markdown("""
    1. **Input:** a user describes an IT issue.
    2. **TF-IDF:** turns words and short word combinations into numeric features.
    3. **Logistic Regression:** learns associations between those features and ticket categories.
    4. **Output:** the predicted category and model confidence.

    This is a text-classification prototype, not a generative-AI chatbot.
    """)

ticket = st.text_area(
    "Describe the IT problem",
    placeholder="Example: My office Wi-Fi keeps disconnecting",
    height=100
)

if st.button("Classify ticket", type="primary", use_container_width=True):
    if not ticket.strip():
        st.warning("Please enter a complaint first.")
    else:
        predicted = model.predict([ticket])[0]
        probabilities = model.predict_proba([ticket])[0]
        classes = model.named_steps["classifier"].classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)

        st.subheader(f"Suggested category: {predicted}")
        st.metric("Model confidence (not a guarantee)", f"{max(probabilities):.1%}")
        st.caption("Confidence is the model's estimated probability, not proof that the answer is correct.")
        st.write("Category probability estimates:")
        st.bar_chart(pd.DataFrame({"Estimated probability": [p for _, p in ranked]}, index=[c for c, _ in ranked]))

st.divider()
st.subheader("Model information")
st.write(f"Demonstration examples: **{len(training_data)}**")
st.write(f"Categories: **{', '.join(sorted(training_data['category'].unique()))}**")
st.caption("Do not use the current model for real ticket routing without more data, a held-out evaluation set, and human review.")
