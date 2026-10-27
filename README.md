# AI-Powered IT Helpdesk Ticket Classifier

A beginner-friendly NLP and machine-learning project that predicts whether an IT support ticket belongs to **Network**, **Hardware**, or **Software**.

## Why this project?
Support teams receive many requests. Automatically suggesting a category can help a team triage tickets faster. This prototype demonstrates the basic workflow; it does not automatically create or route real tickets.

## Features
- Text input for an IT complaint
- TF-IDF text vectorization
- Logistic Regression classification
- Predicted category and estimated class probabilities
- Streamlit user interface

## Tech stack
Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Streamlit.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## How it works
1. A user enters a complaint in plain English.
2. `TfidfVectorizer` represents words and two-word phrases numerically.
3. `LogisticRegression` predicts one of the known categories.
4. Streamlit displays the predicted category and model probability estimates.

## Important limitations
- The demonstration model uses a small, hand-written dataset of 50 examples.
- There is no separate test set in this first version; the displayed probability is not an accuracy score.
- Short or unfamiliar tickets can be misclassified.
- Do not use it for production ticket routing without a larger representative dataset, train/test evaluation, monitoring, and human review.

## Suggested improvements
- Replace the demo examples with a larger, licensed, labelled dataset.
- Add a held-out test set and report precision, recall, F1-score, and a confusion matrix.
- Add priority prediction and a ticket-history database.
- Add human correction so predictions can be reviewed.
- Deploy the app with Streamlit Community Cloud.

## Learning sources
- Scikit-learn text analytics tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- TF-IDF documentation: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Logistic Regression documentation: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Streamlit documentation: https://docs.streamlit.io/

## Interview summary
“I built a Python web application that classifies IT support complaints into Network, Hardware, or Software. TF-IDF converts the text into numeric features, and Logistic Regression predicts the category. Streamlit provides the interface. The current version is a learning prototype trained on a small hand-written dataset, so I would improve it with more representative data and proper held-out evaluation before production use.”
