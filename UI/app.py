# UI/app.py
import streamlit as st
import pickle
import os # Import os to handle file paths

# --- Load the saved model and vectorizer ---
@st.cache_data
def load_assets(model_folder_path):
    """Loads the saved model and vectorizer from .pkl files."""
    # UPDATED FILE PATHS
    model_path = os.path.join(model_folder_path, 'model.pkl')
    vectorizer_path = os.path.join(model_folder_path, 'vectorizer.pkl')
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# UPDATED FILE PATH to point to the Model folder
MODEL_FOLDER_PATH = '../Model'
model, vectorizer = load_assets(MODEL_FOLDER_PATH)

# --- Streamlit App Interface ---
st.title("📰 Fake News Detector")
st.write("Enter the title and text of a news article to check if it's real or fake.")

# --- User Input ---
news_title = st.text_input("News Title")
news_text = st.text_area("News Text", height=200)

# --- Prediction Logic ---
if st.button("Check News"):
    if news_title and news_text:
        full_text = news_title + " " + news_text
        
        vectorized_text = vectorizer.transform([full_text])
        prediction = model.predict(vectorized_text)[0]
        
        st.write("---")
        if prediction == 0:
            st.success("✅ This looks like a Real News article.")
        else:
            st.error("❌ This looks like a Fake News article.")
    else:
        st.warning("Please enter both a title and text for the article.")