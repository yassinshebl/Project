import streamlit as st
import pickle
import os

@st.cache_data
def load_assets(model_folder_path):
    model_path = os.path.join(model_folder_path, 'model.pkl')
    vectorizer_path = os.path.join(model_folder_path, 'vectorizer.pkl')
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

MODEL_FOLDER_PATH = '../Model'
model, vectorizer = load_assets(MODEL_FOLDER_PATH)
print("Model and Vectorizer loaded successfully.")

st.title("📰 Fake News Detector")
st.write("Enter the title and text of a news article to check if it's real or fake.")

news_title = st.text_input("News Title")
news_text = st.text_area("News Text", height=200)

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