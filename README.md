# Fake News Detector 📰

This project is a machine learning application that classifies news articles as either "Real" or "Fake" using Natural Language Processing (NLP). It includes a script to train and compare several models and a Streamlit web app to interact with the best-performing model.

## File Structure

- `/data`: Contains the training dataset. (Note: The dataset is not uploaded to GitHub).
- `/Main Code`: Contains the main Python script (`news_classifier.py`) for training and evaluating the models.
- `/Model`: The folder where the trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) are saved.
- `/UI`: Contains the Streamlit application (`app.py`).

## How to Run

1.  **Train the Model**:
    Navigate to the `Main Code` folder and run the script. This will create the `model.pkl` and `vectorizer.pkl` files in the `Model` folder.
    ```bash
    cd "Main Code"
    python news_classifier.py
    ```

2.  **Launch the Web App**:
    Navigate to the `UI` folder and run the Streamlit app.
    ```bash
    cd "UI"
    streamlit run app.py
    ```
