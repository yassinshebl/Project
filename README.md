# 📰 Fake News Detector with Streamlit

A machine learning application that classifies news articles as either "Real" or "Fake" using Natural Language Processing (NLP). This project includes a Python script to train and compare several classification models and an interactive web app built with Streamlit to demonstrate the best-performing model.

---

## 🔭 Project Overview

The spread of misinformation is a significant challenge in the digital age. This project aims to tackle this problem by building an AI-powered tool to distinguish between legitimate and fake news articles. The application is built on a dataset of real and fake news, using TF-IDF for text vectorization and comparing several classical machine learning models to find the most effective classifier. The final model is then deployed in a simple, user-friendly web interface.

---

## ✨ Features

- **Model Comparison**: Trains and evaluates three different classification models (Passive-Aggressive, Logistic Regression, and Linear SVM).
- **Text Preprocessing**: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into meaningful numerical representations.
- **Interactive UI**: A web application built with Streamlit that allows users to input a news title and text to get an instant classification.
- **Organized Codebase**: The project is structured into separate folders for data, model training, saved models, and the user interface, following best practices.

---

## 🛠️ Technologies Used

- **Python**: The core programming language.
- **Pandas**: For data manipulation and loading the dataset.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Streamlit**: For creating and serving the interactive web application.
- **Matplotlib & Seaborn**: For data visualization and model comparison plots.
- **Git & GitHub**: For version control and project hosting.

---

## ⚙️ Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yassinshebl/Fake-News-Prediction-Streamlit-Application.git
    cd Fake-News-Prediction-Streamlit-Application
    ```
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the Dataset**:
    Download the `WELFake_Dataset.csv` from [this Kaggle page](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification), unzip it, and place the CSV file inside the `Project/Data` folder.

---

## 🚀 Usage

The project is split into two main parts: training the model and running the application.

1.  **Train the Model**:
    This script will train the models, compare them, and save the best-performing model (`model.pkl`) and the vectorizer (`vectorizer.pkl`) to the `Model` folder.
    ```bash
    cd "Main Code"
    python news_classifier.py
    ```

2.  **Launch the Web App**:
    This command will start the Streamlit server and open the web application in your browser.
    ```bash
    cd "UI"
    streamlit run app.py
    ```

---

## 📊 Model Performance

The models were evaluated on a test set (20% of the data). The Linear SVM model demonstrated the highest accuracy and was chosen for deployment in the Streamlit app.

| Model                 | Accuracy |
|-----------------------|----------|
| **Linear SVM** | **95.68 %** |
| **Passive-Aggressive** | **95.55 %** |
| **Logistic Regression** | **94.14 %** |

*(Note: Accuracy may vary slightly on different runs or machines.)*

---

## 🔮 Future Improvements

- **Deploy to the Cloud**: Deploy the Streamlit application using a service like Streamlit Community Cloud to make it publicly accessible.
- **Use Advanced Models**: Experiment with more complex deep learning models like BERT or other transformers for potentially higher accuracy and better contextual understanding.
- **Add More Features**: Enhance the UI to show prediction confidence scores or highlight the words that most influenced the model's decision.
