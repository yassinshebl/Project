import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import pickle
import os

class FakeNewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.models = {
            'Passive-Aggressive': PassiveAggressiveClassifier(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Linear SVM': LinearSVC(random_state=42)
        }
        self.results = {}
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        print("FakeNewsClassifier initialized for model comparison!")

    def load_data(self, filepath):
        print("\n--- Loading Data ---")
        self.data = pd.read_csv(filepath)
        self.data.drop(columns=['Unnamed: 0'], inplace=True)
        self.data.dropna(inplace=True)
        print("Dataset loaded successfully.")
        print("\n", self.data.head())

    def prepare_and_vectorize_data(self):
        print("\n--- Preparing and Vectorizing Data ---")
        features = self.data['text']
        labels = self.data['label']
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        self.X_train = self.vectorizer.fit_transform(X_train_raw)
        self.X_test = self.vectorizer.transform(X_test_raw)
        print("Data has been prepared and vectorized.")
        
    def train_and_evaluate_models(self):
        print("\n--- Training and Evaluating Models ---")
        for name, model in self.models.items():
            print(f"\n----- Testing Model: {name} -----")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results[name] = accuracy
            print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Fake (0)', 'Real (1)']))
    
    def display_comparison(self):
        print("\n\n--- Final Model Comparison ---")
        results_df = pd.DataFrame(list(self.results.items()), columns=['Model', 'Accuracy'])
        results_df = results_df.sort_values(by='Accuracy', ascending=False)
        print(results_df)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Accuracy', y='Model', data=results_df)
        plt.title('Model Accuracy Comparison')
        plt.xlim(0.9, 1.0)
        for index, value in enumerate(results_df['Accuracy']):
            plt.text(value, index, f' {value:.4f}', va='center')
        print("\nShowing the comparison plot. Close the window to exit.")
        plt.show()

    def save_best_model_and_vectorizer(self, model_folder_path):
        print("\n--- Saving the best model and vectorizer ---")
        
        os.makedirs(model_folder_path, exist_ok=True)
        
        best_model_name = max(self.results, key=self.results.get)
        best_model = self.models[best_model_name]
        print(f"Best performing model is '{best_model_name}' with accuracy: {self.results[best_model_name]:.2%}")

        model_path = os.path.join(model_folder_path, 'model.pkl')
        vectorizer_path = os.path.join(model_folder_path, 'vectorizer.pkl')

        with open(model_path, 'wb') as model_file:
            pickle.dump(best_model, model_file)
        
        with open(vectorizer_path, 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)
            
        print(f"Model and vectorizer have been saved in '{model_folder_path}'")

if __name__ == "__main__":
    DATA_FILE_PATH = r'Data\WELFake_Dataset.csv' 
    MODEL_FOLDER_PATH = r'Model'

    classifier = FakeNewsClassifier()
    
    classifier.load_data(DATA_FILE_PATH)
    classifier.prepare_and_vectorize_data()
    classifier.train_and_evaluate_models()
    classifier.display_comparison()
    classifier.save_best_model_and_vectorizer(MODEL_FOLDER_PATH)