import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# Usage - Local-> run streamlit run main.py
class BreastCancerClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def clean_data(self):
        self.data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

    def encode_diagnosis(self):
        self.data['diagnosis'] = np.where(self.data['diagnosis'] == 'M', 1, 0)

    def split_data(self):
        X = self.data.drop('diagnosis', axis=1)
        Y = self.data['diagnosis']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def visualize_correlation(self):
        corr = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
        st.pyplot(plt)

    def visualize_scatterplot(self):
        malignant_data = self.data[self.data['diagnosis'] == 1]
        benign_data = self.data[self.data['diagnosis'] == 0]
        plt.figure(figsize=(10, 8))
        plt.scatter(malignant_data['radius_mean'], malignant_data['texture_mean'], color='red', label='Malignant (M)')
        plt.scatter(benign_data['radius_mean'], benign_data['texture_mean'], color='blue', label='Benign (B)')
        plt.xlabel('Radius Mean')
        plt.ylabel('Texture Mean')
        plt.title('Radius Mean vs Texture Mean')
        plt.legend()
        st.pyplot(plt)

    def train_knn(self):
        param_grid = {'n_neighbors': np.arange(1, 20)}
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5)
        grid_search.fit(self.X_train, self.Y_train)
        best_params = grid_search.best_params_
        knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
        knn.fit(self.X_train, self.Y_train)
        Y_pred = knn.predict(self.X_test)
        return best_params, Y_pred

    def train_svm(self):
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(self.X_train, self.Y_train)
        best_params = grid_search.best_params_
        svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
        svm.fit(self.X_train, self.Y_train)
        Y_pred = svm.predict(self.X_test)
        return best_params, Y_pred

    def train_naive_bayes(self):
        nb = GaussianNB()
        nb.fit(self.X_train, self.Y_train)
        Y_pred = nb.predict(self.X_test)
        return {}, Y_pred

    def show_results(self, model_name, best_params, Y_pred):
        st.subheader(f"Model: {model_name}")
        st.write("En İyi Parametreler:", best_params)
        accuracy = accuracy_score(self.Y_test, Y_pred)
        precision = precision_score(self.Y_test, Y_pred)
        recall = recall_score(self.Y_test, Y_pred)
        f1 = f1_score(self.Y_test, Y_pred)

        st.subheader(f"Görev 4: Model Sonuçları ({model_name})")
        
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1-score:", f1)

        cm = confusion_matrix(self.Y_test, Y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

def main():
    st.title("Breast Cancer Wisconsin (Diagnostic) Data Set")
    # Veri seti seçimi
    dataset_name = st.sidebar.selectbox(
        "Veri Seti Seç",
        ("Breast Cancer Wisconsin (Diagnostic) Data Set",)
    )
    if dataset_name == "Breast Cancer Wisconsin (Diagnostic) Data Set":
        # Veri seti yükleme
        file_path = "data.csv"
        classifier = BreastCancerClassifier(file_path)
        classifier.load_data()

        # Görev 1: Veri setinin ilk 10 satırını ve sütunlarını gösterme
        st.subheader("Görev 1: Veri Setinin İlk 10 Satırı ve Sütunları")
        st.write(classifier.data.head(10))
        st.write("Sütunlar:", classifier.data.columns)

        # Görev 2: Veri temizleme ve ön işleme adımlarını gerçekleştirme
        classifier.clean_data()
        classifier.encode_diagnosis()
        st.subheader("Görev 2: Verinin Son 10 Satırı")
        st.write(classifier.data.tail(10))

        # Korelasyon matrisini çizdirme
        st.subheader("Korelasyon Matrisi")
        classifier.visualize_correlation()

        st.subheader("Korelasyon Grafiği")
        classifier.visualize_scatterplot()

        # Görev 3: Model implementasyonu
        st.subheader("Görev 3: Model Implementasyonu")
        model_name = st.sidebar.selectbox(
            "Model Seç",
            ("KNN", "SVM", "Naive Bayes")
        )

        classifier.split_data()

        if model_name == "KNN":
            best_params, Y_pred = classifier.train_knn()
            classifier.show_results("KNN", best_params, Y_pred)
        elif model_name == "SVM":
            best_params, Y_pred = classifier.train_svm()
            classifier.show_results("SVM", best_params, Y_pred)
        elif model_name == "Naive Bayes":
            best_params, Y_pred = classifier.train_naive_bayes()
            classifier.show_results("Naive Bayes", best_params, Y_pred)
        else:
            st.write("Lütfen bir model seçin.")

if __name__ == "__main__":
    main()
