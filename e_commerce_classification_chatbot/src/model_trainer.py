
import sys

from data import data_analysis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import time
import joblib



def split_data(df):
    print("Step 1: Cleaning dataset indexes...")
    df = df.reset_index(drop=True)

    print(df['encoded_category'].unique())
    print("Step 2: Splitting data using native train_test_split...")
    start_time = time.time()

    X = df['description']
    y = df['encoded_category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"-> Splitting finished in {time.time() - start_time:.2f} seconds.")

    return X_train, X_test, y_train, y_test


def main():
    df = data_analysis.main()
    print(df.info())

    # label encoding
    label_encoder = LabelEncoder()
    df['encoded_category'] = label_encoder.fit_transform(df['category'])

    # split data
    X_tr, X_te, y_tr, y_te = split_data(df)

    print("Step 1: Vectorizing text data (Optimized)...")
    start_time = time.time()

    # create tf-idf vector
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 1), min_df=2)
    X_tr_tfidf = tfidf.fit_transform(X_tr)

    X_te_tfidf = tfidf.transform(X_te)
    print(f"-> Vectorization finished in {time.time() - start_time:.2f} seconds.")

    print("\nStep 2: Training SGD Classifier...")
    start_time = time.time()
    # pass it to Logistic Regression
    log_model = SGDClassifier(loss="log_loss", max_iter=20, random_state=42)
    log_model.fit(X_tr_tfidf, y_tr)
    print(f"-> Training finished in {time.time() - start_time:.2f} seconds.")

    # accuracy
    accuracy = log_model.score(X_te_tfidf, y_te)

    print(f"\nFinal Test Accuracy: {accuracy:.2%}")

    joblib.dump(log_model, "e_commerce_classification_model.pkl")
    joblib.dump(tfidf, "e_commerce_tfidf.pkl")
    joblib.dump(label_encoder, "e_commerce_label_encoder.pkl")


if __name__ == "__main__":
    sys.exit(main())