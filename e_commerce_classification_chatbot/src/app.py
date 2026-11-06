import streamlit as st
import sys
import joblib


def classify_item(input_txt):
    tfidf_vector = joblib.load("./e_commerce_tfidf.pkl")
    tfidf_output = tfidf_vector.transform([input_txt])

    prod_model = joblib.load("./e_commerce_classification_model.pkl")
    pred_value = prod_model.predict(tfidf_output)

    encoder = joblib.load("./e_commerce_label_encoder.pkl")
    encode_pred_values= encoder.inverse_transform(pred_value)

    for text, category in zip(input_txt, encode_pred_values):
        print(f"Text: '{text}' ---> Predicted Category: {category}")

    st.text(body=f"Predicted Category: {category}")



def main():
    st.title("Classify Items")

    input_text = st.text_input(label="Input Item Description")
    if st.button(label="Submit"):
        classify_item(input_text)


if __name__ == "__main__":
    sys.exit(main())