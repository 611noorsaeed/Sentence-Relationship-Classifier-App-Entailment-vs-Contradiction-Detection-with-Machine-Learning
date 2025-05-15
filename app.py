import streamlit as st
import pickle

# Load the saved model and TF-IDF vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

# Detection function
def detect(inp1, inp2, model, tfidf):
    # Combine the input sentences as required by the model
    combined_input = inp1 + " " + inp2  # Concatenating the sentences

    combined_input = combined_input.lower()

    # Transform the combined input using the TF-IDF vectorizer
    transformed_input = tfidf.transform([combined_input])

    # Make prediction using the model
    prediction = model.predict(transformed_input)

    return prediction[0]

# Streamlit UI
st.title("Sentence Pair Relationship Classifier")
st.write("Enter two sentences to classify the relationship as Entailment or Contradiction.")

inp1 = st.text_input("Sentence 1")
inp2 = st.text_input("Sentence 2")

if st.button("Predict"):
    result = detect(inp1, inp2, model, tfidf)
    st.success(f"Prediction: **{result.upper()}**")
