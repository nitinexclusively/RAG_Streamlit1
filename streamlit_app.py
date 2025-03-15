import streamlit as st
import faiss
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np


# ===========================
# Load embedding & language model
# ===========================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
language_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# ===========================
# Load financial data from CSV
# ===========================
data = pd.read_csv('data/microsoft_detailed_financials.csv')

# Combine all columns as text chunks for retrieval
data['combined_text'] = data.apply(lambda x: f"Year: {x['Year']}, Quarter: {x['Quarter']}, Revenue: ${x['Revenue_Billion_USD']} Billion, Net Income: ${x['Net_Income_Billion_USD']} Billion", axis=1)
texts = data['combined_text'].tolist()

# Tokenize the text corpus
tokenized_corpus = [text.split() for text in texts]

# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_corpus)

# ===========================
# Index text chunks with FAISS
# ===========================
embeddings = embedding_model.encode(texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Initialize combined_results to avoid NameError
combined_results = []

# ===========================
# Streamlit Web App
# ===========================
st.title("Microsoft Financial Question Answering System")
query = st.text_input("Ask any financial question based on the financials")

if query:
    # ===========================
    # Hybrid Search (Dense + Sparse)
    # ===========================
    
    # Sparse Search using BM25
    tokenized_query = query.split()
    bm25_results = bm25.get_top_n(tokenized_query, texts, n=5)
    
    # Dense Search using FAISS
    query_embedding = embedding_model.encode(query)
    D, I = index.search(query_embedding.reshape(1, -1), 5)
    dense_results = [texts[i] for i in I[0]]

    # Combine both results
    combined_results = list(set(bm25_results[:2] + dense_results[:2]))

# ===========================
# Generate prompt for LLM
# ===========================
answer = None  # Initialize answer to avoid NameError

if combined_results:
    input_text = "The following are the financial statements of Microsoft.\n\n"
    for i, chunk in enumerate(combined_results):
        if i < 2:  # Only pass the top 2 chunks
            input_text += f"Chunk {i+1}: {chunk}\n\n"
    input_text += f"\nQuestion: {query}\nAnswer:"

    # Pass to language model
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = language_model.generate(**inputs, max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ===========================
    # Calculate Confidence Score (Only if Answer is Found)
    # ===========================
    if answer:
        combined_text_embedding = embedding_model.encode(answer)
        D, I = index.search(combined_text_embedding.reshape(1, -1), 1)
        confidence_score = round((1 - D[0][0]) * 100, 2)
        confidence_score = max(0, min(confidence_score, 100))

        # Display Answer in Streamlit
        st.markdown("### Answer:")
        st.success(answer)
        st.markdown(f"**Confidence Score:** {confidence_score}%")
    else:
        st.warning("No relevant financial statements found.")
else:
    st.warning("Please enter a financial question.")