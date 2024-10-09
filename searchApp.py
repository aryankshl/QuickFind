# Modified code for Streamlit UI in searchApp.py
import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
from io import StringIO

# Define the list of models to be used for the dropdown
MODEL_OPTIONS = {
    'all-mpnet-base-v2': 'all-mpnet-base-v2',
    'distilbert-base-nli-stsb-mean-tokens': 'distilbert-base-nli-stsb-mean-tokens',
    'paraphrase-MiniLM-L6-v2': 'paraphrase-MiniLM-L6-v2',
    'roberta-base-nli-stsb-mean-tokens': 'roberta-base-nli-stsb-mean-tokens'
}

# Elasticsearch setup
indexName = "user_uploaded_data"
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "iamaryan"),
        ca_certs="C:/elasticsearch-8.15.2/config/certs/http_ca.crt"
    )
except ConnectionError as e:
    st.error(f"Connection Error: {e}")

if es.ping():
    st.success("Successfully connected to Elasticsearch!")
else:
    st.error("Cannot connect to Elasticsearch!")

# UI: Dropdown for model selection
st.title("Flexible Search Engine")

st.header("1. Select a model")
selected_model = st.selectbox("Choose a Sentence Transformer model", list(MODEL_OPTIONS.keys()))

# UI: File upload for dataset
st.header("2. Upload your CSV dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# When the dataset is uploaded
if uploaded_file is not None:
    # Load CSV and display a preview
    df = pd.read_csv(uploaded_file)

    # Replace NaN values with the string "none"
    df.fillna("none", inplace=True)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Dropdown for column selection
    st.header("3. Map your dataset columns")
    text_column = st.selectbox("Select the column for the text data (e.g., description)", df.columns)
    id_column = st.selectbox("Select the column for the unique ID (e.g., ProductID)", df.columns)

    # Button to process the dataset and index it
    if st.button("Process and Index Dataset"):
     st.write("Starting to process the dataset...")  # Progress message

     # Load the selected model
     st.write("Loading the selected model...")
     model = SentenceTransformer(selected_model)

     # Generate vector embeddings for the selected text column
     st.info(f"Generating embeddings using model: {selected_model}")
     df['DescriptionVector'] = df[text_column].apply(lambda x: model.encode(x, clean_up_tokenization_spaces=False))
     st.write("Embeddings generated successfully.")

     # Prepare data for Elasticsearch indexing
     record_list = df.to_dict("records")
     st.write("Data prepared for indexing.")

     # Check if index already exists, if not create it
     if not es.indices.exists(index=indexName):
         es.indices.create(index=indexName, ignore=400)
         st.write(f"Created index: {indexName}")

     # Index the data into Elasticsearch
     for record in record_list:
         try:
             es.index(index=indexName, document=record, id=record[id_column])
             st.write(f"Indexed record with ID: {record[id_column]}")
         except Exception as e:
             st.error(f"Error indexing record {record[id_column]}: {e}")
     
     st.success("Dataset processed and indexed successfully!")


# UI: Search section
st.header("4. Search the indexed data")

# Input for search query
search_query = st.text_input("Enter your search query")

if st.button("Search"):
    if search_query:
        # Perform search
        model = SentenceTransformer(selected_model)
        vector_of_input_keyword = model.encode(search_query)
        
        query = {
            "field": "DescriptionVector",
            "query_vector": vector_of_input_keyword,
            "k": 10,
            "num_candidates": 500
        }
        
        try:
            res = es.knn_search(index=indexName, knn=query, source=["ProductName", text_column])
            results = res["hits"]["hits"]

            # Display search results
            st.subheader("Search Results")
            for result in results:
                if '_source' in result:
                    st.header(f"{result['_source'].get('ProductName', 'Unnamed Product')}")
                    st.write(f"Description: {result['_source'].get(text_column, 'No description available')}")
                    st.divider()
        except Exception as e:
            st.error(f"Search failed: {e}")
