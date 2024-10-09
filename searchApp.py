import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd


# Elasticsearch setup
indexName = "user_uploaded_data"
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "iammilind"),
        ca_certs="C:/elasticsearch-8.15.2/config/certs/http_ca.crt"
    )
except ConnectionError as e:
    st.error(f"Connection Error: {e}")

if es.ping():
    st.success("Successfully connected to Elasticsearch!", icon="✅")
else:
    st.error("Cannot connect to Elasticsearch!")

# Title and Model Selection
st.markdown("# 🔍 *Flexible Search Engine*")

st.markdown("### 📘 1. Select a model")
selected_model = st.selectbox("Choose a Sentence Transformer model", ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'])

# File Upload Section
st.markdown("### 📂 2. Upload your CSV dataset")
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    # Load CSV and display a preview
    df = pd.read_csv(uploaded_file)
    df.fillna("none", inplace=True)
    st.dataframe(df.head())

    st.markdown("### ⚙️ 3. Customize your search results")
    text_column = st.selectbox("Select the text column (e.g., description)", df.columns)
    id_column = st.selectbox("Select the unique ID column", df.columns)
    display_columns = st.multiselect("Choose columns to display in results", df.columns.tolist(), default=[text_column, id_column])

    if st.button("Process and Index Dataset"):
        st.write("Starting to process the dataset...")
        model = SentenceTransformer(selected_model)
        df['DescriptionVector'] = df[text_column].apply(lambda x: model.encode(x, clean_up_tokenization_spaces=False))
        record_list = df.to_dict("records")
        if not es.indices.exists(index=indexName):
            es.indices.create(index=indexName)
        for record in record_list:
            try:
                es.index(index=indexName, document=record, id=record[id_column])
            except Exception as e:
                st.error(f"Error: {e}")
        st.success("Data indexed successfully!", icon="✅")

# Search Section
st.markdown("### 🔎 4. Search the indexed data")
search_query = st.text_input("Enter your search query")

if st.button("Search"):
    model = SentenceTransformer(selected_model)
    vector_of_input_keyword = model.encode(search_query)

    query = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 500
    }

    try:
        res = es.knn_search(index=indexName, knn=query, source=display_columns)
        results = res["hits"]["hits"]

        st.subheader("Search Results")
        for result in results:
            if '_source' in result:
                with st.container():
                    for col in display_columns:
                        st.write(f"*{col}:* {result['_source'].get(col, 'No data available')}")
                    st.divider()
    except Exception as e:
        st.error(f"Search failed: {e}")