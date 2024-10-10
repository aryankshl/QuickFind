import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
import time  # To simulate time for task completion

# Inject custom CSS for a dark theme and modern web-like feel
st.markdown(
    """
    <style>
    /* Dark background color for the whole app */
    .stApp {
        background-color: #1e1e1e;
        font-family: 'Roboto', sans-serif;
        color: #f5f5f5;
    }

    /* Custom Header Styling */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #f5f5f5;
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Card-Like Sections */
    .card {
        background-color: #2a2a2a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Custom Button Styles */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        background-color: #0056b3;
    }

    /* Input Fields Styling (excluding select box) */
    .stTextInput input {
        background-color: #333;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* File Uploader Styling */
    .stFileUploader {
        background-color: #333;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #444;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* Styling for search results */
    .search-result {
        background-color: #2a2a2a;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* Divider Styling */
    .css-1y0tads {
        border-top: 1px solid #444;
        margin: 20px 0;
    }

    /* Spacing between elements */
    .block-container {
        padding: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1>🚀 Modern Search App</h1>", unsafe_allow_html=True)

# Elasticsearch setup without CA certs
indexName = "user_uploaded_data"
try:
    es = Elasticsearch(
        "http://localhost:9200",  # Use HTTP instead of HTTPS to avoid SSL
        basic_auth=("elastic", "12345678"),
        verify_certs=False,  # Disable certificate verification
    )
except ConnectionError as e:
    st.error(f"Connection Error: {e}")

if es.ping():
    st.success("Successfully connected to Elasticsearch!", icon="✅")
else:
    st.error("Cannot connect to Elasticsearch!")

# Model Selection Section (with default styling)
st.markdown("<h2>1. Select a Model</h2>", unsafe_allow_html=True)
with st.container():
    selected_model = st.selectbox("Choose a Sentence Transformer model", ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'])

# File Upload Section
st.markdown("<h2>2. Upload Your CSV Dataset</h2>", unsafe_allow_html=True)
with st.container():
    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# If the dataset is uploaded
if uploaded_file is not None:
    # Load CSV and display a preview
    df = pd.read_csv(uploaded_file)
    df.fillna("none", inplace=True)

    # Show the data as a preview
    st.markdown("<h3>Dataset Preview:</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # Column Selection Section
    st.markdown("<h2>3. Customize Search Results</h2>", unsafe_allow_html=True)
    text_column = st.selectbox("Select the text column (e.g., description)", df.columns)
    id_column = st.selectbox("Select the unique ID column", df.columns)
    display_columns = st.multiselect("Choose columns to display in results", df.columns.tolist(), default=[text_column, id_column])

    # Button to Process and Index Dataset
    if st.button("Process and Index Dataset"):
        progress_bar = st.progress(0)  # Initialize the progress bar
        progress_text = st.empty()  # Placeholder for numerical percentage
        
        st.write("Starting to process the dataset...")
        model = SentenceTransformer(selected_model)
        
        # Simulate progress
        for i in range(0, 101, 10):
            time.sleep(0.2)  # Simulate processing time
            progress_bar.progress(i)  # Update the progress bar
            progress_text.text(f"Progress: {i}%")  # Update numerical percentage
        
        df['DescriptionVector'] = df[text_column].apply(lambda x: model.encode(x, clean_up_tokenization_spaces=False))
        record_list = df.to_dict("records")
        
        if not es.indices.exists(index=indexName):
            es.indices.create(index=indexName)
        
        for idx, record in enumerate(record_list):
            try:
                es.index(index=indexName, document=record, id=record[id_column])
            except Exception as e:
                st.error(f"Error: {e}")
            
            # Update progress bar based on number of records
            progress = int(((idx + 1) / len(record_list)) * 100)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {progress}%")  # Update numerical percentage
        
        st.success("Data indexed successfully!", icon="✅")

# Search Section
st.markdown("<h2>4. Search the Indexed Data</h2>", unsafe_allow_html=True)
search_query = st.text_input("Enter your search query")

if st.button("Search"):
    progress_bar = st.progress(0)  # Initialize the progress bar for searching
    progress_text = st.empty()  # Placeholder for numerical percentage

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
        
        # Simulate progress
        for i in range(0, 101, 10):
            time.sleep(0.1)  # Simulate processing time
            progress_bar.progress(i)  # Update the progress bar
            progress_text.text(f"Progress: {i}%")  # Update numerical percentage
        
        st.markdown("<h3>Search Results:</h3>", unsafe_allow_html=True)
        for result in results:
            if '_source' in result:
                with st.container():
                    st.markdown('<div class="search-result">', unsafe_allow_html=True)
                    for col in display_columns:
                        st.write(f"**{col}:** {result['_source'].get(col, 'No data available')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.divider()
    except Exception as e:
        st.error(f"Search failed: {e}")
