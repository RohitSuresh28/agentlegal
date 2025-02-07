import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
import os
from pinecone import Pinecone, ServerlessSpec
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import plotly.graph_objects as go

nltk.download('vader_lexicon')

# Streamlit session states
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "feedback_history" not in st.session_state:
    st.session_state["feedback_history"] = []

if "risk_factor" not in st.session_state:
    st.session_state["risk_factor"] = None

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Streamlit page config
st.set_page_config(page_title="AgentLegal", layout="wide")

dark_mode = st.sidebar.checkbox("🌓 Toggle Dark Mode", value=st.session_state["dark_mode"])
st.session_state["dark_mode"] = dark_mode

# Add custom CSS for background color and conversational layout
if dark_mode:
    st.markdown(
        """
        <style>
        /* Dark mode styles */
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        input {
            border: 2px solid #055289;
            border-radius: 10px;
            padding: 10px;
            color: white;
            background-color: #333;
            font-family: 'Arial', sans-serif;
        }
        button {
            background-color: #055289;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            border-radius: 10px;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            padding: 20px;
            color: white;
        }
        h1, h2, h3 {
            color: #055289;
        }
        .chatbox {
            background-color: #2a2a2a;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
            font-family: 'Arial', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        /* Light mode styles */
        .stApp {
            background-color: #f0f2f6;
            color: black;
        }
        input {
            border: 2px solid #055289;
            border-radius: 10px;
            padding: 10px;
            color: #333;
        }
        button {
            background-color: #055289;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #055289;
        }
        .chatbox {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True
    )

st.sidebar.title("Legal Assistant Chatbot")
st.sidebar.write("Ask legal questions, and the chatbot will provide answers based on Indian legal knowledge.")
st.sidebar.header("📊 AI Insights Dashboard")


total_queries = 0
avg_response_time = 0.0
sentiment_label = "N/A"
compound_score = 0.0


if "chat_history" in st.session_state:
   total_queries = len(st.session_state["chat_history"]) // 2  # Each query has a response
   avg_response_time = round(random.uniform(1, 2), 2)  # Simulating average response time
   sentiment_analyzer = SentimentIntensityAnalyzer()


   # Sentiment analysis
   if "sentiments" not in st.session_state:
       st.session_state["sentiments"] = []


   # Analyze the sentiment of the user's messages
   for i, (speaker, msg) in enumerate(st.session_state["chat_history"]):
       if speaker == "You":  # Only analyze user queries
           sentiment_score = sentiment_analyzer.polarity_scores(msg)
           st.session_state["sentiments"].append(sentiment_score)


   # Display sentiment score for the last user message
   if st.session_state["sentiments"]:
       last_sentiment = st.session_state["sentiments"][-1]
       compound_score = last_sentiment['compound']


       # Determine if the sentiment is Positive, Negative, or Neutral
       if compound_score > 0:
           sentiment_label = "Positive"
       elif compound_score < 0:
           sentiment_label = "Negative"
       else:
           sentiment_label = "Neutral"


# Show some insights on the sidebar
st.sidebar.metric("Total Queries", total_queries)
st.sidebar.metric("Avg Response Time (s)", avg_response_time)
st.sidebar.subheader("Sentiment Analysis")
st.sidebar.write(f"Sentiment of last user message: {sentiment_label} (Score: {compound_score})")
st.title("AgentLegal")
st.write("Welcome to the Legal Assistant Agent! Ask questions about legal matters, contracts, compliance, and more.")

# API setup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_RxSgs_HHEPUnvBpwR7MwXoZyac1WT8JSU3nPuTn6v3esC96WDvq3HPpeF5DbzkcUbonze"
os.environ["PINECONE_ENV"] = "us-east-1"

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "atmecs"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=os.environ["PINECONE_ENV"])
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embedding_model)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Small-Instruct-2409",
    task="text-generation",
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

col1, col2 = st.columns([3, 1])

# Chatbot
with col1:
    user_query = st.text_input("Enter your legal question:")
    query_embedding = embedding_model.embed_query(user_query)
    if user_query:
        instruction = "You are a highly knowledgeable and professional legal assistant with expertise in the Indian legal system, corporate law, contract law, intellectual property law, and civil and criminal law, equipped to provide thorough, accurate, and legally sound answers; ensure clarity, precision, and relevance in all responses, addressing the user's inquiry with a comprehensive legal explanation, applicable statutes, relevant case law, and procedural guidance while avoiding incomplete answers or generalizations; always cite authoritative sources and legal precedents when applicable, offering practical advice and solutions based on the current legal framework in India, and ensure adherence to confidentiality, ethical standards, and legal norms in all interactions. If you are asked non legal questions, reply politely."
        query_with_instruction = instruction + " " + user_query
        
        with st.spinner('Processing your query...'):
            start_time = time.time()
            response = qa_chain.run(query=query_with_instruction, max_length=500, return_only_outputs=True)
            end_time = time.time()
            response_time = end_time - start_time
        
        # Debugging: Log the full response before processing
        st.write("Full response from model:", response)

        # Extract the helpful answer
        if isinstance(response, str):
            helpful_answer = response.split("Helpful Answer:")[-1].strip()  # Split if answer is formatted that way
        else:
            helpful_answer = response.get("output", "No helpful answer found").strip()
        
        # Debugging: Log the extracted helpful answer
        st.write("Extracted helpful answer:", helpful_answer)

        # If the helpful answer is empty or not as expected, try to inspect the response further
        if not helpful_answer or "No helpful answer found" in helpful_answer:
            st.warning("The answer seems unusual. Here's the full response for further debugging:")
            st.write(response)

        st.session_state["chat_history"].append(("You", user_query))
        st.session_state["chat_history"].append(("AgentLegal", helpful_answer))

# Display chat history with black text for responses
for speaker, msg in st.session_state["chat_history"]:
    text_color = "black" if speaker == "AgentLegal" else "#333"
    st.markdown(f"<div style='padding:10px; border-radius:10px; margin:5px; background:#f5f5f5; color:{text_color};'><strong>{speaker}:</strong> {msg}</div>", unsafe_allow_html=True)

with col2:
    st.header("Risk Factor")
    risk_factor = st.number_input("Enter Risk Factor:", min_value=0, max_value=100, value=st.session_state["risk_factor"])
    if st.button("Update Risk"):
        st.session_state["risk_factor"] = risk_factor
        st.success(f"Risk Factor updated to {risk_factor}!")
    st.subheader("Current Risk Factor")
    st.markdown(f"<h3 style='color: #dc3545;'>{st.session_state['risk_factor']}</h3>", unsafe_allow_html=True)
    st.write(f"Calculated Risk Factor: {st.session_state['risk_factor']}")
