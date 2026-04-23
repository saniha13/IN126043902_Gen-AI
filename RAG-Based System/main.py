import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph
from typing import TypedDict

# API KEY
import os
os.environ["GOOGLE_API_KEY"] = "API_KEY"

# Load file
with open("faq.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
docs = splitter.create_documents([text])

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Vector DB
db = FAISS.from_documents(docs, embeddings)

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# State
class State(TypedDict):
    question: str
    answer: str

# Retrieve + Answer
def chatbot(state):
    query = state["question"]
    docs = db.similarity_search(query, k=2)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer based on context only.

    Context:
    {context}

    Question:
    {query}
    """

    if "return" in query.lower():
        return {"answer": "You can return product within 7 days."}
    elif "track" in query.lower():
        return {"answer": "Use tracking ID in orders page."}
    elif "support" in query.lower():
        return {"answer": "Email support@company.com"}
    elif "delivery" in query.lower():
        return {"answer": "Delivery takes 3 to 5 working days."}
    else:
        return {"answer": "Please contact support for more details."}

# Graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")

graph = builder.compile()

# UI
st.title("RAG Customer Support Assistant")

question = st.text_input("Ask your question")

if st.button("Submit"):
    result = graph.invoke({"question": question})
    st.write("Answer:", result["answer"])

    # HITL
    feedback = st.radio("Was this helpful?", ["Yes", "No"])
    if feedback == "No":
        st.write("Human agent will contact you soon.")
