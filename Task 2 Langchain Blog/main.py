from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# STEP 1: LLM (HuggingFace - No API Key)
# -------------------------------
pipe = pipeline("text-generation", model="distilgpt2", max_new_tokens=80)
llm = HuggingFacePipeline(pipeline=pipe)

print("\n--- Basic LLM Call ---")
print(llm.invoke("Explain LangChain in one line:"))

# -------------------------------
# STEP 2: PromptTemplate
# -------------------------------
print("\n--- PromptTemplate Example ---")
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms with an example."
)

formatted_prompt = prompt.format(topic="LangChain")
print(llm.invoke(formatted_prompt))

# -------------------------------
# STEP 3: Chain
# -------------------------------
print("\n--- Chain Example ---")
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"topic": "Vector Databases"}))

# -------------------------------
# STEP 4: Tool
# -------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

print("\n--- Tool Example ---")
print(multiply.invoke({"a": 6, "b": 7}))

# -------------------------------
# STEP 5: Simple Agent Simulation
# -------------------------------
print("\n--- Simple Agent Example ---")

def simple_agent(query: str):
    numbers = [int(x) for x in query.split() if x.isdigit()]
    if "multiply" in query and len(numbers) >= 2:
        result = multiply.invoke({"a": numbers[0], "b": numbers[1]})
        return f"Tool Used: multiply → Result: {result}"
    return "No tool required."

print(simple_agent("multiply 12 8"))

# -------------------------------
# STEP 6: Document Loader
# -------------------------------
print("\n--- Document Loader Example ---")
loader = TextLoader("sample.txt")
docs = loader.load()
print(docs[0].page_content)

# -------------------------------
# STEP 7: Text Splitting
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)

print("\n--- Document Chunks ---")
print(chunks[0].page_content)

# -------------------------------
# STEP 8: Embeddings + Vector Store (FAISS)
# -------------------------------
print("\n--- Vector Store (FAISS) Example ---")
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()
results = retriever.invoke("What is LangChain used for?")

print("\n--- Retrieved Result ---")
print(results[0].page_content)
