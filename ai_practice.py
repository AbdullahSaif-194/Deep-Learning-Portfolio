import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

# NEW IMPORTS FOR 2026
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- 1. INITIALIZE SETTINGS ---
model_name = "BAAI/bge-small-en-v1.5"
qdrant_api_key = os.getenv("qdrant_api_key")
qdrant_url = os.getenv("qdrant_url")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
collection_name = "practice_bot"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# --- 2. VECTOR STORE CONNECTION ---
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

try:
    collection_info = client.get_collection(collection_name=collection_name)
    count = collection_info.points_count
except Exception:
    count = 0

if count == 0:
    print("⏳ Processing PDF...")
    loader = PyPDFLoader(r"A:\AI advance course\scripts\kb-136-will-ai-transform-pakistan-assessing-the-2025-national-policy.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    
    doc_list = []
    for page in pages:
        metadata = {"source": "National Policy Document", "page": page.metadata["page"]}
        chunks = text_splitter.create_documents([page.page_content], metadatas=[metadata])
        doc_list.extend(chunks)
        
    qdrant = QdrantVectorStore.from_documents(
        doc_list, embeddings,
        collection_name=collection_name,
        url=qdrant_url, api_key=qdrant_api_key
    )
else:
    print("📚 Connected to existing Qdrant collection.")
    qdrant = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

# --- 3. INITIALIZE LLM & RAG CHAIN ---
# --- 3. INITIALIZE LLM (Updated for 2026) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", # Use the 2026 standard model
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)

# Define the Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the chains
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(qdrant.as_retriever(), combine_docs_chain)

# --- 4. CHAT LOOP ---
print("\n" + "="*50)
print("🤖 PAKISTAN AI POLICY CHATBOT ACTIVE")
print("="*50 + "\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']: break
    
    print("\nThinking... 🧠")
    try:
        # Note: New chains use 'input' and return 'answer'
        response = qa_chain.invoke({"input": user_input})
        print(f"\nBot: {response['answer']}\n")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Error: {e}")