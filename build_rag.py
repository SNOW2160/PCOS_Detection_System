import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
PDF_PATH = "data/pcos_medical_book.pdf"
DB_PATH = "models/faiss_pcos_index"

# --- EXECUTION ---
print(f"📖 Loading Medical Book: {PDF_PATH}...")
if not os.path.exists(PDF_PATH):
    print("❌ Error: PDF not found. Please download it to data/ folder.")
    exit()

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"   - Loaded {len(pages)} pages.")

# 2. Split Text (Chunks)
print("✂️  Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # Smaller chunks are better for local models
    chunk_overlap=50
)
chunks = text_splitter.split_documents(pages)
print(f"   - Created {len(chunks)} chunks.")

# 3. Create FAISS Database (The "Offline" Magic)
print("🧠 Creating Local Database ")
try:
    # This downloads the model ONCE (80MB), then works offline forever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"✅ SUCCESS! Database saved locally at '{DB_PATH}'")
except Exception as e:
    print(f"❌ Error: {e}")