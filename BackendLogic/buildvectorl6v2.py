import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Setup Embedder
print("🔌 Initializing local model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Check Folder
folder_path = "singapore_data"
if not os.path.exists(folder_path):
    print(f"❌ ERROR: Folder '{folder_path}' not found.")
    exit()

# 3. Robust Loading Loop (The Fix)
print(f"📂 Scanning '{folder_path}' for .pdf files...")

# Find all PDF files manually
pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
print(f"   Found {len(pdf_files)} PDF files. Starting load...")

documents = []
failed_files = 0

for file_path in pdf_files:
    try:
        # Try to load this specific file
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
        print(f"   ✅ Loaded: {os.path.basename(file_path)}")
        
    except Exception as e:
        # If it fails, print error but DO NOT crash
        print(f"   ⚠️ CORRUPT - SKIPPING: {os.path.basename(file_path)}")
        failed_files += 1

print(f"\n📊 Summary: {len(documents)} pages loaded. {failed_files} files failed.")

if len(documents) == 0:
    print("❌ ERROR: No valid PDFs could be loaded. Exiting.")
    exit()

# 4. Split Text
print("✂️ Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"   Created {len(chunks)} text chunks.")

# 5. Build & Save
print("🧠 Building Index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index_minilm")

print("✅ SUCCESS! Database saved to 'faiss_index_minilm'.")