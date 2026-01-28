#We are gonna use this to convert goveerment pdf files to vector data so that and Ai Model can work efficiently without giving false information baically this is the first step implement Rag.
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Setup Embedder
print("🔌 Initializing local model (all-MiniLM-L6-v2)...")       #this is the most commonly used Ai model to integrate Retrieval Augmented Generation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")            #using hugging faces platform to implement this model

# 2. Check Folder(Use your Own folder with the pdf files)
folder_path = "singapore_data" # edit the folder name here
if not os.path.exists(folder_path):
    print(f"❌ ERROR: Folder '{folder_path}' not found.")
    exit()


print(f"📂 Scanning '{folder_path}' for .pdf files...")

# Find all PDF files manually
pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True) # reads pdfs
print(f"   Found {len(pdf_files)} PDF files. Starting load...")

documents = []
failed_files = 0 # shows corrupted pdfs

for file_path in pdf_files:
    try:
        # Try to load this specific file
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
        print(f"   ✅ Loaded: {os.path.basename(file_path)}")
        
    except Exception as e:
        # If the pdf is corrupt make the program continue(so that we dont lose our progress)
        print(f"   ⚠️ CORRUPT - SKIPPING: {os.path.basename(file_path)}")
        failed_files += 1

print(f"\n📊 Summary: {len(documents)} pages loaded. {failed_files} files failed.")

if len(documents) == 0:
    print("❌ ERROR: No valid PDFs could be loaded. Exiting.")
    exit()

# 4. Splitting text and dividing them into chunks
print("✂️ Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"   Created {len(chunks)} text chunks.")

# 5. building the final vector file and saving  it
print("🧠 Building Index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index_minilm")#you can rename the vector file if you want


print("✅ SUCCESS! Database saved to 'faiss_index_minilm'.")
