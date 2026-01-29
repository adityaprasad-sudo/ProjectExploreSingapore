# --- INSTALL STUFF ---
print("instaling libs...")
!pip install -q langchain-community pypdf tqdm langchain-google-genai faiss-cpu requests==2.32.4

# VERY IMPORTANT use this python file if yoy are using google's text embedder

import os
import zipfile
import time
import getpass
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# getting the api key securely
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("enter google api key: ")

# check if zip exists and unzip it
if os.path.exists("singapore_data.zip"):
    print("unzipping data...")
    with zipfile.ZipFile("singapore_data.zip", 'r') as zip_ref:
        zip_ref.extractall(".") # extract here
    print("unzip done")
else:
    print("cant find 'singapore_data.zip', asuming files are already here")

# setup the google embedding model
print("setting up google model...")
# using the basic embedding-001
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print("scanning folders for pdfs...")
pdf_files = []
# walk thru directory to find files
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pdf"):
            # ignore config folders
            if ".config" not in root:
                pdf_files.append(os.path.join(root, file))

if len(pdf_files) == 0:
    print("error: no pdfs found anywhere")
    exit()

print(f"found {len(pdf_files)} pdfs. loading them now...")

# load the files
documents = []
for file_path in tqdm(pdf_files, desc="reading files"):
    try:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    except:
        # skip if it fails
        continue

if not documents:
    print("error: found pdfs but couldnt read them")
    exit()

print(f"total pages: {len(documents)}")

print("spliting text into chunks...")
# keeping chunk size 1000 standard for rag
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"total chunks: {len(chunks)}")

# build vector db
print("building vector db with google api...")

# IMPORTANT: google api has rate limits so we cant send everything at once
# doing it in small batches so it doesnt crash
batch_size = 100
vectorstore = None

for i in tqdm(range(0, len(chunks), batch_size), desc="embedding batchs"):
    batch = chunks[i : i + batch_size]
    try:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        # sleep a sec to not hit rate limit
        time.sleep(1)
    except Exception as e:
        print(f"error on batch {i}: {e}")
        # wait a bit longer if it fails
        time.sleep(10)

# save and zip the result
if vectorstore is not None:
    print("saving index locally...")
    # naming it google so i know which one it is
    vectorstore.save_local("faiss_index_google")

    print("zipping it up...")
    !zip -r faiss_index_google.zip faiss_index_google

    print("done. download 'faiss_index_google.zip' from the side panel")
else:
    print("error: something went wrong creating vectorstore")
