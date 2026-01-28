#In This biuld vector we are gonna be using google embedder to process vector file and pdfs 
#⚠️Use this when you are using google embedder as your RAG Engine in the main app.py File
import os
import json
import time
import random
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 0. SETUP & KEYS ---
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")

if not API_KEY:
    print("❌ ERROR: GENAI_API_KEY not found in .env")
    exit()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "laws_db.json")
VECTOR_FILE = os.path.join(BASE_DIR, "laws_vectors.npy")

print("🤖 Connecting to Google AI (text-embedding-004)...")
embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=API_KEY
)


def get_embedding_with_retry(text, retries=5):
    """
    Handles Rate Limits (429) AND the Fake 'API Key Expired' (400) error.
    """
    for i in range(retries):
        try:
            return embedder.embed_query(text)
        except Exception as e:
            error_msg = str(e)
            
            # Catch 429 (Rate Limit) OR the glitchy 400 (Key Expired)
            if "429" in error_msg or "API key expired" in error_msg or "400" in error_msg:
                wait_time = (2 ** i) + random.uniform(1, 3) # Wait longer (3s, 5s, 9s...)
                print(f"   ⚠️ Glitch detected ({i+1}/{retries}). Pausing for {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # If it's a real error (like text too long), print it but return None
                print(f"   ❌ Fatal Error: {error_msg}")
                return None
    
    print("   ❌ Gave up after multiple retries.")
    return None

def build_vectors():
    # 1. Load Data
    if not os.path.exists(DB_FILE):
        print("❌ Error: laws_db.json not found.")
        return

    print("📚 Loading Law Text...")
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        law_data = json.load(f)

    # If you want to restart, just delete the old .npy file.
    
    embeddings = []
    print(f"⚡ Converting {len(law_data)} laws to vectors...")
    
    for index, entry in enumerate(law_data):
        # Text Chunking
        text_preview = entry['content'][:2000] 
        combined_text = f"Law Title: {entry['filename']}\nContent: {text_preview}"
        
        # Get Vector with the new "Smart Retry"
        vector = get_embedding_with_retry(combined_text)
        
        if vector:
            embeddings.append(vector)
            print(f"   ✅ Processed [{index+1}/{len(law_data)}]: {entry['filename']}")
        else:
            print(f"   ❌ FAILED [{index+1}/{len(law_data)}]: {entry['filename']}")
            # Important: This ensures that even when an error occurs, your data structure remains organized and aligned.
            embeddings.append([0.0] * 768)

        # Standard safety pause
        time.sleep(1.0)

    # 3. Save
    print(f"💾 Saving Vectors to {VECTOR_FILE}...")
    np.save(VECTOR_FILE, np.array(embeddings))
    print("✅ Vector Database Rebuilt Successfully!")

if __name__ == "__main__":

    build_vectors()
