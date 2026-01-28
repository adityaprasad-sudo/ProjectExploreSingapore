import os
import time
import random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.genai import types
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

#SETUP
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY") # Gemini Key
OR_API_KEY = os.getenv("OPENROUTER_API_KEY") # OpenRouter Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Groq Key

# Initialize Clients
OR_CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OR_API_KEY) if OR_API_KEY else None
GROQ_CLIENT = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow all origins for simplicity (or restrict to your github.io)

# This Block is used to setup the memoru(Vector data) which is then used for search
print("⏳ Initializing AI Memory...") # i will use for debugging

# this AI model will run local on the backend platform SO MAKE SURE THAT YOUR PLANTFORM HAS SUFFICIENT POWER AND RAM
# This must match the model used to build the vector data
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# B. Load the Pre-built Vector Database
# We use 'allow_dangerous_deserialization=True' because we created the vector data locally
DB_FOLDER = "faiss_index_minilm"
vectorstore = None

if os.path.exists(DB_FOLDER):
    try:
        vectorstore = FAISS.load_local(
            DB_FOLDER, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ Vector Database Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading Vector DB: {e}")
else:
    print(f"⚠️ Warning: '{DB_FOLDER}' folder not found. Semantic search will fail.")

# --- 2. SEARCH FUNCTION ---
def find_best_match(query):
    """
    Searches the vector database for the most relevant document.
    Returns: (content_text, score, source_name)
    """
    if vectorstore is None:
        return None, 0.0, "System Error"

    try:
        # Search for the top 1 closest chunk
        # Note: FAISS default metric is L2 Distance (Lower is better). 
        # A distance of 0 is a perfect match. A distance > 1.5 is usually irrelevant.
        results = vectorstore.similarity_search_with_score(query, k=1)
        
        if not results:
            return None, 0.0, None
            
        doc, score = results[0]
        
        # Convert L2 distance to a "Confidence Score" (approximate)
        # 0.0 -> 100% confidence, 1.0 -> ~50% confidence
        # We invert it so higher = better for your logic
        confidence = 1.0 / (1.0 + score) 
        
        return doc.page_content, confidence, doc.metadata.get('source', 'Unknown PDF')

    except Exception as e:
        print(f"⚠️ Search Error: {e}")
        return None, 0.0, None

# USE THIS TO GIVE INSTRUCTIONS TO THE AI
base_system_instruction = """
ROLE: You are a comprehensive Singapore Expert.

STRICT CONSTRAINTS:
1. ONLY answer questions related to Singapore. This includes laws, government, history, infrastructure (e.g., Changi Airport, architecture), tourism, and culture.
2. If the user asks anything unrelated to Singapore (e.g., other countries, general coding, or life advice), say: "I am only programmed to discuss Singapore."
3. MAXIMUM WORD LIMIT: 150 words. Be extremely concise.
4. FORMAT: Use a Source line, then 2-3 bullet points minimum.

SOURCE RULES:
- If context is provided, start with: "Source: [Document Name]"
"""

# --- 4. API ROUTE ---
@app.route('/ask', methods=['POST'])
def ask_gemini():
    data = request.get_json(force=True, silent=True) or {}
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"answer": "Please ask a question."})

    # --- STEP 1: RETRIEVAL ---
    context_text = ""
    source_name = "General Knowledge"
    
    # Perform Search
    found_text, score, src = find_best_match(user_query)
    
    # Threshold: If confidence > 0.4 (approx L2 distance < 1.5), we use it.
    if found_text and score >= 0.4:
        print(f"🔍 Match Found: {src} (Confidence: {score:.2f})")
        context_text = found_text
        source_name = src
    else:
        print(f"⚠️ Low Match (Confidence: {score:.2f}). Fallback to General Knowledge.")
        context_text = "No specific document found. Use general knowledge."

    # Construct Final Prompt
    final_prompt = f"CONTEXT (Source: {source_name}):\n{context_text}\n\nQUESTION: {user_query}"
    
    # MAIN LOGIC IF YOU GET THE CODE YOU CAN ADD AS MANY BACKUP AI AS YOU WANT
    
    # 1. Try Google Gemini
    if API_KEY:
        try:
            print("🚀 Attempting Primary (Gemini)...")
            client = genai.Client(api_key=API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Updated to latest fast model
                contents=[base_system_instruction, final_prompt],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=300)
            )
            return jsonify({"answer": response.text})
        except Exception as e:
            print(f"⚠️ Gemini Failed: {e}") #would probably fail if we hit the gemini rate limit 

    # 2. Try OpenRouter (Backup)(FREE)
    if OR_CLIENT:
        try:
            print("🔄 Attempting Backup (OpenRouter)...")
            response = OR_CLIENT.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free", # Fast & Free
                messages=[
                    {"role": "system", "content": base_system_instruction},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3,                   #controls Creativity of the AI
                max_tokens=300                      #controls Creativity of the AI
            )
            return jsonify({"answer": f"{response.choices[0].message.content}\n\n"})
        except Exception as e:
            print(f"⚠️ OpenRouter Failed: {e}")

    # 3. Try Groq (Last Resort)
    if GROQ_CLIENT:
        try:
            print("🔄 Attempting Tertiary (Groq)...")
            response = GROQ_CLIENT.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": base_system_instruction},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3,              #controls Creativity of the AI
                max_tokens=300                 #controls Creativity of the AI
            )
            return jsonify({"answer": f"{response.choices[0].message.content}\n\n"})
        except Exception as e:
            print(f"⚠️ Groq Failed: {e}")

    return jsonify({"answer": "System Overload. All AI models are currently busy. Please try again in 5 minute."}), 503

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=7860)
