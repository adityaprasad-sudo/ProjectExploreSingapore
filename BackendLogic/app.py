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

# Initializing Clients
OR_CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OR_API_KEY) if OR_API_KEY else None
GROQ_CLIENT = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow CORS for all routes and to prevent browser from blocking our api
# This Block is used to setup the memoru(Vector data) which is then used for search
print("Initializing AI Memory...") # i will use this for debugging

# this AI model will run local on the backend platform SO MAKE SURE THAT YOUR PLANTFORM HAS SUFFICIENT POWER AND RAM
# This must match the model used to build the vector data
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# B. Load the Pre-built Vector Database
# We use 'allow_dangerous_deserialization=True' because we created the vector data locally
vector_FOLDER = "faiss_index_minilm"
vectorstore = None

if os.path.exists(vector_FOLDER):
    try:
        vectorstore = FAISS.load_local(
            vector_FOLDER, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector Database Loaded")
    except Exception as e:
        print(f"Error while loading vector DB: {e}")
else:
    print(f"The: '{vector_FOLDER}' folder not found. semantic search will fail.")

# search function to find the best matching document from the vector database
def findbestmatch(query):
    """
    Searches the vector database for the most relevant document.(If any)
    Returns: (contenttext, score, sourcename)
    """
    if vectorstore is None:
        return None, 0.0, "System Error"        # most probable cause is the vector DB failed to load or maybe the vector database is corrupted

    try:
        # Search for the top 1 closest chunk ie most relevant
        # Note: FAISS(vector data bae file format) default metric is L2 Distance (Lower is better)
        # A distance of 0 is a perfect match. A distance > 1.5 is usually irrelevant.(this is the distance of vector from the use query to rvevant document)
        results = vectorstore.similarity_search_with_score(query, k=1)
        
        if not results:
            return None, 0.0, None
            
        doc, score = results[0]
        
        # Convert L2 distance to a "Confidence Score" (approximate)
        # if score is 0 then the confidence would be 1 ie 100% confidence and if score is 1 then confidence would be 1/2 ie 50%
        # basically score is the actual distance of the vector from the revlevant vector vector and confidence i just a way to represent how sure the ai is based on the distance of vector(score)
        confidence = 1.0 / (1.0 + score) 
        
        return doc.page_content, confidence, doc.metadata.get('source', 'Unknown PDF')

    except Exception as e:
        print(f"Search Error: {e}")           #usally happens if the server ram is overused which doesnt allow our local ai model alllmminil6v2 to work
        return None, 0.0, None

# USE THIS TO GIVE INSTRUCTIONS TO THE AI
instructions = """
ROLE: You are a comprehensive Singapore Expert.

STRICT CONSTRAINTS:
1. ONLY answer questions related to Singapore. This includes laws, government, history, infrastructure (e.g., Changi Airport, architecture), tourism, and culture.
2. If the user asks anything unrelated to Singapore (e.g., other countries, general coding, or life advice), say: "I am only programmed to discuss Singapore."
3. MAXIMUM WORD LIMIT: 150 words. Be extremely concise.
4. FORMAT: Use a Source line, then 2-3 bullet points minimum.

SOURCE RULES:
- If context is provided, start with: "Source: [Document Name]"
"""

# Api route
@app.route('/ask', methods=['POST']) #/ask allows us to ive the prompt to the Chat generating AI
def ask_gemini():
    data = request.get_json(force=True, silent=True) or {}
    user_query = data.get('query', '')

    if not user_query: #if enter is pressed without giving any prompt
        return jsonify({"answer": "Please ask a question."})

    # retriving the info our searching model gave us
    contexttext = ""
    sourcename = "General Knowledge"

    # this performs the searcg based on use query(uer prompt)
    found_text, score, src = findbestmatch(user_query)
    
    # if the searched content is having confidence greater than 0.4 use it.
    if found_text and score >= 0.4:
        print(f"Match Found: {src} (Confidence: {score:.2f})")
        contexttext = found_text
        sourcename = src
    else:
        print(f" Low Match (Confidence: {score:.2f}). Fallback to General Knowledge.")
        context_text = "No specific document found. Use general knowledge." # give the chat generation that no relevent document is found matching the user's query

    # Constructing the final prompt for aaour chat generation ai
    finalprompt = f"CONTEXT (Source: {sourcename}):\n{contexttext}\n\nQUESTION: {user_query}"
    
    # MAIN LOGIC ADD AS MANY BACKUP AI AS YOU WANT
    
    #first lets try Google Gemini
    if API_KEY:
        try:
            print("🚀 Attempting Primary (Gemini)...")
            client = genai.Client(api_key=API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Using the latest available model
                contents=[instructions, finalprompt],#feeding the ai the final prompt and system instructions
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=300)
            )
            return jsonify({"answer": response.text})
        except Exception as e:
            print(f"Gemini Failed: {e}") #would probably fail if we hit the gemini rate limit 

    # 2. Try OpenRouter (Backup)(FREE)
    if OR_CLIENT:
        try:
            print("🔄 Attempting Backup (OpenRouter)...")
            response = OR_CLIENT.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free", # Fast & Free
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": finalprompt}
                ],
                temperature=0.3,                   #controls Creativity of the AI
                max_tokens=300                      #the max words the ai can print
            )
            return jsonify({"answer": f"{response.choices[0].message.content}\n\n"})
        except Exception as e:
            print(f"OpenRouter Failed: {e}") #would fail if the models is down

    # 3. Try Groq (Last Resort)
    if GROQ_CLIENT:
        try:
            print("🔄 Attempting Tertiary (Groq)...")
            response = GROQ_CLIENT.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": finalprompt}
                ],
                temperature=0.3,          
                max_tokens=300                 
            )
            return jsonify({"answer": f"{response.choices[0].message.content}\n\n"})
        except Exception as e:
            print(f"Groq Failed: {e}")

    return jsonify({"answer": "No available models to respond"}), 503

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=7860)
