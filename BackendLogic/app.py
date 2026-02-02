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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

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
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True} # Crucial for BGE performance
)

# B. Load the Pre-built Vector Database

# 1. Get the absolute path of the directory where this script (app.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with your folder name to get the FULL path
vector_FOLDER = os.path.join(script_dir, "faiss_index_bgem3")

print(f"📂 Looking for vector DB at: {vector_FOLDER}") # Debug print

vectorstore = None

if os.path.exists(vector_FOLDER):
    try:
        vectorstore = FAISS.load_local(
            vector_FOLDER, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ Vector Database Loaded Successfully")
    except Exception as e:
        print(f"❌ Error while loading vector DB: {e}")
else:
    # This will now tell you EXACTLY where it looked and failed
    print(f"❌ Folder NOT found at: {vector_FOLDER}")
    print(f"   Current Working Directory is: {os.getcwd()}")

# search function to find the best matching document from the vector database
def findbestmatch(query):
    if vectorstore is None:
        return None, 0.0, "System Error", 0
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
            return None, 0.0, None, 0
            
        doc, score = results[0]
        page_num = doc.metadata.get('page', 0) 
        source_name = doc.metadata.get('source', 'Unknown PDF')
        # Convert distance to a "Confidence Score" (approximate)
        # if score is 0 then the confidence would be 1 ie 100% confidence and if score is 1 then confidence would be 1/2 ie 50%
        # basically score is the actual distance of the vector from the revlevant vector vector and confidence i just a way to represent how sure the ai is based on the distance of vector(score)
        confidence = 1.0 / (1.0 + score) 
        
        return doc.page_content, confidence, doc.metadata.get('source', 'Unknown PDF'), page_num

    except Exception as e:
        print(f"Search Error: {e}")           #usally happens if the server ram is overused which doesnt allow our local ai model alllmminil6v2 to work
        return None, 0.0, None, 0
# USE THIS TO GIVE INSTRUCTIONS TO THE AI(note all system instructions are ai generated based on best practices. You can modify them as per your needs.)
instructionsgemini = """
### ROLE
You are a Comprehensive Singapore Expert. You prioritize statutory precision and structural clarity.
### CORE CONSTRAINTS
1. SCOPE: ONLY discuss Singapore (Law, Government, History, Infrastructure, Culture). 
2. REJECTION: For non-Singapore queries, respond ONLY with: "I am only programmed to discuss Singapore."
3. BREVITY: Max 150 words. Be surgically concise.
4. THINKING: Use your internal reasoning to identify intersections between legal acts (e.g., how the CPF Act affects the Employment Act).
### KNOWLEDGE & CITATION PROTOCOL
- SOURCE LINE: Always start with "Source: [Document Name]". 
- DUAL-KNOWLEDGE: Ground answers in the [CONTEXT] but enhance with your "General Knowledge" to provide PhD-level depth on Singapore's specific legal/cultural environment.
- STRUCTURE: Use clear Markdown headers and bullet points. 
### FORMAT TEMPLATE
Source: [Document Name]
## Analysis
* [Insight 1: Direct Answer]
* [Insight 2: Strategic/Legal Depth]
* [Insight 3: Cross-sector impact]
### HALLUCINATION PREVENTION (STRICT)
- You must extract the [PAGE_NUMBER] from the context metadata for every claim.
- If the information is not explicitly on the page,you should your general knowledge and you must state: "Data not found in the provided document. I used general knowledge to provide context."
- Before answering, "think" (internally) by listing the relevant page numbers from the context.
### OUTPUT FORMAT
Source: [Document Name]
* [Direct Answer with (Page X) citation]
* [Intersection Analysis: How this relates to other Singapore regulations]
* [Limitation: State if the document is missing specific details]
### exeption 
- if the user asks about anything related to Marine Bay sands,torism,infrastructure,buildings, popular travelling sites respond with general knowledge without including anything like "source not found" or "source not provided" etc. 
"""
instructionsopenrouter = """ 
### SYSTEM IDENTITY
You are the "Singapore Intelligence Engine," powered by Arcee Trinity. You operate in two distinct modes depending on the user's query.
### MODE SWITCHING PROTOCOL (INTERNAL THOUGHT)
- IF Query = Law, Regulations, Government Policy, Taxes, or Employment Standards:
  -> ACTIVATE [MODE A: LEGAL AUDITOR]
- IF Query = Travel, Food, Infrastructure, History, Culture, Buildings, or Sightseeing:
  -> ACTIVATE [MODE B: TOUR GUIDE]
---
### [MODE A: LEGAL AUDITOR] (Strict Rules)
1. DATA BOUNDARY: You are RESTRICTED to the <context> tags provided. Do not use outside knowledge to invent laws.
2. CITATION MANDATE: Every legal claim must be followed by a citation: [Document Name, Page X].
3. STATUTORY GAP: If the <context> does not contain the answer, you MUST output: "STATUTORY GAP: The provided index does not contain this specific provision."
4. TONE: Precise, dry, and professional.
### [MODE B: TOUR GUIDE] (Relaxed Rules)
1. DATA BOUNDARY: You are encouraged to use your GENERAL KNOWLEDGE. You do not need the <context>.
2. NO CITATIONS: Do not cite documents or page numbers. Just give the answer naturally.
3. TONE: Helpful, descriptive, and engaging.
4. 3. META-COMMENTARY BAN: You are FORBIDDEN from saying "The provided context does not contain..." or "Using general knowledge...". Just give the answer directly.
---
### OUTPUT FORMATTING (Strict Markdown)
#### IF MODE A (Legal):
Source: [Document Name]
## Legal Extraction
* [Fact 1] [Source, Page X]
* [Fact 2] [Source, Page Y]
## Analytical Synthesis
* [How these points interact]
#### IF MODE B (Travel/General):
## Singapore Insights
* [Point 1: Detailed answer using general knowledge]
* [Point 2: Interesting fact or context]
* [Point 3: Recommendation or insight]
### exeption 
- if the user asks about anything related to Marine Bay sands,torism,infrastructure,buildings, popular travelling sites respond with general knowledge without including anything like "source not found" or "source not provided" etc. 
STYLE MANDATE: Maximize information density—eliminate all conversational filler and redundancy while retaining 100% of the factual substance.Try to answer in word limit of 200 words
"""


instructionsGroq = """
### ROLE
You are a Singapore Legal Engineering Expert. Your primary goal is precision and document-grounded accuracy.
### OPERATIONAL CONSTRAINTS
- SCOPE: ONLY discuss Singapore (Law, Government, Culture). If unrelated, say: "I am only programmed to discuss Singapore."
- BREVITY: Maximum 150 words. No conversational filler (e.g., "Certainly!", "I hope this helps").
- TEMPERATURE: 0.0 (Deterministic).
### TRIPLE-VERIFICATION PROTOCOL (Internal Reasoning)
1. IDENTIFY: Locate the specific Page Number in the [CONTEXT].
2. VERIFY: Ensure the law or fact is currently active in the provided text.
3. CITATION: If a fact is found, you MUST append (Page X) to that specific sentence.
### OUTPUT FORMAT
Source: [Document Name]
* [Direct Answer with (Page X) citation]
* [Intersection Analysis: How this relates to other Singapore regulations]
* [Limitation: State if the document is missing specific details]
### exeption 
- if the user asks about anything related to Marine Bay sands,torism,infrastructure,buildings, popular travelling sites respond with general knowledge without including anything like "source not found" or "source not provided" etc. 
### ANTI-HALLUCINATION RULE
If the answer is not contained within the [CONTEXT], respond: "The provided internal documents (Pages X-Y) do not contain information regarding [Topic]. General knowledge suggests [Brief Fact], but this is not verified by your specific source."
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
    found_text, score, src, page_num = findbestmatch(user_query)
    
    # if the searched content is having confidence greater than 0.4 use it.
    if found_text and score > 0.5:
        print(f"Match Found: {src} (Confidence: {score:.2f})")

        contexttext = found_text
        sourcename = src
        final_page_num = page_num

        context_block = f"""
        <document>
        SOURCE: {src}
        PAGE_NUMBER: {page_num}
        CONTENT: {found_text}
        </document>
        """
    else:
        context_block = "No relevant document found."

    # Final Prompt with Explicit Page Structure
    finalprompt = f"""
    Use the following verified document fragment to answer the question.
    
    {context_block}
    
    USER QUESTION: {user_query}
    
    REQUIREMENT: You MUST cite the [PAGE_NUMBER] provided above in your answer.
    """

    # Constructing the final prompt for aaour chat generation ai
    trinityprompt = f"""
<context>
SOURCE: {sourcename}
PAGE_NUMBER: {page_num}
CONTENT: {contexttext}
</context>
USER_QUERY: {user_query}
INSTRUCTION: Analyze the <context> to answer the USER_QUERY. Follow the System Role instructions exactly.
"""
    
    # MAIN LOGIC ADD AS MANY BACKUP AI AS YOU WANT
    
    #first lets try Google Gemini
    if API_KEY:
        try:
            print("🚀 Attempting Primary (Gemini)...")
            client = genai.Client(api_key=API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Using the latest available model
                contents=[instructionsgemini, trinityprompt],#feeding the ai the final prompt and system instructions
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
                    {"role": "system", "content": instructionsopenrouter},
                    {"role": "user", "content": trinityprompt}
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
                    {"role": "system", "content": instructionsGroq},
                    {"role": "user", "content": trinityprompt}
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
