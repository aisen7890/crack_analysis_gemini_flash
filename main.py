# ------------------------------------------
# Monkey-patch sqlite3 to use pysqlite3-binary
# ------------------------------------------
try:
    import pysqlite3 as sqlite3_alt
    import sys
    # Replace the built-in sqlite3 with pysqlite3
    sys.modules['sqlite3'] = sqlite3_alt
except ImportError:
    # If pysqlite3-binary isn't installed, fall back to system sqlite3
    pass

import streamlit as st
import google.generativeai as old_genai
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_text
import stat
import shutil
import tempfile
import gc
import re
from gtts import gTTS

import whisper
import uuid

# from google import genai as new_genai
# import asyncio

from langdetect import detect

# Configure the Gemini API
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()   # local only
    import os
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    st.stop()



old_genai.configure(api_key=api_key)

# Initialize the model
model = old_genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

# Set page config
st.set_page_config(
    page_title="Analyze Concrete Crack using AI",
    page_icon="🤖",
    layout="wide"
)

CONVERSATION_HISTORY_PATH = Path("./conversation_history.json")

# ChromaDB and RAG functions
client = chromadb.Client()

REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(exist_ok=True)


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_client():
    return client


def initialize_db():
    return get_client().get_or_create_collection(
        name="image_knowledge",
        embedding_function=ef,
    )

def extract_text_from_pdf(pdf_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    text = extract_text(tmp_path)
    os.unlink(tmp_path)
    return text

def split_into_chunks(text: str, chunk_size: int = 600):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def add_pdf_to_db(pdf_file):
    col = initialize_db()
    text = extract_text_from_pdf(pdf_file)
    for i, chunk in enumerate(split_into_chunks(text)):
        col.add(
            documents=[chunk],
            ids=[f"doc_{col.count()+1}"],
            metadatas=[{"source": pdf_file.name, "chunk": i}],
        )

def get_relevant_knowledge(query: str, k: int = 5) -> str:
    col = initialize_db()
    res = col.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""

def load_conversation_history():
    if CONVERSATION_HISTORY_PATH.exists():
        try:
            return json.loads(CONVERSATION_HISTORY_PATH.read_text(encoding='utf-8'))
        except Exception as e:
            st.error(f"Error loading conversation history: {e}")
    return []

def save_conversation_history(history):
    try:
        CONVERSATION_HISTORY_PATH.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    except Exception as e:
        st.error(f"Error saving conversation history: {e}")

def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_conversation_history()

# At the top, after session state initialization
if 'save_to_reports' not in st.session_state:
    st.session_state.save_to_reports = False


def get_gemini_response(prompt, image=None):
    try:
        if image:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create the content parts
            content_parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode()}}
            ]
        else:
            content_parts = [{"text": prompt}]

        # Generate response
        response = model.generate_content(
            contents=content_parts,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        return response.text
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def autoplay_audio(audio_file_path):
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    md = f'''
    <audio autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    '''
    st.markdown(md, unsafe_allow_html=True)

# App title and description
st.title("🤖 Analyze Concrete Crack using AI")
st.markdown("""
This app allows you to chat with Gemini Flash model using both text and images.
Upload an image or type your message to start the conversation!
""")

# --- Image input (upload or capture) ---
st.header("Image Input")
col1, col2 = st.columns(2)

if "recorder_key" not in st.session_state:
    st.session_state.recorder_key = str(uuid.uuid4())

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        st.session_state["pending_image"] = uploaded_image
        st.image(uploaded_image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)  
        typed_prompt=''
        audio_bytes = None
        user_prompt = None
        st.session_state["last_voice_input"] = None
        
        #if pending image true...
with col2:
    camera_file = st.camera_input("Capture an image")
    if camera_file:
        camera_image = Image.open(camera_file)
        st.session_state["pending_image"] = camera_image
        typed_prompt=''
        audio_bytes = None
        user_prompt = None
        st.session_state["last_voice_input"] = None
        
    # Do not display the captured image preview here to keep UI clean

# Use the pending image if present
image = st.session_state.get("pending_image", None)

# --- Chat input and audio recorder side by side ---
st.header("Chat")
col_chat, col_audio = st.columns([3, 1])
with col_chat:
    typed_prompt = st.chat_input("Enter text here")
with col_audio:
    # Try the built-in widget first:
    if hasattr(st, "audio_input"):
        audio_bytes = st.audio_input("🎤 Record your voice", label_visibility="collapsed", key=st.session_state.recorder_key)

        
        # `.audio_input` returns a BytesIO-like object
        if audio_bytes is not None:
            audio_bytes = audio_bytes.read()

    else:
        st.warning("Audio input is not available in this environment.")
        audio_bytes = None
    
    
# If audio is recorded, transcribe and use as prompt; else use typed prompt
user_prompt = None

if audio_bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        f.close()
        audio_path = f.name
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(audio_path)
    transcript = result.get("text", "")
    st.session_state["last_voice_input"] = transcript

    #erase audio data. 
    if os.path.exists(audio_path):
        os.remove(audio_path)
    audio_bytes = None






else:
    st.session_state["last_voice_input"] = None

# Use the most recent input: prefer typed text if present, else voice
if typed_prompt:
    user_prompt = typed_prompt
    audio_bytes = None
    typed_prompt=''
elif st.session_state.get("last_voice_input"):
    typed_prompt=''
    user_prompt = st.session_state["last_voice_input"]
else:
    typed_prompt=''
    audio_bytes = None
    user_prompt = None

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "image" in message and message["image"] is not None:
            st.image(message["image"], caption="Uploaded Image", width=300)

# --- Sidebar: Knowledge Base Management ---
st.sidebar.title("📚 Knowledge Base")
st.sidebar.header("Upload & Stats")
pdf_file = st.sidebar.file_uploader("Upload PDF to add", type=["pdf"])
if st.sidebar.button("Add PDF"):
    if pdf_file:
        add_pdf_to_db(pdf_file)
        st.sidebar.success(f"PDF '{pdf_file.name}' added.")
    else:
        st.sidebar.warning("Please select a PDF first.")

col = initialize_db()
count = col.count()
ct = st.sidebar.info(f"Total docs in DB: {count}")

# Clear DB with confirmation
st.sidebar.markdown("---")
if st.sidebar.button("Clear DB", type="secondary"):
    col = initialize_db()
    ids = col.get(ids=None)["ids"]
    if ids:
        col.delete(ids=ids)
    st.sidebar.success("Cleared")
    st.rerun()








# --- Sidebar: Saved Reports Management ---
st.sidebar.markdown("---")
st.sidebar.header("📂 Saved Reports")

# reports 폴더 파일 리스트
report_files = sorted(REPORTS_DIR.iterdir())
if report_files:
    for file in report_files:
        with open(file, "rb") as f:
            st.sidebar.download_button(
                label=f"Download {file.name}",
                data=f,
                file_name=file.name,
                mime="application/octet-stream",
            )
else:
    st.sidebar.info("No reports available.")


# --- Sidebar: Clear Saved Reports ---
if st.sidebar.button("Clear All Reports", type="secondary"):
    # Delete every file in the reports folder
    report_files = list(REPORTS_DIR.iterdir())
    if report_files:
        for f in report_files:
            try:
                f.unlink()
            except Exception as e:
                st.sidebar.error(f"Failed to delete {f.name}: {e}")
        st.sidebar.success("All report files have been deleted.")
    else:
        st.sidebar.info("No reports to delete.")
    # Refresh the app so download list disappears immediately
    st.rerun()






















# --- Modify chat input logic to use RAG ---
if user_prompt:
    # If user asks to save to reports, set the flag
    if re.search(r"save to reports", user_prompt, re.IGNORECASE):
        st.session_state.save_to_reports = True
    elif re.search(r"save to report", user_prompt, re.IGNORECASE):
        st.session_state.save_to_reports = True
    elif re.search(r"저장해", user_prompt, re.IGNORECASE):
        st.session_state.save_to_reports = True

    # Retrieve relevant knowledge
    context = get_relevant_knowledge(user_prompt, k=5)

    with open('compliance.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()


    # Prepend context to prompt
    full_prompt = f"""
    {system_prompt}

    배경 지식 및 관련규정:
    {context}
    
    사용자 입력 메세지:
    {user_prompt}
    """ 
    
    # Add user message to chat history
    msg = {
        "role": "user",
        "content": user_prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.chat_history.append(msg)
    save_conversation_history(st.session_state.chat_history)
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_prompt)
        if image:
            if uploaded_file and image == st.session_state.get("pending_image"):
                st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            elif camera_file and image == st.session_state.get("pending_image"):
                st.image(image, caption="Captured Image", use_container_width=True)
            else:
                st.image(image, use_container_width=True)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_gemini_response(full_prompt, image)
            st.session_state.response = response


        
            if response:
                st.write(response)
                # Add assistant response to chat history
                msg = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.chat_history.append(msg)
                save_conversation_history(st.session_state.chat_history)
                # If save_to_reports flag is set, save this response, image, and audio
                if st.session_state.save_to_reports:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"report_{timestamp}"
                    # Save image if present
                    if image:
                        img_path = REPORTS_DIR / f"{base_filename}.png"
                        image.save(img_path)
                    # Save assistant response
                    txt_path = REPORTS_DIR / f"{base_filename}.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(response)
                    st.session_state.save_to_reports = False
    # After sending, clear the pending image and user input so they are not sent again
    st.session_state["pending_image"] = None
    st.session_state["last_voice_input"] = None

    st.session_state.recorder_key = str(uuid.uuid4())
    st.rerun()
    
# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    if CONVERSATION_HISTORY_PATH.exists():
        CONVERSATION_HISTORY_PATH.unlink()
    st.rerun()




try:
    txt =st.session_state.response
    lang = detect(txt)
    tts = gTTS(text=txt, lang=lang)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        tts.save(tmp_audio.name)
        audio_path = tmp_audio.name
        autoplay_audio(audio_path)
    st.session_state.response = None
except:
    print()



st.session_state["pending_image"] = None






