"""
PDF Chatbot — Streamlit + LangChain + Google Gemini + FAISS
============================================================
Bot listesi bots.json'da saklanır.
Her botun FAISS indexi   ./faiss_indexes/<bot_id>/  klasöründe tutulur.
API anahtarı  st.secrets['GEMINI_API_KEY']  ile okunur.
"""

import os
import json
import uuid
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ─────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────
BOTS_FILE = Path("bots.json")
INDEX_DIR = Path("faiss_indexes")
INDEX_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Sayfa yapılandırması
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — premium dark-mode tasarımı
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    .app-header { text-align: center; padding: 2rem 0 1rem; }
    .app-header h1 {
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 700; margin: 0;
    }
    .app-header p { color: #8b949e; font-size: 0.95rem; margin-top: 0.4rem; }

    .sidebar-title {
        font-size: 1.1rem; font-weight: 700; color: #a78bfa;
        margin-bottom: 1rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid #30363d;
    }

    .chat-bubble-user { display:flex; justify-content:flex-end; margin:0.6rem 0; }
    .chat-bubble-user .bubble {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: #fff; border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1.1rem; max-width: 72%;
        font-size: 0.9rem; line-height: 1.5;
        box-shadow: 0 4px 12px rgba(124,58,237,0.3);
    }
    .chat-bubble-bot { display:flex; justify-content:flex-start; margin:0.6rem 0; }
    .chat-bubble-bot .bubble {
        background: linear-gradient(135deg, #1c2333, #21262d);
        color: #e6edf3; border-radius: 18px 18px 18px 4px;
        padding: 0.7rem 1.1rem; max-width: 72%;
        font-size: 0.9rem; line-height: 1.5;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .avatar {
        width:32px; height:32px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:1rem; flex-shrink:0;
    }
    .avatar-bot { background: linear-gradient(135deg,#34d399,#059669); margin-right:8px; }
    .avatar-user { background: linear-gradient(135deg,#a78bfa,#7c3aed); margin-left:8px; }

    .info-box {
        background: linear-gradient(135deg, #0d2137, #0d3251);
        border: 1px solid #1f6feb; border-radius: 12px;
        padding: 1.2rem 1.5rem; color: #58a6ff;
        font-size: 0.9rem; margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #0d2b1d, #063a20);
        border: 1px solid #238636; border-radius: 12px;
        padding: 1rem 1.4rem; color: #3fb950;
        font-size: 0.88rem; margin: 0.8rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #2b1d0d, #3a2006);
        border: 1px solid #9e6a03; border-radius: 12px;
        padding: 1rem 1.4rem; color: #d29922;
        font-size: 0.88rem; margin: 0.8rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important; border-radius: 10px !important;
    }

    .chat-container {
        max-height: 500px; overflow-y: auto;
        padding: 1rem;
        background: rgba(13,17,23,0.5);
        border-radius: 16px; border: 1px solid #21262d;
        margin-bottom: 1rem;
    }
    .chat-container::-webkit-scrollbar { width: 6px; }
    .chat-container::-webkit-scrollbar-track { background: #0d1117; border-radius: 3px; }
    .chat-container::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    hr { border-color: #30363d !important; }
    label { color: #8b949e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────

def load_bots():
    if BOTS_FILE.exists():
        with open(BOTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_bots(bots):
    with open(BOTS_FILE, "w", encoding="utf-8") as f:
        json.dump(bots, f, ensure_ascii=False, indent=2)


def get_bot(bot_id):
    return next((b for b in load_bots() if b["id"] == bot_id), None)


def extract_text_from_pdf(pdf_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp.close()
    reader = PdfReader(tmp.name)
    os.unlink(tmp.name)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def build_faiss_index(text, bot_id, api_key):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key,
    )
    store = FAISS.from_texts(chunks, embeddings)
    index_path = INDEX_DIR / bot_id
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))


@st.cache_resource(show_spinner=False)
def load_faiss_index(bot_id, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key,
    )
    return FAISS.load_local(
        str(INDEX_DIR / bot_id), embeddings,
        allow_dangerous_deserialization=True,
    )


def get_conversation_chain(vector_store, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4},
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        return_source_documents=False, verbose=False,
    )

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

for k, v in {
    "selected_bot_id": None,
    "chat_histories": {},
    "chains": {},
    "creating_bot": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# API anahtarı
# ─────────────────────────────────────────────

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = None

# ─────────────────────────────────────────────
# KENAR ÇUBUĞU
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 PDF Botlarım</div>', unsafe_allow_html=True)

    if st.button("➕  Yeni Bot Oluştur", use_container_width=True):
        st.session_state.creating_bot = True

    st.markdown("---")

    bots = load_bots()
    if not bots:
        st.markdown(
            '<div class="info-box">Henüz bot yok.<br>➕ ile ilk botunu oluştur!</div>',
            unsafe_allow_html=True,
        )
    else:
        for bot in bots:
            is_active = st.session_state.selected_bot_id == bot["id"]
            icon = "🟣" if is_active else "⚪"
            if st.button(
                f"{icon} **{bot['name']}**",
                key=f"sel_{bot['id']}",
                use_container_width=True,
            ):
                st.session_state.selected_bot_id = bot["id"]
                st.session_state.creating_bot = False

    st.markdown("---")

    if st.session_state.selected_bot_id:
        if st.button("🗑️  Seçili Botu Sil", use_container_width=True):
            bid = st.session_state.selected_bot_id
            bots = [b for b in load_bots() if b["id"] != bid]
            save_bots(bots)
            idx_path = INDEX_DIR / bid
            if idx_path.exists():
                shutil.rmtree(idx_path)
            st.session_state.chains.pop(bid, None)
            st.session_state.chat_histories.pop(bid, None)
            st.session_state.selected_bot_id = None
            st.cache_resource.clear()
            st.rerun()

# ─────────────────────────────────────────────
# ANA EKRAN
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>📄 PDF Chatbot</h1>
        <p>PDF belgelerinizi yükleyin, yapay zeka ile sohbet edin.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not API_KEY:
    st.markdown(
        """
        <div class="warning-box">
        ⚠️ <strong>GEMINI_API_KEY bulunamadı!</strong><br>
        <code>.streamlit/secrets.toml</code> dosyasına ekleyin:<br>
        <code>GEMINI_API_KEY = "AIza..."</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Yeni bot oluşturma formu ──
if st.session_state.creating_bot:
    st.markdown("### ➕ Yeni Bot Oluştur")
    col1, col2 = st.columns([2, 1])
    with col1:
        bot_name = st.text_input("Bot Adı *", placeholder="Örn: Hukuk Asistanı")
    with col2:
        bot_desc = st.text_input("Açıklama (isteğe bağlı)", placeholder="Örn: Sözleşme analizi")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Oluştur", use_container_width=True):
            if not bot_name.strip():
                st.error("Bot adı boş olamaz!")
            else:
                new_bot = {
                    "id": str(uuid.uuid4()),
                    "name": bot_name.strip(),
                    "description": bot_desc.strip(),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "has_index": False,
                    "pdf_name": "",
                }
                bots = load_bots()
                bots.append(new_bot)
                save_bots(bots)
                st.session_state.selected_bot_id = new_bot["id"]
                st.session_state.creating_bot = False
                st.rerun()
    with c2:
        if st.button("❌ İptal", use_container_width=True):
            st.session_state.creating_bot = False
            st.rerun()
    st.stop()

# ── Bot seçilmemişse ──
if not st.session_state.selected_bot_id:
    st.markdown(
        """
        <div class="info-box">
        👈 <strong>Sol menüden bir bot seç</strong> ya da yeni bir bot oluştur.<br><br>
        <b>Nasıl çalışır?</b><br>
        1️⃣ Sol menüden <em>➕ Yeni Bot Oluştur</em> butonuna tıkla<br>
        2️⃣ Bot adını gir ve oluştur<br>
        3️⃣ PDF dosyanı yükle — bot indexini otomatik oluşturur<br>
        4️⃣ Soru sor, sadece o PDF'e göre cevap alırsın!
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Seçili bot ──
bot = get_bot(st.session_state.selected_bot_id)
if not bot:
    st.error("Bot bulunamadı.")
    st.session_state.selected_bot_id = None
    st.stop()

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">
        <div style="width:48px;height:48px;border-radius:50%;
                    background:linear-gradient(135deg,#a78bfa,#7c3aed);
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;">🤖</div>
        <div>
            <div style="font-size:1.4rem;font-weight:700;color:#e6edf3;">{bot['name']}</div>
            <div style="font-size:0.85rem;color:#8b949e;">{bot.get('description','') or 'PDF destekli sohbet botu'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── PDF Yükleme ──
index_path = INDEX_DIR / bot["id"]
has_index = index_path.exists() and any(index_path.iterdir())

with st.expander("📤 PDF Yükle / Güncelle", expanded=not has_index):
    uploaded_file = st.file_uploader(
        "PDF dosyasını sürükle ya da seç", type=["pdf"], key=f"pdf_{bot['id']}",
    )

    if uploaded_file:
        st.markdown(
            f'<div class="success-box">📎 <strong>{uploaded_file.name}</strong> '
            f'({uploaded_file.size/1024:.1f} KB)</div>',
            unsafe_allow_html=True,
        )
        if st.button("⚡ FAISS Index Oluştur", use_container_width=True, key="build_index"):
            progress = st.progress(0, text="PDF okunuyor...")
            try:
                text = extract_text_from_pdf(uploaded_file.read())
                if not text.strip():
                    st.error("PDF'den metin çıkarılamadı (görüntü tabanlı PDF olabilir).")
                    st.stop()
                progress.progress(40, text="Metin parçalara bölünüyor ve gömülüyor...")
                build_faiss_index(text, bot["id"], API_KEY)
                progress.progress(90, text="Kaydediliyor...")
                bots = load_bots()
                for b in bots:
                    if b["id"] == bot["id"]:
                        b["has_index"] = True
                        b["pdf_name"] = uploaded_file.name
                save_bots(bots)
                st.session_state.chains.pop(bot["id"], None)
                st.cache_resource.clear()
                progress.progress(100, text="Tamamlandı!")
                st.success(f"✅ Index oluşturuldu: {uploaded_file.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Hata: {exc}")

    elif has_index and bot.get("pdf_name"):
        st.markdown(
            f'<div class="success-box">✅ Aktif PDF: <strong>{bot["pdf_name"]}</strong>'
            f'<br>Yeni PDF yükleyerek güncelleyebilirsiniz.</div>',
            unsafe_allow_html=True,
        )

if not has_index:
    st.markdown(
        '<div class="info-box">⬆️ Sohbete başlamak için önce bir PDF yükleyin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────
# Sohbet
# ─────────────────────────────────────────────

st.markdown("### 💬 Sohbet")

bid = bot["id"]
if bid not in st.session_state.chat_histories:
    st.session_state.chat_histories[bid] = []

if bid not in st.session_state.chains:
    with st.spinner("Vektör veritabanı ve model yükleniyor..."):
        try:
            vs = load_faiss_index(bid, API_KEY)
            st.session_state.chains[bid] = get_conversation_chain(vs, API_KEY)
        except Exception as exc:
            st.error(f"Index yüklenemedi: {exc}")
            st.stop()

chain = st.session_state.chains[bid]
history = st.session_state.chat_histories[bid]

# Mesajları göster
if history:
    parts = []
    for msg in history:
        if msg["role"] == "user":
            parts.append(
                f'<div class="chat-bubble-user">'
                f'<div class="bubble">{msg["content"]}</div>'
                f'<div class="avatar avatar-user">👤</div>'
                f'</div>'
            )
        else:
            parts.append(
                f'<div class="chat-bubble-bot">'
                f'<div class="avatar avatar-bot">🤖</div>'
                f'<div class="bubble">{msg["content"]}</div>'
                f'</div>'
            )
    st.markdown(
        f'<div class="chat-container">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="info-box" style="text-align:center;">💡 PDF hakkında ilk sorunuzu yazın!</div>',
        unsafe_allow_html=True,
    )

# Soru formu
with st.form(key=f"chat_form_{bid}", clear_on_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_question = st.text_input(
            "Sorunuzu yazın",
            placeholder="Bu PDF'te hangi konular işleniyor?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Gönder ➤", use_container_width=True)

col_clr, _ = st.columns([1, 5])
with col_clr:
    if st.button("🗑️ Sohbeti Temizle", key=f"clear_{bid}"):
        st.session_state.chat_histories[bid] = []
        st.session_state.chains.pop(bid, None)
        st.cache_resource.clear()
        st.rerun()

if submitted and user_question.strip():
    with st.spinner("Düşünüyor..."):
        try:
            result = chain({"question": user_question})
            answer = result.get("answer", "Cevap üretilemedi.")
        except Exception as exc:
            answer = f"⚠️ Hata: {exc}"
    history.append({"role": "user", "content": user_question})
    history.append({"role": "bot", "content": answer})
    st.session_state.chat_histories[bid] = history
    st.rerun()
