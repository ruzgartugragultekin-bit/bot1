"""
PDF Chatbot — Streamlit + LangChain + HuggingFace Embeddings + Google Gemini
=============================================================================
Özellikler:
  - Çoklu bot desteği: her bot ayrı bir FAISS indeksine sahiptir
  - Kalıcılık: bot listesi bots.json'da, indeksler vectorstores/<bot_id>/ içinde tutulur
  - Yerel embedding: HuggingFace all-MiniLM-L6-v2 (504 zaman aşımı riskini ortadan kaldırır)
  - LLM: Google Gemini 1.5 Flash (ChatGoogleGenerativeAI)
  - Sohbet hafızası: ConversationBufferMemory (session_state içinde)
"""

import json
import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ─────────────────────────────────────────────────────────────────────────────
# Sayfa Ayarları
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Stil (CSS)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Arka plan */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Kenar çubuğu */
[data-testid="stSidebar"] {
    background: rgba(13, 17, 35, 0.95) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.25);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Başlık */
.main-title {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-title h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}
.main-title p { color: #94a3b8; font-size: 0.9rem; margin-top: 0.3rem; }

/* Sidebar bot butonları */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
}

/* Input ve textarea */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(100, 116, 139, 0.4) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
}

/* Bilgi ve uyarı kutuları */
.info-box {
    background: rgba(30, 58, 138, 0.35);
    border: 1px solid rgba(96, 165, 250, 0.4);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #93c5fd;
    font-size: 0.88rem;
    margin: 0.75rem 0;
}
.success-box {
    background: rgba(6, 78, 59, 0.4);
    border: 1px solid rgba(52, 211, 153, 0.4);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    color: #6ee7b7;
    font-size: 0.88rem;
    margin: 0.75rem 0;
}
.warning-box {
    background: rgba(92, 45, 5, 0.5);
    border: 1px solid rgba(251, 191, 36, 0.4);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    color: #fcd34d;
    font-size: 0.88rem;
    margin: 0.75rem 0;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b !important;
    padding: 0.25rem 0;
}
hr { border-color: rgba(100, 116, 139, 0.25) !important; }
label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler — dosya ve klasör yolları
# ─────────────────────────────────────────────────────────────────────────────

BOTS_FILE = Path("bots.json")           # Bot meta verilerinin saklandığı dosya
VECTORSTORE_DIR = Path("vectorstores")  # FAISS indekslerinin saklandığı kök klasör
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Bot listesi yönetimi — JSON okuma / yazma
# ─────────────────────────────────────────────────────────────────────────────

def load_bots() -> list[dict]:
    """bots.json dosyasını okur; dosya yoksa boş liste döner."""
    if BOTS_FILE.exists():
        with open(BOTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_bots(bots: list[dict]) -> None:
    """Bot listesini bots.json dosyasına yazar."""
    with open(BOTS_FILE, "w", encoding="utf-8") as f:
        json.dump(bots, f, ensure_ascii=False, indent=2)


def find_bot(bot_id: str) -> dict | None:
    """ID'ye göre bot kaydını döner."""
    return next((b for b in load_bots() if b["id"] == bot_id), None)

# ─────────────────────────────────────────────────────────────────────────────
# PDF işleme — metin çıkarma ve indeks oluşturma
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(pdf_bytes: bytes) -> str:
    """PdfReader ile yüklenen PDF'ten düz metin çıkarır."""
    import tempfile
    # Geçici dosyaya yaz, PdfReader ile oku
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    reader = PdfReader(tmp_path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    os.unlink(tmp_path)
    return text


def build_vectorstore(text: str, bot_id: str) -> None:
    """
    Metni parçalara böler, HuggingFace embeddingleri ile FAISS indeksi oluşturur
    ve vectorstores/<bot_id>/ dizinine kalıcı olarak kaydeder.
    
    chunk_size=400 ve chunk_overlap=50:
      - Küçük parçalar → embedding modelin 512 token sınırı içinde kalır
      - 50 token örtüşme → bağlam kayıplarını azaltır
    """
    # 1) Metin bölme
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    # 2) Yerel embedding modeli (Google API gerekmez, 504 hata riski yok)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3) FAISS indeksi oluştur ve diske kaydet
    store = FAISS.from_texts(chunks, embeddings)
    save_path = str(VECTORSTORE_DIR / bot_id)
    store.save_local(save_path)


@st.cache_resource(show_spinner=False)
def load_vectorstore(bot_id: str) -> FAISS:
    """
    Daha önce kaydedilmiş FAISS indeksini belleğe yükler.
    allow_dangerous_deserialization=True: pickle tabanlı FAISS dosyaları için gereklidir.
    Sonuç cache_resource ile önbelleğe alınır; aynı bot için tekrar yüklenmez.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        str(VECTORSTORE_DIR / bot_id),
        embeddings,
        allow_dangerous_deserialization=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sohbet zinciri oluşturma
# ─────────────────────────────────────────────────────────────────────────────

def make_chain(vector_store: FAISS, api_key: str) -> ConversationalRetrievalChain:
    """
    FAISS retriever + ConversationBufferMemory + Gemini LLM'i birleştiren
    ConversationalRetrievalChain döner.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,  # Gemini sistem mesajlarını desteklemez
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # En benzer 4 parçayı getir
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Session state başlatma
# ─────────────────────────────────────────────────────────────────────────────

# Her anahtar için varsayılan değeri yalnızca ilk açılışta ata
_defaults = {
    "selected_bot_id": None,      # Şu an aktif botun ID'si
    "creating_bot": False,        # Yeni bot oluşturma formu açık mı?
    "chat_histories": {},         # bot_id → [{"role": ..., "content": ...}]
    "chains": {},                 # bot_id → ConversationalRetrievalChain
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# API anahtarı kontrolü
# ─────────────────────────────────────────────────────────────────────────────

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = None

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  KENAR ÇUBUĞU (SIDEBAR)  ════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 PDF Botlarım")
    st.markdown("---")

    # ── Yeni bot oluşturma butonu ──────────────────────────────────────────
    if st.button("➕ Yeni Bot Oluştur", use_container_width=True, key="btn_new"):
        st.session_state.creating_bot = True
        st.session_state.selected_bot_id = None  # Aktif botu kapat

    st.markdown("---")

    # ── Kayıtlı bot listesi ───────────────────────────────────────────────
    bots = load_bots()

    if not bots:
        st.markdown(
            '<div class="info-box">Henüz bot yok.<br>➕ ile ilk botunu oluştur.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sidebar-label">Botlarım</div>', unsafe_allow_html=True)
        for bot in bots:
            is_active = st.session_state.selected_bot_id == bot["id"]
            prefix = "🟣" if is_active else "⚪"
            # Her bot için bir buton; tıklanınca o bot seçilir
            if st.button(
                f"{prefix}  {bot['name']}",
                key=f"bot_btn_{bot['id']}",
                use_container_width=True,
            ):
                st.session_state.selected_bot_id = bot["id"]
                st.session_state.creating_bot = False  # Formu kapat

    st.markdown("---")

    # ── Seçili botu sil ───────────────────────────────────────────────────
    if st.session_state.selected_bot_id:
        if st.button("🗑️ Seçili Botu Sil", use_container_width=True, key="btn_delete"):
            bid = st.session_state.selected_bot_id
            # bots.json'dan çıkar
            updated = [b for b in load_bots() if b["id"] != bid]
            save_bots(updated)
            # FAISS klasörünü sil
            import shutil
            vs_path = VECTORSTORE_DIR / bid
            if vs_path.exists():
                shutil.rmtree(vs_path)
            # Önbellek ve oturum temizliği
            st.cache_resource.clear()
            st.session_state.chains.pop(bid, None)
            st.session_state.chat_histories.pop(bid, None)
            st.session_state.selected_bot_id = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════  ANA EKRAN  ═════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# Başlık
st.markdown(
    """
    <div class="main-title">
        <h1>📄 PDF Chatbot</h1>
        <p>PDF belgelerinizi yükleyin — yapay zeka ile sohbet edin.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── API anahtarı yoksa uyarı ver ve dur ──────────────────────────────────────
if not GEMINI_API_KEY:
    st.markdown(
        """
        <div class="warning-box">
        ⚠️ <strong>GEMINI_API_KEY bulunamadı!</strong><br>
        <code>.streamlit/secrets.toml</code> dosyasına aşağıdaki satırı ekleyin:<br><br>
        <code>GEMINI_API_KEY = "AIza..."</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Ekran Modu 1 — Yeni Bot Oluşturma Formu
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.creating_bot:
    st.markdown("### ➕ Yeni Bot Oluştur")
    st.markdown("---")

    # Bot adı girişi
    bot_name = st.text_input(
        "Bot Adı *",
        placeholder="Örn: Sözleşme Asistanı",
        key="input_new_bot_name",
    )

    # PDF yükleme — birden fazla dosya kabul edilir
    uploaded_files = st.file_uploader(
        "PDF Dosyası Yükle (bir veya birden fazla) *",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader_new_bot",
    )

    col_ok, col_cancel = st.columns(2)

    with col_ok:
        create_clicked = st.button("✅ Oluştur ve Kaydet", use_container_width=True, key="btn_create")

    with col_cancel:
        if st.button("❌ İptal", use_container_width=True, key="btn_cancel"):
            st.session_state.creating_bot = False
            st.rerun()

    # ── Form gönderildiğinde botu oluştur ────────────────────────────────
    if create_clicked:
        if not bot_name.strip():
            st.error("⚠️ Lütfen bir bot adı girin.")
        elif not uploaded_files:
            st.error("⚠️ En az bir PDF dosyası yükleyin.")
        else:
            with st.spinner("PDF okunuyor ve FAISS indeksi oluşturuluyor..."):
                try:
                    # 1) Tüm PDF'lerden metin çıkar ve birleştir
                    combined_text = ""
                    for uf in uploaded_files:
                        combined_text += extract_text(uf.read()) + "\n\n"

                    if not combined_text.strip():
                        st.error("PDF'lerden metin çıkarılamadı (görüntü tabanlı PDF olabilir).")
                        st.stop()

                    # 2) Benzersiz bot ID'si üret
                    import uuid
                    new_id = str(uuid.uuid4())

                    # 3) FAISS indeksini oluştur ve kaydet
                    build_vectorstore(combined_text, new_id)

                    # 4) Bot kaydını bots.json'a ekle
                    from datetime import datetime
                    new_bot = {
                        "id": new_id,
                        "name": bot_name.strip(),
                        "pdf_files": [uf.name for uf in uploaded_files],
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    existing = load_bots()
                    existing.append(new_bot)
                    save_bots(existing)

                    # 5) Yeni botu hemen seç
                    st.session_state.selected_bot_id = new_id
                    st.session_state.creating_bot = False

                    st.success(f"✅ '{bot_name}' botu oluşturuldu!")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Hata oluştu: {exc}")

    st.stop()  # Formun altında başka içerik gösterme

# ─────────────────────────────────────────────────────────────────────────────
# Ekran Modu 2 — Bot Seçilmemişse Karşılama Ekranı
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.selected_bot_id:
    st.markdown(
        """
        <div class="info-box">
        👈 <strong>Sol menüden bir bot seçin</strong> ya da yeni bir bot oluşturun.<br><br>
        <b>Nasıl çalışır?</b><br>
        1️⃣ <em>➕ Yeni Bot Oluştur</em> butonuna tıklayın<br>
        2️⃣ Bot adını girin ve PDF dosyalarınızı yükleyin<br>
        3️⃣ <em>Oluştur ve Kaydet</em> butonuna tıklayın<br>
        4️⃣ Sol menüden botu seçip sohbete başlayın!<br><br>
        <b>Not:</b> Embedding işlemi tamamen yerel çalışır (internet gerekmez).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Ekran Modu 3 — Seçili Bot Sohbet Arayüzü
# ─────────────────────────────────────────────────────────────────────────────

# Bot meta verilerini yükle
active_bot = find_bot(st.session_state.selected_bot_id)
if not active_bot:
    st.error("Bot bulunamadı. Lütfen tekrar seçin.")
    st.session_state.selected_bot_id = None
    st.stop()

bid = active_bot["id"]

# Bot başlığı
pdf_list = ", ".join(active_bot.get("pdf_files", []))
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;padding:0.5rem 0 1.25rem;">
        <div style="width:52px;height:52px;border-radius:50%;flex-shrink:0;
                    background:linear-gradient(135deg,#a78bfa,#7c3aed);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.6rem;box-shadow:0 4px 14px rgba(167,139,250,0.4);">🤖</div>
        <div>
            <div style="font-size:1.35rem;font-weight:700;color:#e2e8f0;">{active_bot['name']}</div>
            <div style="font-size:0.78rem;color:#64748b;">📎 {pdf_list or 'PDF bilgisi yok'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── FAISS indeksini yükle (ilk erişimde, sonra önbellekten gelir) ────────────
if bid not in st.session_state.chains:
    vs_path = VECTORSTORE_DIR / bid
    if not vs_path.exists():
        st.markdown(
            '<div class="warning-box">⚠️ Bu bota ait FAISS indeksi bulunamadı.<br>'
            'Lütfen botu silin ve yeniden oluşturun.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    with st.spinner("Vektör veritabanı yükleniyor..."):
        try:
            vs = load_vectorstore(bid)
            st.session_state.chains[bid] = make_chain(vs, GEMINI_API_KEY)
        except Exception as exc:
            st.error(f"İndeks yüklenemedi: {exc}")
            st.stop()

chain = st.session_state.chains[bid]

# ── Sohbet geçmişini başlat ──────────────────────────────────────────────────
if bid not in st.session_state.chat_histories:
    st.session_state.chat_histories[bid] = []

history: list[dict] = st.session_state.chat_histories[bid]

# ── Geçmiş mesajları göster ──────────────────────────────────────────────────
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Geçmiş boşsa ipucu göster
if not history:
    st.markdown(
        '<div class="info-box" style="text-align:center;">'
        '💡 Bu PDF hakkında bir soru yazın!</div>',
        unsafe_allow_html=True,
    )

# ── Kullanıcı girdisi — st.chat_input ────────────────────────────────────────
user_input = st.chat_input("Sorunuzu yazın…", key=f"chat_input_{bid}")

if user_input and user_input.strip():
    # Kullanıcı mesajını geçmişe ekle ve ekranda göster
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Asistan cevabını üret
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor..."):
            try:
                result = chain({"question": user_input})
                answer: str = result.get("answer", "Cevap üretilemedi.")
            except Exception as exc:
                answer = f"⚠️ Hata: {exc}"
        st.markdown(answer)

    # Cevabı geçmişe kaydet
    history.append({"role": "assistant", "content": answer})
    st.session_state.chat_histories[bid] = history

# ── Sohbeti sıfırla butonu ───────────────────────────────────────────────────
if history:
    st.markdown("---")
    if st.button("🗑️ Sohbeti Sıfırla", key=f"clear_{bid}"):
        st.session_state.chat_histories[bid] = []
        # Zinciri de sıfırla (hafıza temizlenir)
        st.session_state.chains.pop(bid, None)
        st.cache_resource.clear()
        st.rerun()
