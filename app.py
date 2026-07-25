import streamlit as st
import json
import os
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ─────────────────────────────────────────────
# Sabitler ve Dizin Yapısı
# ─────────────────────────────────────────────
BOTS_FILE = Path("bots.json")
VECTORSTORES_DIR = Path("vectorstores")
VECTORSTORES_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Bot Kayıt Dosyası Yardımcıları
# ─────────────────────────────────────────────
def load_bots() -> dict:
    """bots.json dosyasını okur; yoksa boş sözlük döner."""
    if BOTS_FILE.exists():
        with open(BOTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_bots(bots: dict) -> None:
    """Botlar sözlüğünü bots.json dosyasına yazar."""
    with open(BOTS_FILE, "w", encoding="utf-8") as f:
        json.dump(bots, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# PDF İşleme Yardımcıları
# ─────────────────────────────────────────────
def extract_text_from_pdfs(pdf_files) -> str:
    """Yüklenen PDF dosyalarından metin çıkarır."""
    full_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def split_text_into_chunks(text: str):
    """Metni LangChain chunk'larına böler."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.create_documents([text])


# ─────────────────────────────────────────────
# OpenAI Nesneleri
# ─────────────────────────────────────────────
def get_embeddings():
    """OpenAI embedding nesnesini döner."""
    return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])


def get_llm():
    """OpenAI ChatLLM nesnesini döner."""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=st.secrets["OPENAI_API_KEY"],
    )


# ─────────────────────────────────────────────
# FAISS Vektör Deposu İşlemleri
# ─────────────────────────────────────────────
def build_and_save_vectorstore(bot_id: str, docs) -> FAISS:
    """Chunk'lardan FAISS indeksi oluşturur ve diske kaydeder."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    save_path = str(VECTORSTORES_DIR / bot_id)
    vectorstore.save_local(save_path)
    return vectorstore


def load_vectorstore(bot_id: str) -> FAISS:
    """Diskten mevcut FAISS indeksini yükler."""
    embeddings = get_embeddings()
    save_path = str(VECTORSTORES_DIR / bot_id)
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def append_docs_to_vectorstore(bot_id: str, new_docs) -> None:
    """Mevcut FAISS indeksine yeni belgeler ekler ve kaydeder."""
    vectorstore = load_vectorstore(bot_id)
    vectorstore.add_documents(new_docs)
    save_path = str(VECTORSTORES_DIR / bot_id)
    vectorstore.save_local(save_path)


# ─────────────────────────────────────────────
# Konuşma Zinciri
# ─────────────────────────────────────────────
def build_conversation_chain(bot_id: str) -> ConversationalRetrievalChain:
    """
    Belirtilen bot için FAISS indeksini yükler ve
    ConversationalRetrievalChain oluşturur.
    """
    vectorstore = load_vectorstore(bot_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
    )
    return chain


# ─────────────────────────────────────────────
# Session State Başlatma
# ─────────────────────────────────────────────
def init_session_state() -> None:
    """Gerekli session_state anahtarlarını başlatır."""
    if "bots" not in st.session_state:
        st.session_state.bots = load_bots()
    if "active_bot_id" not in st.session_state:
        st.session_state.active_bot_id = None
    if "view" not in st.session_state:
        # 'chat' | 'new_bot'
        st.session_state.view = "chat"
    if "chains" not in st.session_state:
        # bot_id -> ConversationalRetrievalChain
        st.session_state.chains = {}
    if "chat_histories" not in st.session_state:
        # bot_id -> list of {"role": ..., "content": ...}
        st.session_state.chat_histories = {}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar() -> None:
    """Sol kenar çubuğunu oluşturur."""
    with st.sidebar:
        st.title("🤖 PDF Chatbots")
        st.divider()

        if st.button("➕ Yeni Bot Oluştur", use_container_width=True, type="primary"):
            st.session_state.view = "new_bot"
            st.session_state.active_bot_id = None

        st.divider()
        st.subheader("Kayıtlı Botlar")

        bots = st.session_state.bots
        if not bots:
            st.info("Henüz kayıtlı bot yok.")
        else:
            for bot_id, bot_info in bots.items():
                label = f"💬 {bot_info['name']}"
                if st.button(label, key=f"btn_{bot_id}", use_container_width=True):
                    st.session_state.active_bot_id = bot_id
                    st.session_state.view = "chat"


# ─────────────────────────────────────────────
# Yeni Bot Oluşturma Ekranı
# ─────────────────────────────────────────────
def render_new_bot_view() -> None:
    """Yeni bot oluşturma formunu gösterir."""
    st.header("➕ Yeni Bot Oluştur")
    st.write("Bir bot adı girin ve PDF dosyalarını yükleyin.")

    bot_name = st.text_input(
        "Bot Adı",
        placeholder="Örn: Hukuk Asistanı",
        max_chars=50,
    )

    uploaded_files = st.file_uploader(
        "PDF Dosyaları Yükle",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    create_btn = st.button(
        "🚀 Botu Oluştur",
        type="primary",
        disabled=(not bot_name or not uploaded_files),
    )

    if create_btn:
        with st.spinner("PDF'ler işleniyor ve vektör deposu oluşturuluyor…"):
            import uuid
            bot_id = str(uuid.uuid4())[:8]

            raw_text = extract_text_from_pdfs(uploaded_files)
            if not raw_text.strip():
                st.error("Yüklenen PDF'lerden metin çıkarılamadı. Lütfen geçerli bir PDF yükleyin.")
                return

            docs = split_text_into_chunks(raw_text)
            build_and_save_vectorstore(bot_id, docs)

            bot_info = {
                "name": bot_name,
                "pdf_count": len(uploaded_files),
            }
            st.session_state.bots[bot_id] = bot_info
            save_bots(st.session_state.bots)

        st.success(f"✅ **{bot_name}** başarıyla oluşturuldu!")
        st.session_state.active_bot_id = bot_id
        st.session_state.view = "chat"
        st.rerun()


# ─────────────────────────────────────────────
# Sohbet Ekranı
# ─────────────────────────────────────────────
def get_or_create_chain(bot_id: str) -> ConversationalRetrievalChain:
    """
    Belirtilen bot için zinciri session_state'ten alır
    ya da yeniden oluşturur.
    """
    if bot_id not in st.session_state.chains:
        st.session_state.chains[bot_id] = build_conversation_chain(bot_id)
    return st.session_state.chains[bot_id]


def render_chat_view(bot_id: str) -> None:
    """Seçili botun sohbet arayüzünü gösterir."""
    bot_info = st.session_state.bots[bot_id]
    bot_name = bot_info["name"]

    st.header(f"💬 {bot_name}")

    # ── Ek PDF Yükleme Bölümü ──────────────────────────────────
    with st.expander("📎 Bu Bota Ek PDF Yükle", expanded=False):
        extra_files = st.file_uploader(
            "Eklemek istediğiniz PDF dosyalarını seçin",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"extra_pdf_{bot_id}",
        )
        add_btn = st.button(
            "Ekle",
            key=f"add_btn_{bot_id}",
            disabled=not extra_files,
            type="secondary",
        )
        if add_btn and extra_files:
            with st.spinner("Yeni PDF'ler indekse ekleniyor…"):
                new_text = extract_text_from_pdfs(extra_files)
                if not new_text.strip():
                    st.error("PDF'lerden metin çıkarılamadı.")
                else:
                    new_docs = split_text_into_chunks(new_text)
                    append_docs_to_vectorstore(bot_id, new_docs)

                    # Zinciri sıfırla: indeks güncellendiği için yeniden yüklenecek
                    if bot_id in st.session_state.chains:
                        del st.session_state.chains[bot_id]

                    # PDF sayısını güncelle
                    st.session_state.bots[bot_id]["pdf_count"] = (
                        bot_info.get("pdf_count", 0) + len(extra_files)
                    )
                    save_bots(st.session_state.bots)
                    st.success(f"✅ {len(extra_files)} PDF başarıyla eklendi.")
                    st.rerun()

    st.divider()

    # ── Sohbet Geçmişi ──────────────────────────────────────────
    if bot_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[bot_id] = []

    history = st.session_state.chat_histories[bot_id]

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Kullanıcı Girişi ────────────────────────────────────────
    user_input = st.chat_input("Bir şey sorun…")

    if user_input:
        # Kullanıcı mesajını göster ve kaydet
        with st.chat_message("user"):
            st.markdown(user_input)
        history.append({"role": "user", "content": user_input})

        # Yanıt üret
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor…"):
                chain = get_or_create_chain(bot_id)
                response = chain.invoke({"question": user_input})
                answer = response.get("answer", "Yanıt alınamadı.")
            st.markdown(answer)

        history.append({"role": "assistant", "content": answer})


# ─────────────────────────────────────────────
# Ana Uygulama
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📄",
        layout="wide",
    )

    init_session_state()
    render_sidebar()

    # Aktif görünümü belirle
    if st.session_state.view == "new_bot":
        render_new_bot_view()
    elif st.session_state.view == "chat" and st.session_state.active_bot_id:
        bot_id = st.session_state.active_bot_id
        if bot_id in st.session_state.bots:
            render_chat_view(bot_id)
        else:
            st.warning("Seçilen bot bulunamadı.")
    else:
        # Hoşgeldiniz ekranı
        st.title("📄 PDF Chatbot")
        st.write(
            "Sol menüden bir bot seçin ya da **➕ Yeni Bot Oluştur** "
            "butonuna tıklayarak başlayın."
        )
        st.info(
            "Bu uygulama PDF dosyalarınızı analiz ederek sorularınızı yanıtlar. "
            "Birden fazla PDF yükleyebilir ve istediğiniz kadar bot oluşturabilirsiniz."
        )


if __name__ == "__main__":
    main()
