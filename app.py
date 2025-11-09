import streamlit as st
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from db.educosys_vector_db import get_vector_store

# --------------------------------------------------------
# 1️⃣ Page Setup
# --------------------------------------------------------
st.set_page_config(
    page_title="📚 Educosys Retrieval Agent (Streaming)",
    page_icon="🔎",
    layout="wide"
)

load_dotenv()

st.markdown(
    "<h1 style='text-align:center;color:#2E86DE;'>📚 Educosys Retrieval Agent</h1>",
    unsafe_allow_html=True
)
st.caption("Powered by Gemini + Chroma Vector DB (RAG + Live Streaming)")
st.divider()

# --------------------------------------------------------
# 2️⃣ Load Model and Vector Store
# --------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return init_chat_model("google_genai:gemini-2.5-flash-lite")

@st.cache_resource(show_spinner=False)
def load_vector_store():
    return get_vector_store()

model = load_model()
vector_store = load_vector_store()

# --------------------------------------------------------
# 3️⃣ User Input
# --------------------------------------------------------
st.subheader("🔍 Ask a Question Based on Your Documents")
query = st.text_input("Enter your query:", placeholder="e.g., Explain Deep Learning")

if st.button("Retrieve & Stream Answer 🚀"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid query.")
    else:
        # --------------------------------------------------------
        # 4️⃣ Retrieve Context from Vector Store
        # --------------------------------------------------------
        with st.spinner("🔎 Searching for relevant content..."):
            retrieved_docs = vector_store.similarity_search(query, k=2)

        if not retrieved_docs:
            st.error("❌ No relevant documents found in your Chroma DB.")
        else:
            st.success(f"✅ Found {len(retrieved_docs)} relevant document(s).")

            # Combine top chunks into one prompt
            context = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )

            full_prompt = (
                f"You are a helpful assistant. Use the following document context to answer clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            # --------------------------------------------------------
            # 5️⃣ Stream the Gemini Response
            # --------------------------------------------------------
            st.divider()
            st.markdown("### 🧠 Gemini Answer (Live Stream)")
            response_container = st.empty()

            full_response = ""
            for chunk in model.stream([HumanMessage(content=full_prompt)]):
                delta = chunk.content
                if delta:
                    full_response += delta
                    response_container.markdown(full_response + "▌")
                    time.sleep(0.01)

            st.success("✅ Answer generation complete!")

            # --------------------------------------------------------
            # 6️⃣ Display Final Answer and Sources
            # --------------------------------------------------------
            st.divider()
            st.markdown("### 📄 Final Answer")
            st.markdown(full_response)

            with st.expander("📚 Document Sources"):
                for doc in retrieved_docs:
                    st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                    st.write(doc.page_content[:500] + "...")
