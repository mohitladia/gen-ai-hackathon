import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------
# 1️⃣ Load environment variables
# --------------------------------------------------------
load_dotenv()

# --------------------------------------------------------
# 2️⃣ Initialize Gemini embeddings
# --------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# --------------------------------------------------------
# 3️⃣ Create (or connect to) Chroma collection
# --------------------------------------------------------
vector_store = Chroma(
    collection_name="educosys_learning_material",
    embedding_function=embeddings,
    persist_directory="./chroma_educosys_db"
)

# --------------------------------------------------------
# 4️⃣ Load PDFs from the docs directory
# --------------------------------------------------------
loader = DirectoryLoader(
    "./docs",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)
docs = loader.load()
print(f"📄 Loaded {len(docs)} pages from './docs/'")

# --------------------------------------------------------
# 5️⃣ Split documents into manageable text chunks
# --------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
splits = splitter.split_documents(docs)
print(f"✂️  Created {len(splits)} text chunks.")

# --------------------------------------------------------
# 6️⃣ Add chunks to the Chroma vector store
# --------------------------------------------------------
doc_ids = vector_store.add_documents(splits)
print(f"🧠 Added {len(doc_ids)} chunks to Chroma DB.")

# --------------------------------------------------------
# 7️⃣ Persist to disk for reuse
# --------------------------------------------------------
print("✅ Vector store successfully built and saved to './chroma_educosys_db'")
