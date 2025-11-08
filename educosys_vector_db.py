import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


#Load the environment variable
load_dotenv()

#initiate the Model
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

#Initiate the embeddings 
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#Create the vector store
vector_store = Chroma(
    collection_name="educosys_learning_material",
    embedding_function=embeddings,
    persist_directory="./chroma_educosys_db",  # Where to save data locally, remove if not necessary
)


#Load the File
loader = DirectoryLoader(
    "./docs",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)

docs = loader.load()
print(f"Loaded {len(docs)} pages with PyMuPDFLoader.")

#Split the Document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

#Add all the Documents
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])