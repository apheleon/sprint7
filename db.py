from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50
DB_DIR = "vector_db"
DB_TYPE = "faiss"
DOCS_DIRECTORY = "knowledge_base/origin"

# Инициализация модели для эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Инициализация сплиттера
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True,
)

def process_documents(directory_path):
    documents = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    metadata = {
                        "source": file_path,
                        "title": os.path.splitext(file)[0]
                    }
                    
                    chunks = text_splitter.create_documents([text], [metadata])
                    
                    for i, chunk in enumerate(chunks):
                        chunk.metadata["chunk_id"] = i
                        chunk.metadata["start_index"] = chunk.metadata.get("start_index", 0)
                    
                    documents.extend(chunks)
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    return documents

def create_vector_db(documents):
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_DIR)
    
    return db

    
documents = process_documents(DOCS_DIRECTORY)
print(f"Created {len(documents)} chunks")
    
db = create_vector_db(documents)
print(f"DB created in  {DB_DIR}")