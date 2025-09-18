import sys
import logging

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_and_chunk_corpus(corpus_path="corpus/"):
    """
    Load and chunk documents from the specified corpus path.
    """
    logging.info(f"Loading documents from {corpus_path}...")

    loader = DirectoryLoader(corpus_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        logging.warning("No documents found in the specified corpus path.")
        return None

    logging.info(f"Loaded {len(documents)} documents. Starting chunking...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunked_docs = text_splitter.split_documents(documents)
    logging.info(f"Chunked documents into {len(chunked_docs)} pieces.")
    return chunked_docs

def build_vector_db(chunked_docs, db_path="vector_db", collection_name="ontology_corpus"):
    """
    Build a vector database from chunked documents.
    """
    if not chunked_docs:
        logging.error("No chunked documents provided to build the vector database.")
        return None

    logging.info("Initializing embeddings model...")

    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    logging.info("Creating a Chroma vector database...")
    vector_db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_function,
        persist_directory=db_path,
        collection_name=collection_name
    )

    logging.info(f"Vector database created at {db_path} with collection '{collection_name}'.")
    return vector_db

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("build_vector.log", mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)])

    chunked_corpus = load_and_chunk_corpus(corpus_path="corpus/")
    if chunked_corpus:
        build_vector_db(chunked_corpus, db_path="vector_db", collection_name="ontology_corpus")
