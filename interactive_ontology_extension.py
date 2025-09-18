import os
import sys
import re
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import rdflib

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("ontology_extension.log", mode='w', encoding='utf-8')])

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY not found in environment variables.")


genai.configure(api_key=API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

logging.info("Initializing the embedding model...")
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

logging.info("Loading Chroma vector database...")
vector_store = Chroma(
    embedding_function=embedding_function,
    persist_directory="vector_db",
    collection_name="ontology_corpus"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

ONTOLOGY_FILE = "CausalRoadAccidentOntology_2_0.ttl"

def retrieve_context(query: str) -> str:
    """
    Retrieve relevant context from the vector database based on the query.
    """
    logging.info("Retrieving context from vector database...")
    relevant_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    logging.debug(f"Retrieved context:\n{context}")
    return context

def generate_suggestions(user_query, context_docs, ontology_context):
    """
    Construct the augmented prompt and get suggestions from the LLM.
    """

    prompt = f"""
    You are an expert in Autonomous Vehicle (AV) safety engineering, specializing in identifying and modeling catastrophic failure scenarios (also known as 'black swans').
    Your task is to help extend an existing ontology in Turtle (TTL) format by adding new classes, properties, or relationships based on the user's extension goal.

    ## User's Extension Goal
    {user_query}

    ## Current Ontology Context (in TTL format)
    ```ttl
    {ontology_context}
    ```

    ## Relevant Context from Documents
    {context_docs}

    ## Instructions
    1. Based on the user's goal and the provided context, suggest 5-7 new, highly specific classes, properties, or relationships to add to the ontology.
    2. Use proper TTL syntax and ensure the suggestions are syntactically valid.
    3. Focus on concepts like AV components, failure modes, and external threats.
    4. Use rdfs:subClassOf to define hierarchies and owl:ObjectProperty for relationships.
    5. Respond ONLY with the new TTL statements.
    6. DO NOT repeat any part of the original ontology.
    7. Answer ONLY with the TTL code block. Do not add explanations or introductory text.
    """

    try:
        response = llm.generate_content(prompt)
        raw_text = response.text.strip()

        logging.debug(f"Model response:\n{raw_text}")

        match = re.search(r"```ttl\s*(.+?)\s*```", raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            if raw_text.startswith(("@prefix", "<")) and raw_text.endswith("."):
                return raw_text
            else:
                print("\n--- ATTENZIONE: Nessun blocco di codice TTL valido trovato nella risposta del modello. ---")
                print("Risposta ricevuta:\n" + raw_text)
                return None

    except Exception as e:
        logging.error(f"Error generating suggestions: {e}")
        return None

def get_ontology_context(ontology_file):
    """
    Load the current ontology content to provide context to the model.
    """
    try:
        with open(ontology_file, "r", encoding="utf-8") as file:
            ontology_content = file.read()
        logging.info("Loaded current ontology content.")
        return ontology_content
    except Exception as e:
        logging.error(f"Error loading ontology file: {e}")
        return "No ontology content available."

def validate_and_append(suggestions_ttl, ontology_file):
    """
    Validate the TTL syntax and append valid suggestions to the ontology file.
    """
    if not suggestions_ttl:
        logging.warning("No suggestions to validate.")
        return

    ontology_context = get_ontology_context(ontology_file)
    prefixes = "\n".join([line for line in ontology_context.splitlines() if line.strip().startswith("@prefix")])
    full_ttl_to_validate = prefixes + "\n\n" + suggestions_ttl

    g = rdflib.Graph()
    try:
        g.parse(data=full_ttl_to_validate, format="n3")
        logging.info("Suggestions are syntactically valid TTL.")
    except Exception as e:
        logging.error(f"Invalid TTL syntax in suggestions: {e}")
        return

    print("\n --- Suggested Additions ---\n")
    print(suggestions_ttl)
    print("\n---------------------------\n")

    choice = input("Do you want to append these suggestions to the ontology? (yes/no): ").strip().lower()
    if choice in ['yes', 'y']:
        try:
            with open(ontology_file, "a", encoding="utf-8") as file:
                file.write("\n\n# --- New Additions ---\n")
                file.write(suggestions_ttl)
            logging.info("Suggestions appended to the ontology file.")
            print("Suggestions successfully appended to the ontology.")
        except Exception as e:
            logging.error(f"Error appending suggestions to ontology file: {e}")
    else:
        logging.info("User chose not to append suggestions.")

def main_loop():
    logging.info("Starting interactive ontology extension loop.")

    while True:
        user_query = input("\n> Enter your extension goal (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            logging.info("Exiting the interactive loop.")
            break

        print("Retrieving context...")
        context_docs = retrieve_context(user_query)

        logging.info("Retrieved context documents.")
        for idx, doc in enumerate(context_docs.split("\n\n")):
            logging.info(f"--- Document {idx+1} ---\n{doc}\n")

        logging.info("... Generating suggestions from the model.")
        ontology_context = get_ontology_context(ONTOLOGY_FILE)
        suggestions_ttl = generate_suggestions(user_query, context_docs, ontology_context)

        validate_and_append(suggestions_ttl, ONTOLOGY_FILE)

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)