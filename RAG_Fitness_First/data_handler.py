import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from chromadb.config import Settings

class DataHandler:
    def __init__(
        self,
        data_path,
        embedding_model_name="/PATH_TO_A_SENTENCE_TRANSFORMER_MODEL/all-MiniLM-L6-v2", #change this  to a local path
        collection_name="fitness_passport_faqs",
        persist_directory="./chroma_db"
    ):
        self.data_path = data_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.settings = Settings(allow_reset=True)

        # Create a PersistentClient without tenant/database parameters
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=self.settings
        )

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print(f"ChromaDB persist directory: {self.persist_directory}")

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        return data

    def chunk_data(self, data):
        chunked_data = []
        for item in data:
            question = item["question"]
            answer = item["answer"]
            paragraphs = answer.split("\n\n")  # Split into paragraphs
            for paragraph in paragraphs:
                if paragraph:  # Check if paragraph is not empty
                    chunked_data.append({
                        "question": question,
                        "answer_chunk": paragraph.strip(),
                    })
        return chunked_data

    def create_embeddings(self, data):
        data_with_embeddings = []
        for item in data:
            chunk_embedding = self.embedding_model.encode(item["answer_chunk"]).tolist()
            data_with_embeddings.append({
                "question": item["question"],
                "answer_chunk": item["answer_chunk"],
                "answer_chunk_embedding": chunk_embedding
            })
        return data_with_embeddings

    def create_chroma_collection(self, data_with_embeddings):
        metadatas = []  # we create a metadata list
        for item in data_with_embeddings:  # we append the question to the metadata list
            metadatas.append({"source": item["question"]})

        collection = self.chroma_client.create_collection(name=self.collection_name)
        if data_with_embeddings:  # we add this test
            collection.add(
                embeddings=[item["answer_chunk_embedding"] for item in data_with_embeddings],
                documents=[item["question"] + "\n" + item["answer_chunk"] for item in data_with_embeddings],
                metadatas=metadatas,  # we add the metadatas
                ids=[f"id{i}" for i in range(len(data_with_embeddings))]
            )
        return collection

    def query_chroma(self, query, n_results=2):
        query_embedding = self.embedding_model.encode(query).tolist()
        collection = self.chroma_client.get_collection(name=self.collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas']  # we include the metadatas and documents
        )
        return results

    def process_data_and_create_collection(self):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        data = self.load_data()
        chunked_data = self.chunk_data(data)
        data_with_embeddings = self.create_embeddings(chunked_data)
        try:
            collection = self.create_chroma_collection(data_with_embeddings)
        except Exception as e:
            print(f"An error occurred while creating the collection: {e}")
            raise
        return collection

    def delete_chroma_collection(self):
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Successfully deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection {self.collection_name}: {e}")
