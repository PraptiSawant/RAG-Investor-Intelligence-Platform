from dotenv import load_dotenv
import os

from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def create_index():
    # Initialize the modern Pinecone client using your .env API key
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Check if the index already exists to prevent crash errors
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists. Skipping creation.")
        return

    print(f"Creating a brand new Pinecone index: '{index_name}'...")
    
    # Programmatically define your dimensions, distance metrics, and infrastructure type
    pc.create_index(
        name=index_name,
        dimension=384,          
        metric="cosine",        
        spec=ServerlessSpec(
            cloud="aws",        
            region="us-east-1" 
        )
    )
    print(f"Success! Index '{index_name}' created successfully on the Pinecone Cloud.")

if __name__ == "__main__":
    create_index()