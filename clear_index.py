from pinecone import Pinecone

# Import API key from environment variables
from dotenv import load_dotenv
import os

PINECONE_KEY = os.getenv("PINECONE_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_KEY)

# Specify the name of the index you want to delete
index_name = "liq-registrations"

# Delete the index
pc.delete_index(name=index_name)

# Confirm that the index has been deleted
print(f"Index '{index_name}' has been deleted.")
