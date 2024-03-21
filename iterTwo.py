#libraries to interact with  system
import os
import numpy as np
import json
from dotenv import load_dotenv
from typing import List


#libraries to interact with  Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from fastembed.embedding import DefaultEmbedding


# Load environment variables from .env file
load_dotenv()
# Accessing environment variables
API_KEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")



#libraries to interact with langchain
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader


qdrant_client = QdrantClient(
    DATABASE_URL,
    api_key=API_KEY,
)

qdrant_client.recreate_collection(
    collection_name='diseases', 
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Specify the folder path containing text files
folder_path = r'C:\Users\Admin\CLASS\final-project\dataChat'

# Initialize lists to store startup data and vectors
startup_data = []
vectors = []

# Iterate over each file in the folder
#seems to store txt file as a np array without embedding


# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.txt'):
#         # Load text data from the current file
#         with open(os.path.join(folder_path, file_name), 'r') as file:
#             content = file.read()
#             # Check if the text file contains JSON data
#             try:
#                 data = json.loads(content)
#                 startup_data.append(data)
#             except json.JSONDecodeError:
#                 # If not JSON, assume it contains vectors (e.g., numpy arrays)
#                 vector = np.fromstring(content, dtype=float, sep=' ')  # Adjust dtype and sep as needed
#                 vectors.append(vector)

# # Convert vectors list to a single numpy array
# vectors = np.stack(vectors)

# # Now you can use startup_data and vectors as needed


# #embedding the txt files
# embedding_model = DefaultEmbedding()

# for file_name in os.listdir(folder_path):
#   loader = TextLoader(file_name)
#   documents = loader.load()
#   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  #might change chunk_overlap
#   docs = text_splitter.split_documents(documents)

# loader = TextLoader("potato context.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embeddings: List[np.ndarray] = list(embedding_model.embed(docs))


# List to store document contents
documents: List[str] = []

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Construct the full path of the file
        file_path = os.path.join(folder_path, filename)
        # Read the contents of the file and append to the documents list
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())


#using the auto way of qdrant client
            
# client = QdrantClient(":memory:")
# client.add(collection_name="diseases", documents=documents)

            
# Initialize the DefaultEmbedding class
embedding_model = DefaultEmbedding()
embeddings: List[np.ndarray] = list(embedding_model.embed(documents))
print(embeddings[0].shape)