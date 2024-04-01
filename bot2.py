
#  to start python -m venv fastapi
# fastapi/Scripts/activate
#python -m pip install fastapi uvicorn[standard]
# to run uvicorn fast2:app --reload
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
qdrant_link =  os.environ.get("QDRANT_LINK")
qdrant_api = os.environ.get("QDRANT_API")
openai_api = os.environ.get("OPEN_AI_API")

#loading qdrant and open ai apis
import openai

openai_client = openai.Client(
    api_key= openai_api
)

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client =  QdrantClient(
    qdrant_link,
    api_key= qdrant_api,
)



from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

# MODIFIED FUNCTIONS

def askQuestion(question):
    search_result = client.query(
    collection_name="diseases",
    query_text= question)
    #call the save to text function?
    return search_result


# retrieve answer
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

import numpy as np

# Define the function to save the vector db output to a text file
def save_to_text_file(output_list, file_path):
    with open(file_path, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)


file_path = 'output.txt'

# Save the output list to the text file
save_to_text_file(search_result, file_path)



# search_result = client.query(
#     collection_name="diseases",
#     query_text="what is latent TB"
# )
# print(search_result)


loader = TextLoader("output.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)

docs = text_splitter.split_documents(documents)


# formulating answer
model_name = "gpt-3.5-turbo-instruct"
# query = " what is latent TB"
llm = OpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

answer = chain.run(input_documents=docs, question=query)
print(answer)



