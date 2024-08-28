import boto3
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

## S3 client (optional, if you still want to use it for other operations)
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())

## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)

    return docs

## Create vector store and save locally
def create_vector_store(request_id, documents, local_save_path):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    
    # Ensure the local_save_path directory exists
    os.makedirs(local_save_path, exist_ok=True)

    # Define the file paths
    file_name = f"{request_id}.bin"
    faiss_file_path = os.path.join(local_save_path, file_name + ".faiss")
    pkl_file_path = os.path.join(local_save_path, file_name + ".pkl")

    # Save the files locally
    vectorstore_faiss.save_local(index_name=file_name, folder_path=local_save_path)

    print(f"Files saved locally at: {faiss_file_path} and {pkl_file_path}")

    return True

## Main function
def main():
    # Path to the PDF file and output directory
    base_path = "/home/ec2-user"
    pdf_file_path = os.path.join(base_path, "PDFNAMEHERE.pdf")
    
    if os.path.exists(pdf_file_path):
        request_id = get_unique_id()
        print(f"Processing file with Request Id: {request_id}")

        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()

        print(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)    # 1000 characters 200 overlaps
        print(f"Splitted Docs length: {len(splitted_docs)}")
        print("Creating the Vector Store")

        # Define the path where you want to save the output files inside the container
        local_save_path = base_path  # Save in /home/ec2-user

        result = create_vector_store(request_id, splitted_docs, local_save_path)

        if result:
            print("Congratulations! PDF processed and files saved successfully.")
        else: 
            print("Error! Please check logs.")
    else:
        print(f"File not found: {pdf_file_path}")

if __name__ == "__main__":
    main()


### To Run this Script:
# Configure docker and AWS CLI and run the following:    
# docker build -t pdf-reader-admin .    
# docker run -e BUCKET_NAME=<bucket-name> -v ~/.aws:/root/.aws -v /home/ec2-user:/home/ec2-user -p 8083:8083 -it pdf-reader-admin
