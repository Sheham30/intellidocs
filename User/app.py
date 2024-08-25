import boto3
import streamlit as st
import os
import uuid


## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

## import langchain libraries
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


bedrock_client = boto3.client(service_name="bedrock-runtime")

# bedrock_client = boto3.client(
#     service_name="bedrock-runtime",
#     region_name="us-east-1",
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
#     # aws_session_token=os.getenv("AWS_SESSION_TOKEN")  # If using temporary credentials
# )

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = "/tmp/"

def get_unique_id():
    return str(uuid.uuid4())


## To request query from claude model here 
def get_llm():

    ## For Anthropic Claude
    llm = Bedrock(
            model_id="anthropic.claude-v2:1", 
            client=bedrock_client, 
            model_kwargs={"max_tokens_to_sample":512}
            )
    return llm

    ## For Amazon Titan
    # llm = Bedrock(
    #         model_id="amazon.titan-text-lite-v1", 
    #         client=bedrock_client, 
    #         model_kwargs={"maxTokenCount":250}
    #         )
    # return llm


## To get response

def get_response(llm, vectorstore, question):

    ## Creating prompt template

    ## **By Using Anthropic Claude**
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:
    
    """

    ## **By Using Amazon Titan**
    # prompt_template = """

    # {context}

    # Based on the information above, {question}

    # """


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5} # it will get 5 similar chunks
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']



## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

## Main function
def main():
    st.header("This is User site for chat with PDF demo using Bedrock")

    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

    ## create index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path = folder_path,
        embeddings = bedrock_embeddings,
        allow_dangerous_deserialization = True
        )

    st.write("INDEX IS READY")

    query = st.text_input("Please ask your question")

    if st.button("Ask Question"):
        with st.spinner("Querying....."):

            llm = get_llm()

            # get response
            st.write(get_response(llm, faiss_index, query))
            st.success("Done")

if __name__=="__main__":
    main()