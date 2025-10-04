import os
import sys

from dotenv import load_dotenv

from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


IndexName = os.getenv("INDEX_NAME")

def run_llm(query: str):

    embeddings = OpenAIEmbeddings(model="text-embeddings-3-small")
    docsearch = PineconeVectorStore(index_name=IndexName, embedding=embeddings)

    chat = ChatOpenAI(temperature=0, verbose=True)
    retrieval_qa_chat = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
        )

    result = qa.invoke(input={"input": query})
    return result



if __name__ == "__main__":
    res = run_llm(query="What is a langchain chain?")
    print(res['answer'])






