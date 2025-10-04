from typing import Set
from tornado.gen import sleep

from backend.core import run_llm
import streamlit as st

st.header("Langchain Udemy Course - Documentation helper bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here ....")

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string



if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)

        print (generated_response)

        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )