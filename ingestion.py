import asyncio
import os
import ssl
from typing import Any, List, Dict

import certifi

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract
from numpy.array_api import result_type
from openai import embeddings
from sqlalchemy.testing.suite.test_reflection import metadata

from langchain_pinecone.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from logger import (Colors, log_info, log_error, log_header, log_success, log_warning)

load_dotenv()

# Configure SSL Certificate to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
    )
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
#vectorstore = PineconeVectorStore(
#    index_name="langchain-docs-2025", embedding=embeddings
#)
#vectorstore = Pinecone.from_existing_index(index_name="langchain-docs-2025", embedding=embeddings)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl: TavilyCrawl = TavilyCrawl()

def chunk_urls_v2(urls: List[str], chunk_size: int) -> List[str]:
    list_of_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    return list_of_chunks

def chunk_urls(urls: List[str], chunk_size: int) -> List[str]:
    """Chunks urls into chunks of size chunk_size"""
    chunks: list[Any] = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)

    return chunks

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Processes documents asynchronously in batches"""
    log_header("VECTOR STORAGE PAGE")

    log_info(
        "^^ VectorStore Indexing: Preparing to add {len(documents)} to VectorStore ^^",
        Colors.PURPLE
    )

    #Create Batches
    batches = [
        documents[i: i+batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"VectorStore Indexing Split into {len(batches)} batches of {batch_size} documents each ^^")

    #Processall the batches asynchronously
    async def add_batch(batch: List[Document], batch_num: int):
        for attempt in range(5):
            try:
                await vectorstore.aadd_documents(batch)
                log_success(f"VectorStore Indexing: Successfully added batch number {batch_num} / {len(batches)} ({len(batch)} documents to VectorStore ^^")
                return True
            except Exception as e:
                log_error(f"Failed to add batch number {batch_num} - {e}")
                if attempt == 4:
                    raise
                    # small backoff
                await asyncio.sleep(0.5 * (2 ** attempt))

        return False


    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def extract_batches(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extracts batches from urls"""
    try:
        log_info(f"Tavily Extract: Processing {batch_num} with {len(urls)}")
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"Tavily Extract: Completed batch {batch_num} - extracted with {len(docs.get('results', []))} documents")
        return docs
    except Exception as e:
        log_error(f"Tavily Extract: Failed to extract batch {batch_num} - {e}")
        return[]

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACT PHASE")
    log_info(f"Tavily Extract: Starting concurrent extraction of {len(url_batches)} batches", Colors.DARKCYAN)

    tasks = [extract_batches(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #filter out exceptions and flaws
    all_pages = []
    failed_batches =0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"Travily Extract: Batch failed with exception: {result}")
            failed_batches += 1
        else:
            for extracted_pages in result["results"]:
                document = Document(page_content=extracted_pages["raw_content"],
                                    metadata={"source": extracted_pages["url"]},)
                all_pages.append(document)

    log_success(f"Tavily Extract: Extraction complete! Total pages extracted: {len(all_pages)}")

    if failed_batches > 0:
        log_warning(f"Tavily Extract: Batch failed with errors: {failed_batches}")

    return all_pages

async def tavily_crawl_v1():
    """Main function to orchestrate the execution of the program"""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "^^ Travily Crawl: Starting to Crawl documentation from https://python.langchain.com ^^",
        Colors.PURPLE
    )
    res = tavily_crawl.invoke(
        input={
            "url" : "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
            "instructions": "content on ai agents"
        }
    )
    all_docs = res["results"]
    all_docs = [Document(page_content=result["raw_content"], metadata={"source":result["url"]}) for result in res["results"]]
    log_success( f"Tavily Crawl: Successfully crawled {len(all_docs)} URL's from documentation site" )

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Split: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Split: Successfully split {len(splitted_docs)} chunks from {len(all_docs)} documents")

    # Process Document Asymchronosly
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(res)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

async def tavily_map_v1():
    """Main function to orchestrate the execution of the program"""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "^^ Travily Map: Starting to Crawl documentation from https://python.langchain.com ^^",
        Colors.PURPLE
    )
    sitemap = tavily_map.invoke("https://python.langchain.com/")

    log_success(f"Tavily Map: Successfully crawled {len(sitemap['results'])} URL's from documentation site")
    url_batches = chunk_urls_v2(list(sitemap['results']), 20)
    log_info(f"URL Processing Split: {len(sitemap['results'])} URLs Processed to {len(url_batches)} batches")

    all_docs = await async_extract(url_batches)

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Split: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Split: Successfully split {len(splitted_docs)} chunks from {len(all_docs)} documents")

    # Process Document Asymchronosly
    await index_documents_async(splitted_docs, batch_size=250)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs mapped: {len(sitemap['results'])}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(tavily_map_v1())


