import os

from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter # To chunkify the HTML loaded documents
from langchain_community.document_loaders import ReadTheDocsLoader
# The LangChain documentation was written using ReadTheDocs. Hence we load the LangChain documentation using the
# ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# In the above line of code, we don't pass the OPENAI_API_KEY because the OpenAI library automatically fetches the
# API_KEY from the loaded environment variables.
# While defining in the .env file, we just have to name it properly as 'OPENAI_API_KEY'.


def ingest_docs():
    #loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    # loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest", encoding="ISO-8859-1")
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")

    # The 'latest' folder contains many folders and many html links (say type A).
    # The folders (which are inside the 'latest' folder) also contain many html links (say type B).
    # ReadTheDocsLoader() loads both the html links type A and type B into the loader
    # The ReadTheDocsLoader handles both by recursively navigating through the folder structure and loading all
    # relevant HTML files.
    raw_documents = loader.load()
    # raw_documents is a list of n LangChain documents. The original pdf document has been broken down into
    # n LangChain documents.
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    # CharacterTextSplitter divides text into chunks purely based on character counts without considering the
    # natural structure of the text. Suitable when we need a simple and fast splitting mechanism where
    # precise control over chunk size is needed, but where the context of the text isn't as important.

    # RecursiveTextSplitter is a more sophisticated splitter that attempts to split the text at natural breakpoints
    # while still trying to stay within the chunk_size. It first tries to split text into chunks by paragraphs.
    # If a paragraph is larger than chunk_size=600, it will then attempt to split by sentences, ensuring that
    # chunks are as coherent as possible without exceeding the token limits.

    # Note
    # Total number of tokens for an API call = Total tokens in Input Prompt + Total tokens output by LLM.
    # The rule of thumb is to send at most 4-5 contexts (chunks) as the final complete context
    # Suppose we are reserving 2000 tokens for the context. Each chunk is then of 500 tokens.
    # So chunk_size should be around 500 tokens. This is how we determine chunk_size.
    # If the chunk_size was very very small, then the relevant chunks picked up by the retriever and when these
    # relevant small chunks are sent to the LLM, the LLM may not have the general broad context of these chunks.
    # Like from which general broad context, these chunks were taken from.
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        new_url = new_url.replace("\\", "/")
        doc.metadata.update({"source": new_url})

    # We have web scraped the langchain docs in a very structured manner such that there is api.python.langchain.com
    # folder. Inside which there is 'en' folder. Inside which there is 'latest' folder and inside which are the
    # html links and various other folders.
    # Since it has been web scraped in the above structured manner, we just have to replace "langchain-docs" with
    # "https:/" in doc.metadata['source'] and we have the live html web links.

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


def ingest_docs2() -> None:
    from langchain_community.document_loaders import FireCrawlLoader
    # When we use FireCrawl, the chunks of data indexed to our VectorStore will be of higher quality.
    # Firecrawl can both scrape and web crawl the pages efficiently. It removes all html tags in the process and hence
    # the data is of higher quality.
    # The content crawled by FireCrawl is very readable and very ingestible to a vector store.

    langchain_documents_base_urls = [
        "https://python.langchain.com/v0.2/docs/integrations/chat/",
        "https://python.langchain.com/v0.2/docs/integrations/llms/",
        "https://python.langchain.com/v0.2/docs/integrations/text_embedding/",
        "https://python.langchain.com/v0.2/docs/integrations/document_loaders/",
        "https://python.langchain.com/v0.2/docs/integrations/document_transformers/",
        "https://python.langchain.com/v0.2/docs/integrations/vectorstores/",
        "https://python.langchain.com/v0.2/docs/integrations/retrievers/",
        "https://python.langchain.com/v0.2/docs/integrations/tools/",
        "https://python.langchain.com/v0.2/docs/integrations/stores/",
        "https://python.langchain.com/v0.2/docs/integrations/llm_caching/",
        "https://python.langchain.com/v0.2/docs/integrations/graphs/",
        "https://python.langchain.com/v0.2/docs/integrations/memory/",
        "https://python.langchain.com/v0.2/docs/integrations/callbacks/",
        "https://python.langchain.com/v0.2/docs/integrations/chat_loaders/",
        "https://python.langchain.com/v0.2/docs/concepts/",
    ]
    # The above is a list of base urls.

    langchain_documents_base_urls2 = [langchain_documents_base_urls[1]]
    # 'langchain_documents_base_urls2' contains only one element of the list for demo purpose.
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "limit": 5,
            },
        )
        docs = loader.load()
    # Note that the 'docs' is not further split by text_splitter because the chunks of text we get from
    # FireCrawLoader is suitable to be directly uploaded to the VectorStore.

    # 'docs' will hold the LangChain documents for that URL only, and not accumulate documents from previous iterations.
    # In each iteration, each url is broken down into n langchain documents and uploaded to vector store.
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print(f"****Loading {url}* to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
