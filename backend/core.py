from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from typing import Any, Dict, List

from langchain import hub # For downloading the augmentation prompts
from langchain.chains.combine_documents import create_stuff_documents_chain
# The create_stuff_documents_chain takes the prompt template and llm as parameters.
# It receives the relevant documents (context) from the retriever. It plugs in the context into the prompt template
# and sends them to the llm
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    # docsearch is going to perform as our retriever.
    chat = ChatOpenAI(verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    # The chat-langchain-rephrase prompt is as below

    # Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    # Chat History:
    # {chat_history}
    # Follow Up Input: {input}
    # Standalone Question:

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # The retrieval-qa-chat-prompt is as below

    # Answer any use questions based solely on the context below:
    # <context>
    # {context}
    # </context>
    # PLACEHOLDER
    # chat_history
    # HUMAN
    # {input}

    # The PLACEHOLDER chat_history suggests that there is a placeholder for chat history, but it is not being
    # directly utilized in this version of the prompt.

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    # The above is Augmentated Generation

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    # We pass the llm parameter to the 'history_aware_retriever' because we can use a different LLM to perform the
    # rephrasing task. The LLM passed to history_aware_retriever doesn't necessarily have to be the same LLM used for
    # generating the final answer. More the flexibility, hence we pass the llm as a parameter.

    # The history_aware_retriever rephrases the follow-up question into a standalone question using the llm and
    # the rephrase_prompt.
    # The history_aware_retriever retrieves relevant documents based on the standalone question.
    # The history_aware_retriever sends the relevant documents and standalone question to the combine_docs_chain.

    # When the chat_history is empty, the followup question and rephrase_prompt is sent to llm. But the llm doesn't
    # rephrase. The standalone question remains the same as the follow up question.

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    # The 'combine_docs_chain' may put all retrieved documents into one string or may summarize all the retrieved
    # documents into a string or any other use case.

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    # result is a dictionary which contains 3 keys namely 'input', 'context' and 'answer'.

    # The input dictionary uses specific key names because 'input' is present as a variable in
    # retrieval_qa_chat_prompt and 'chat_history' is present as a variable in rephrase_prompt.

    # The retrieval_qa_chat_prompt also expects a variable called context, which is the string that contains all the
    # retrieved documents.
    # The context documents are retrieved by the retriever, and then they are combined into a single string by the
    # combine_docs_chain.
    # Once the documents are combined into a single context string, the chain automatically plugs this string into
    # the 'context' variable of the retrieval_qa_chat_prompt.
    # This is done by the chain’s internal logic, which knows how to map the output of the
    # combine_docs_chain (i.e., the combined context string) to the correct placeholder (context) in the prompt template.

    return result

    # Note 1
    # # The rephrasing step converts the follow-up question into a standalone question that no longer relies on the
    # past conversation.
    # By the time the LLM generates the final answer, it only needs the rephrased standalone question and the
    # retrieved documents, so there’s no need to pass the chat_history again to the final retrieval_qa_chat_prompt.

    # Note 2 (Very good observation)

    # The first question asked was - "What is LangChain?"
    # The llm answered it correctly by using the concept of RAG.
    # The second question asked was - "What did I just ask you?"
    # I checked the LangSmith traces. The second follow up question has been converted into the
    # standalone question - "What is LangChain?"
    # Though both the api calls are stateless and independent, yet the LLM came up with an answer that
    # 'You just asked me about LangChain and explain LangChain".
    # The standalone question was simply "What is LangChain" and we are not passing the chat_history to the
    # retrieval_qa_chat_prompt and still the LLM began the answer with 'You just asked me about so and so...."

    # How?

    # Even though the api calls are independent of each other, since I am making the api calls in the same session,
    # the LLM is able to sense the pattern (maybe based on it's trained data) that the standalone question is
    # actually a follow up question and is equivalent to - 'What is LangChain and What did I ask you?'

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rag_chain = (
            {
                "input": ({"input": query, "chat_history": chat_history} | rephrase_prompt),
                "context": (docsearch.as_retriever() | format_docs)
            }
            | retrieval_qa_chat_prompt
            | chat
            | StrOutputParser()
    )

    # Note - The mistake in the above rag_chain is that the retriever is not being provided with the rephrased query.
    # It lacks any input to perform a similarity search.
    # Without specifying the input query, the retriever may not function correctly or may use a default value,
    # leading to irrelevant document retrieval.

    # The below is an equivalent chain of the above rag_chain (which has a mistake though)

    # rag_chain = (
    #         {
    #             "input": ({"input": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | rephrase_prompt),
    #             "context": (docsearch.as_retriever() | format_docs)
    #         }
    #         | retrieval_qa_chat_prompt
    #         | chat
    #         | StrOutputParser()
    # )

    # Basically what RunnablePassthrough() does is that it maps the values passed during the invoking of the
    # chain to the values in the chain setup without any transformation.

    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result

def run_llm2_corrected(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Rephrase the query
    rephrased_input = ({"input": query, "chat_history": chat_history} | rephrase_prompt)

    # Retrieve documents based on the rephrased query
    retrieved_context = (
        rephrased_input
        | docsearch.as_retriever()
        | format_docs
    )
    # The 'retrieved_context' is also a string (sll retrieved documents put together as one string)

    rag_chain = (
        {
            "input": rephrased_input,
            "context": retrieved_context
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result


def run_llm3(query: str, chat_history: List[Dict[str, Any]] = []):
    # Initialize embeddings and document retriever
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Initialize the LLM (gpt-4)
    chat = ChatOpenAI(model_name="gpt-4", verbose=True, temperature=0)

    # Load rephrase and retrieval QA prompts
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Rephrasing the query based on chat history
    rephrased_input_chain = {
        "input": {"input": query, "chat_history": chat_history} | rephrase_prompt
    }
    # 'rephrased_input_chain' is a dictionary with the 'input' key value = 'standalone_question'

    # Document retrieval chain (retrieve documents based on the rephrased query)
    retrieve_docs_chain = rephrased_input_chain | (lambda x: x["input"]) | docsearch.as_retriever() | format_docs
    # In the above line of code, 'x' refers to the output of rephrased_input_chain.
    # retrieve_docs_chain is nothing but 'context' (retrieved documents combined and put together).
    # The documents retrieved are based on the 'standalone' question.

    # Build the final RAG chain with the rephrased input and retrieved context
    rag_chain = (
            {
                "input": rephrased_input_chain["input"],  # standalone_question
                "context": retrieve_docs_chain  # 'context' (retrieved documents combined and put together)
            }
            | retrieval_qa_chat_prompt
            | chat
            | StrOutputParser()
    )

    # Run the chain
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result

    # The below function 'run_llm4' is a more simplified version of 'run_llm3'

def run_llm4(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Rephrase the query based on chat history
    rephrased_query = ({"input": query, "chat_history": chat_history} | rephrase_prompt)

    # Retrieve documents using the rephrased query
    retrieved_context = (
        rephrased_query
        | docsearch.as_retriever()
        | format_docs
    )

    # Build the RAG chain with rephrased input and retrieved context
    rag_chain = (
        {
            "input": rephrased_query,       # Rephrased standalone question
            "context": retrieved_context    # Retrieved documents formatted as string
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    # Run the chain
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result


if __name__ == "__main__":
    result = run_llm("What is LangChain?", [])
    print(result)
