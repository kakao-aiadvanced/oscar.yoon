import os
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from typing import List
from tavily import TavilyClient

def load_from_web(urls) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    return docs

def get_relevant_docs(query: str, llm: ChatOpenAI, retrieved_docs: List[Document]) -> List[Document]:
    prompt_template = """
    You are an expert evaluator designed to assess the relevance of retrieved text chunks based on a user's query. 
    Your task is to determine whether the retrieved chunk of text is relevant to the user's query. 
    Relevance is defined as the degree to which the retrieved chunk answers, provides context, or directly relates to the user's query.

    Please provide your assessment in a valid JSON format, as a single string. Your response should be exactly one of the following:
    - If the retrieved chunk is relevant to the user's query, respond with: {{"relevance": "yes"}}.
    - If the retrieved chunk is not relevant to the user's query, respond with: {{"relevance": "no"}}.

    Do not include any other text or explanation in your response.

    User Query: {query}
    Retrieved Chunk: {retrieved_chunk}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "retrieved_chunk"]
    )

    def evaluate_relevance(query: str, retrieved_chunk: str):
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"query": query, "retrieved_chunk": retrieved_chunk})

        return result

    relevant_docs = []

    for doc in retrieved_docs:
        result = evaluate_relevance(query, doc.page_content)
        if result["relevance"] == "yes":
            relevant_docs.append(doc)

    return relevant_docs

def get_answer(llm: ChatOpenAI, query: str, context: str) -> str:
    prompt_template = """
                Based on the following context, please answer the user's question. If you have a source in the form of a URL, include it as well. :

                Context: {context}

                User Question: {query}

                Answer:
                """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query, "context": context})

    return result

def check_validation(llm: ChatOpenAI, context: str, answer: str) -> bool:
    prompt_template = """
    Given the context below and the answer provided, determine if the answer is fully supported by the context or if there is any hallucination or unsupported information.

    Context: {context}

    Answer: {answer}

    If the answer is fully supported by the context, respond with: {{"hallucination": "no"}}.
    If there is any hallucination or unsupported information, respond with: {{"hallucination": "yes"}}.

    Response:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "answer"]
    )

    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({"context": context, "answer": answer})

    return True if result["hallucination"] == "yes" else False

def retrieve(query: str, llm: ChatOpenAI, vectorstore: Chroma):
    result = vectorstore.similarity_search(query)
    return get_relevant_docs(query, llm, result)

def search(query: str, llm: ChatOpenAI, tavily: TavilyClient):
    response = tavily.search(query=query, max_results=3)
    result = [Document(page_content=obj["content"], metadata={"source": obj["url"]}) for obj in response['results']]
    return get_relevant_docs(query, llm, result)

def print_log(log: str):
    print(log)
    print("---------------------------")

if __name__ == '__main__':
    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        print_log(f"openai_api_key: {openai_api_key}")

        if openai_api_key is None:
            raise Exception("openai_api_key is invalid")

        tavily_api_key = os.environ["TAVILY_API_KEY"]
        print_log(f"tavily_api_key: {tavily_api_key}")

        if tavily_api_key is None:
            raise Exception("tavily_api_key is invalid")

    except Exception as e:
        print(f"error: {e}")
        exit(-1)

    if not os.path.exists("./data"):
        os.mkdir("./data")

    vectorstore = Chroma(
        persist_directory="./data/chromadb",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    if vectorstore._collection.count() == 0:
        urls = (
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        )

        docs = load_from_web(urls)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vectorstore.add_documents(splits)

    llm = ChatOpenAI(model="gpt-4o-mini")
    tavily = TavilyClient(api_key=tavily_api_key)

    while True:
        query = input("query: ")
        if query == "quit":
            break

        print_log("")

        current_task = 'retrieve'
        relevant_docs = []
        answer = ''
        search_count = 0
        hallucination_count = 0

        while True:
            try:
                print_log(current_task)

                if current_task == 'retrieve':
                    relevant_docs = retrieve(query, llm, vectorstore)
                    if len(relevant_docs) == 0:
                        current_task = 'search'
                    else:
                        current_task = 'get_answer'
                elif current_task == 'search':
                    # Allow only one search
                    if search_count > 0:
                        print_log(f"Already searched: {search_count}")
                        break

                    search_count += 1
                    relevant_docs = search(query, llm, tavily)
                    if len(relevant_docs) == 0:
                        current_task = 'retrieve'
                    else:
                        current_task = 'get_answer'
                elif current_task == 'get_answer':
                    context = "\n".join([f"source: {doc.metadata['source']}, content: {doc.page_content}" for doc in relevant_docs])
                    answer = get_answer(llm, query, relevant_docs)
                    print_log(f"answer: \n{answer}")

                    result = check_validation(llm, context, answer)
                    print_log(f"Hallucination check: {result}")

                    # Hallucination exists, retry answer
                    if result:
                        # Hallucination is allowed up to three times.
                        if hallucination_count > 2:
                            break

                        hallucination_count += 1
                        print_log(f"Hallucination count: {hallucination_count}, retry answer")
                        continue
                    else:
                        break
            except Exception as e:
                print_log(f"error: {e}")
                continue