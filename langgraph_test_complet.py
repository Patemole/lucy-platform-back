#To be depreciated
import logging
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import langchain_pinecone
import os
from pinecone import Pinecone, Index
import langchain
from typing import List
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from langchain.schema import Document
import asyncio

# Load environment variables from .env file
load_dotenv()


# Environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_ACADEMIC_ADVISOR")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
PINECONE_POOL_THREADS = 4


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialization of the Groq LLM
from langchain_groq import ChatGroq
model = "llama3-70b-8192"
GROQ_LLM = ChatGroq(model_name=model, streaming=True)

# Pinecone API Client
class PineconeApiClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PineconeApiClient, cls).__new__(cls)
            pinecone = Pinecone(api_key=PINECONE_API_KEY, environment="gcp-starter")
            cls.index = pinecone.Index("academic-advisor-upenn-test1", pool_threads=PINECONE_POOL_THREADS)
            text_embeddings = OpenAIApiClient().text_embeddings
            cls.vectorstore = langchain_pinecone.Pinecone(cls.index, text_embeddings, "text")
        return cls._instance

# OpenAI API Client
class OpenAIApiClient:
    _instance = None

    def __init__(self):
        self.text_embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        self.open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

    def get_langchain_open_ai_api_client(self):
        return langchain.OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Query Pinecone
def query_pinecone(user_input, class_id, keywords):
    query = generate_enhanced_query(user_input, keywords)
    filter = {"class_id": class_id}
    retrieved_docs: List[Document] = PineconeApiClient().vectorstore.similarity_search(query=query, k=3, filter=filter)
    return retrieved_docs

def generate_enhanced_query(user_input, keywords):
    combined_query = user_input + " " + " ".join(keywords)
    return combined_query

# RAG Keywords Chain
rag_keywords_prompt = PromptTemplate(
    template="""You are an expert academic advisor for university students, you are a master at answering question from student about classes informations and details helping them choosing their classes. Your task is to analyze the user's input to identify the class or classes being referred to and to extract three keywords that are highly relevant to the student's query.

    Context:
    You will be provided with a student's query (USER_INPUT) and the conversation history (MEMORY). Your goal is to:

    Identify the class or classes the student is asking about questions about.
    Extract three highly relevant keywords from the query to help filter out the class content for further information retrieval. The keywords must be about the topic and the area the student is looking an info for.
    The lastest user message is USER_INPUT but take the MEMORY as a context to understand what was the student referring to.
    If no class is mentioned directly in the USER_INPUT refer to the last one mentioned in MEMORY.
    Instructions:
    Analyze the query (USER_INPUT) and MEMORY to understand the context and the specific needs of the student.
    Identify the class or classes the query pertains to. Consider any references to course names, ids, codes.
    Extract keywords max 5, that are most relevant to the student's query about the class. These keywords should help in filtering the class syllabus content accurately. So highlight the main thing the question is about.
    Example:
    USER_INPUT: "What kind of assessment will there be in my DATA2002 class?"

    Analysis:

    Class Identified: DATA2002 (this must absolutely upper case letters and no space just like in the example DATA2002)
    Keywords: assessment method type 
    Additional Information:
    Class Identifiers: Course names, codes, or specific topics mentioned in the query.
    Keyword Relevance: Keywords should directly relate to the subject of the question issue the student is inquiring about.
    Evaluation Criteria:
    Accuracy: Correct identification of the class or classes and relevant keywords.
    Clarity: The clarity of the identified class and extracted keywords.
    Call to Action:
    
    You must absolutely return a JSON with the "class_names" with the list of classes mentioned and then "keywords" no premable or explaination of the following above:
    
    Analyze the following user query and provide the class or classes along with three relevant keywords and return a JSON:
    
    Make sure that you are returning a JSON never return a string or something else than a JSON
    user
    USER_INPUT: {user_input} \n\n
    MEMORY: {memory} \n\n
    assistant
    
    assistant
    """,
    input_variables=["user_input", "memory"],
)

rag_keywords_chain = rag_keywords_prompt | GROQ_LLM | JsonOutputParser()

def invoke_chain_rag_keywords(data):
    return rag_keywords_chain.invoke(data)

# Draft Answer Chain
draft_answer_prompt = PromptTemplate(
    template="""system
    You are an expert academic advisor for university students with over 20 years of experience. Your task is to draft a complete and accurate answer to a student's question by analyzing the provided information.

    Given the USER_INPUT (the student question), the CLASS_NAMES (the class IDs), the RAG_KEYWORD (keywords describing the student question topic), the RAG_INFO (the information retrieved from the syllabus of the class the info is per class), and the MEMORY (the history of the conversation), your goal is to:

    Identify the class or classes being discussed based on USER_INPUT and MEMORY.
    If no class is mentioned directly in the USER_INPUT refer to the last one mentioned in MEMORY.
    Analyze the RAG_INFO and extract the relevant information needed to answer the USER_INPUT, considering that it is in accordance with the CLASS_NAME and RAG_KEYWORD.
    Draft a concise and accurate answer to the USER_INPUT using the relevant information.
    Instructions:
    Identify the class or classes: Determine the class being referred to in the USER_INPUT and MEMORY using the CLASS_NAMES.
    Extract relevant information: From the RAG_INFO, identify and extract only the information that is for the correct class and pertinent to the USER_INPUT, and RAG_KEYWORD.
    Draft the answer: Combine the relevant information into a coherent and accurate response to the USER_INPUT. Ensure the answer is clear and addresses the student's query directly.
    Tone to use: Be friendly like you are talking to a friend but stay polite but don't be formal. 
    Example Format:
    USER_INPUT: "How will I be assessed in ECON0100"

    CLASS_NAME: "ECON0100"
    RAG_KEYWORD: "'assessment', 'evaluation', 'grading'"
    RAG_INFO: "legal framework; control and culture of the modern corporation; operations of a Board; role of board sub-committees; Boards and the development or endorsement of strategies; measuring and rewarding performance; corporate governance, financial reporting and disclosure; corporate governance and the audit process; governance within the global financial crisis.
    Assessment Overview:
    The key components include:
    - Final exam (Take-home short release) accounting for 40% of the grade.
    - Small test (Online open book without invigilation) accounting for 10% of the grade.
    "

    MEMORY: "Student previously asked about basic information about ECON0100."

    Drafted Answer:
    In the ECON0100 class you will be assessed by a final exam which account for the majority of the class, but that's not all there will also be small tests (Online open book without invigilation) throughout the semester from weeks to weeks. Let me know if you any more info? 


    user
    USER_INPUT: {user_input} \n\n
    MEMORY: {memory} \n\n
    RAG_KEYWORD: {rag_keywords} \n\n
    CLASS_NAMES: {class_names} \n\n
    RAG_INFO: {rag_info}
    
    assistant
    
    assistant""",
    input_variables=["user_input", "rag_info", "rag_keywords", "class_names", "memory"],
)

draft_answer_chain = draft_answer_prompt | GROQ_LLM | StrOutputParser()

async def invoke_chain_draft_answer(data):
    return await draft_answer_chain.ainvoke(data)

# State Management
class GraphState(TypedDict):
    user_input: str
    class_names: List[str]
    rag_keywords: List[str]
    rag_info: str
    draft_answer: str
    memory: List[str]

def user_input_question(state):
    user_input = state["user_input"]
    memory = state["memory"]
    return {"user_input": user_input, "memory": memory}

def rag_keywords_search(state):
    user_input = state["user_input"]
    memory = state["memory"]
    rag_info_retrieved = invoke_chain_rag_keywords({"user_input": user_input, "memory": memory})
    class_names = rag_info_retrieved["class_names"]
    keywords = rag_info_retrieved["keywords"]
    return {"class_names": class_names, "rag_keywords": keywords}

async def draft_answer_generation(state):
    user_input = state["user_input"]
    memory = state["memory"]
    class_id = state["class_names"][0] if state["class_names"] else None
    keywords = state["rag_keywords"]
    rag_info = query_pinecone(user_input, class_id, keywords)
    draft_answer = await invoke_chain_draft_answer({
        "user_input": user_input,
        "rag_keywords": keywords,
        "rag_info": rag_info,
        "class_names": class_id,
        "memory": memory
    })
    return {"draft_answer": draft_answer, "rag_info": rag_info}

# Build the Graph
workflow = StateGraph(GraphState)
workflow.add_node("user_input_question", user_input_question)
workflow.add_node("rag_keywords_search", rag_keywords_search)
workflow.add_node("draft_answer_generation", draft_answer_generation)
workflow.set_entry_point("user_input_question")
workflow.add_edge("user_input_question", "rag_keywords_search")
workflow.add_edge("rag_keywords_search", "draft_answer_generation")
workflow.add_edge("draft_answer_generation", END)
app = workflow.compile()

# Generation of the Answer
async def academic_advisor_answer_generation(user_input, chat_history):
    current_node = ""
    async for event in app.astream_events(
        {"user_input": user_input},
        {"memory": chat_history},
        stream_mode="updates",
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chain_start" and event["name"] == 'draft_answer_generation':
            current_node = "final_answer"
        if kind == "on_chat_model_stream" and current_node == "final_answer":
            content = event["data"]["chunk"].content
            if content:
                yield content + "|"

# Test the academic advisor answer generation
async def main():
    user_input = "What courses for computer science?"
    chat_history = []  # Assuming no prior chat history for this example

    async for response in academic_advisor_answer_generation(user_input, chat_history):
        print(response, end='')

if __name__ == "__main__":
    asyncio.run(main())
