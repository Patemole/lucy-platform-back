
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import logging
import langchain_pinecone
from pinecone import Pinecone, Index
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from dotenv import load_dotenv


############################################################
#ENVIRONMENT VARIABLES
############################################################
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_ACADEMIC_ADVISOR")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

PINECONE_POOL_THREADS = 4


############################################################
#INITIALISATION DE L'API GROQ WITH LLAMA3
############################################################
model = "llama3-70b-8192"
# Initialize Groq Langchain chat object and conversation
GROQ_LLM = ChatGroq(
        model_name=model,
        streaming=True
)


logging.basicConfig(level=logging.INFO)



############################################################
#INITIALISATION DE L'API PINECONE AVEC L'INDEX ACADEMIC-ADVISOR-UPENN-TEST1
############################################################

class PineconeApiClient:
    _instance = None

    def _new_(cls, *args, **kwargs):
        if not cls._instance:
            logging.info("Initialisation de l'instance PineconeApiClient...")
            cls.instance = super(PineconeApiClient, cls).new_(cls)
            try:
                print(PINECONE_API_KEY)
                pinecone = Pinecone(api_key=PINECONE_API_KEY)
                cls.index = pinecone.Index("academic-advisor-upenn-test1", pool_threads=PINECONE_POOL_THREADS)
                text_embeddings = OpenAIApiClient().text_embeddings
                cls.vectorstore = langchain_pinecone.Pinecone(cls.index, text_embeddings, "text")
                logging.info("PineconeApiClient a été initialisé avec succès.")
            except Exception as e:
                logging.error(f"Erreur lors de l'initialisation de PineconeApiClient: {e}")
                raise e
        return cls._instance
    

############################################################
#INITIALISATION DE L'API OPENAI
############################################################

class OpenAIApiClient:
    _instance = None

    def _new_(cls, *args, **kwargs):
        if not cls._instance:
            cls.instance = super(OpenAIApiClient, cls).new_(cls)
            cls.text_embeddings = OpenAIEmbeddings(
                openai_api_key = OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            cls.open_ai_client = OpenAI(api_key= "OPENAI_API_KEY")
            

        return cls._instance

############################################################
# GRAPH'S LOGIC
############################################################
# find  methodology for the question test2
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
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    USER_INPUT: {user_input} \n\n
    MEMORY: {memory} \n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    assistant
    """,
    input_variables=["user_input", "memory"],
)

rag_keywords_chain = rag_keywords_prompt | GROQ_LLM | JsonOutputParser()

def invoke_chain_rag_keywords(data):
    return rag_keywords_chain.invoke(data)

'''
def query_pinecone(user_input, class_id, keywords):
    # Generate the enhanced query using user input and keywords
    query = generate_enhanced_query(user_input, keywords)
    #print(query)
    
    # Generate the query embedding NOT NEEDED
    #client = OpenAIApiClient()
    #query_embedding = client.text_embeddings.embed_query(query)
    
    # Define the Pinecone client
    #index = PineconeApiClient().index
    
    #TODO make sure about the format of the class name without spaces and upper letters

    # Define the metadata filter
    filter = {"class_id": class_id}

    # Parse the retrieved documents
    retrieved_docs : List[Document] = PineconeApiClient().vectorstore.similarity_search(query=query, k=3, filter=filter)
    
    return retrieved_docs
'''


def query_pinecone(user_input, class_id, keywords):
    # Génère la requête améliorée en utilisant l'entrée utilisateur et les mots-clés
    query = generate_enhanced_query(user_input, keywords)

    # Définir le filtre de métadonnées
    filter = {"class_id": class_id}

    # Récupérer les documents
    retrieved_docs : List[Document] = PineconeApiClient().vectorstore.similarity_search(query=query, k=3, filter=filter)
    #retrieved_docs = PineconeApiClient().vectorstore.similarity_search(query=query, k=3, filter=filter)
    
    return retrieved_docs


def generate_enhanced_query(user_input, keywords):
    combined_query = user_input + " " + " ".join(keywords)
    return combined_query


def rag_info(data):
    responses = query_pinecone(data)
    return responses
        
## Draft socratic answer using the information found and the methodology
draft_answer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
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


    <|eot_id|><|start_header_id|>user<|end_header_id|>
    USER_INPUT: {user_input} \n\n
    MEMORY: {memory} \n\n
    RAG_KEYWORD: {rag_keywords} \n\n
    CLASS_NAMES: {class_names} \n\n
    RAG_INFO: {rag_info}
    
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    assistant""",
    input_variables=["user_input", "rag_info", "rag_keywords", "class_names", "memory"],
)
 # METHODOLOGY: {methodology} \n\n
drat_answer_chain = draft_answer_prompt | GROQ_LLM | StrOutputParser()

# @timing_decorator
def invoke_chain_draft_answer(data):
    return drat_answer_chain.ainvoke(data)


### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    user_input : str
    class_names: List[str]
    rag_keywords: List[str]
    rag_info: str
    draft_answer : str
    memory : List[str]
    
    

def state_printer(state):
    """print the state"""
    print("---STATE PRINTER---")
    print(f"user_input: {state['user_input']} \n" )
    print(f"memory: {state['memory']} \n")
    print(f"class_names: {state['class_names']} \n")
    print(f"rag_keywords: {state['rag_keywords']} \n")
    print(f"rag_info: {state['rag_info']} \n")
    print(f"draft_answer: {state['draft_answer']} \n")
    
    
    return


### Nodes

def user_input_question(state):

    print("---USER INPUT---")
    
    #get chat memory from user and put it into "memory"
    
    user_input = state["user_input"]
    memory = state["memory"]
    
    return {"user_input": user_input, "memory": memory}

def rag_keywords_search(state):
    """
    Take the user_input and find the methodology and guidance.
    """
    user_input = state["user_input"]
    memory = state["memory"]

    rag_info_retrieved = invoke_chain_rag_keywords({"user_input": user_input, "memory": memory})

    class_names = rag_info_retrieved["class_names"]
    keywords = rag_info_retrieved["keywords"]

    return {"class_names": class_names, "rag_keywords": keywords}


'''
async def draft_answer_generation(state):
    """
    Using RAG to get the info.
    """

    user_input = state["user_input"]
    memory = state["memory"]

    # Retrieve keywords and class names using rag_keywords
    class_id = state["class_names"][0]  # Assuming you want to use the first class name
    keywords = state["rag_keywords"]

    # Query Pinecone with metadata filter and enhanced query
    rag_info = query_pinecone(user_input, class_id, keywords)

    # Invoke the chain with the retrieved Pinecone information
    draft_answer = await invoke_chain_draft_answer({"user_input": user_input, "rag_keywords":keywords, "rag_info": rag_info, "class_names":class_id, "memory":memory})
    
    return {"draft_answer": draft_answer, "rag_info": rag_info}
    #return {"draft_answer": draft_answer}
'''

async def draft_answer_generation(state):
    """
    Utilisation de RAG pour obtenir les informations.
    """

    user_input = state["user_input"]
    memory = state["memory"]

    # Debug: Print l'état actuel
    print("Entrée dans draft_answer_generation")
    print(f"user_input: {user_input}")
    print(f"memory: {memory}")

    # Vérifiez si class_names n'est pas vide avant d'accéder
    if state["class_names"]:
        class_id = state["class_names"][0]  # Utilisez le premier nom de classe si disponible
    else:
        # Gérez le cas où aucun nom de classe n'est disponible
        class_id = None  # Ou gérer de manière appropriée
    
    keywords = state["rag_keywords"]

    # Interrogez Pinecone avec le filtre de métadonnées et la requête améliorée
    rag_info = query_pinecone(user_input, class_id, keywords)

    # Invoquez la chaîne avec les informations récupérées de Pinecone
    draft_answer = await invoke_chain_draft_answer({
        "user_input": user_input,
        "rag_keywords": keywords,
        "rag_info": rag_info,
        "class_names": class_id,
        "memory": memory
    })
    
    return {"draft_answer": draft_answer, "rag_info": rag_info}

# ## Build the Graph

workflow = StateGraph(GraphState)


### Add Nodes


# Define the nodes
workflow.add_node("user_input_question", user_input_question) # starting node get the user input 
workflow.add_node("draft_answer_generation", draft_answer_generation) # find methodology and guide the user
workflow.add_node("rag_keywords_search", rag_keywords_search) # retrieve the info from the rag
#workflow.add_node("state_printer", state_printer) # print state



# ### Add Edges


workflow.set_entry_point("user_input_question")


workflow.add_edge("user_input_question", "rag_keywords_search")
workflow.add_edge("rag_keywords_search", "draft_answer_generation")
workflow.add_edge("draft_answer_generation", END)
#workflow.add_edge("state_printer", END)


# Compile
app = workflow.compile()

#nest_asyncio.apply()



############################################################
#GENERATION OF THE ANSWER
############################################################
async def academic_advisor_answer_generation(user_input, chat_history):
    #onfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    current_node = ""


    #Stream generation
    async for event in app.astream_events(
        {"user_input": user_input},
        {"memory": chat_history},
        #config=config,
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



