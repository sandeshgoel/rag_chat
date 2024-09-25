from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

import dotenv
import logging
import sys

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    )
log = logging.getLogger()

dotenv.load_dotenv()
db_dir = 'chromadb'

# load vectorDB
vectorstore = Chroma(
    embedding_function=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    persist_directory=db_dir,
    collection_name="rag-chroma",
)

num_docs = len(vectorstore.get('documents',[]))
log.info(f'Number of docs in db: {num_docs}')
if num_docs == 0:
    log.error('No documents in database!!!')
    sys.exit(1)
retriever = vectorstore.as_retriever()

log.info('retriever generated - testing it ...')

# Test retriever
question = "financial plan"
docs = retriever.invoke(question)

log.info(f'Question:\n\n{question}\n')
log.info(f'Number of relevant docs: {len(docs)}')


# setup the LLM

#llm_mode = 'local'
llm_mode = 'groq'

log.info(f'******  LLM Mode: {llm_mode}  ******')

# LLM
if llm_mode == 'local':
    llm_model = "llama3.1"# "mistral"
    llm = ChatOllama(model=llm_model, temperature=0)
elif llm_mode == 'groq':
    llm_model = "llama-3.1-70b-versatile"
    llm = ChatGroq(model=llm_model, temperature=0)
else:
    log.error(f'Unknown LLM mode {llm_mode}')


### Retrieval Grader

#llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

# Test Retrieval Grader
"""
doc_txt = docs[1].page_content
score = retrieval_grader.invoke({"question": question, "document": doc_txt})
log.info(score)

if score.get('score', '') != 'yes':
    log.error('Retrieval Grader failed!!')
    sys.exit(1)
"""

### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")
log.info(prompt)

"""
[HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['context', 'question'], 
        template="You are an assistant for question-answering tasks. Use the following 
        pieces of retrieved context to answer the question. If you don't know the answer, 
        just say that you don't know. Use three sentences maximum and keep the answer concise.\n
        Question: {question} \nContext: {context} \nAnswer:"
    )
)]
"""

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Test Generate
#generation = rag_chain.invoke({"context": docs, "question": question})
#log.info(f'RESPONSE:\n\n{generation}\n')



### Graph state

from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: list[str]
    messages: Annotated[list, add_messages]

### Nodes


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, 'messages': [question, generation]}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---"
        )
        return False
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return True

def fail(state):
    print("---FAIL---")
    return {'generation': 'I can not answer the question based on the context provided'}

### Build Graph

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# whether to include grading step in the graph or not
include_grading_step = False

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
if (include_grading_step):
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("fail", fail)  # generatae
workflow.add_node("generate", generate)  # generatae

# Build graph
workflow.add_edge(START, "retrieve")
if include_grading_step:
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            False: "fail",
            True: "generate",
        },
    )
    workflow.add_edge("fail", END)
else:
    workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# Compile
app = workflow.compile()#checkpointer=memory)

#from IPython.display import Image, display
#display(Image(app.get_graph().draw_png()))
#app.get_graph().print_ascii()

#app.get_graph().draw_png(output_file_path='graph.png')

config = {"configurable": {"thread_id": "1"}}

def predict(ques, history=None):
    log.info(f'Question:\n\n{ques}\n')
    result = app.invoke({'question': ques})

    log.info(f'Answer:\n\n{result.get("generation", result)}\n')
    return result.get('generation', 'ERROR')

ques = 'how should i plan for my retirement?'
#predict(ques)

# add chat history
# add tool
# add user context
# selectable chat model
# start with interview the user