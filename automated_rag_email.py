"""
RAG Workflow with LangGraph
Author: Mehdi Rezvandehy

This script builds a dynamic Retrieval-Augmented Generation (RAG) pipeline using
LangGraph. It takes a user query and a list of source websites, retrieves documents,
evaluates their relevance, and generates an answer.

Workflow:
1. Fetch documents from provided websites.
2. Grade relevance using an LLM.
3. If no relevant documents are found:
       - Rewrite (refine) the query and retry document retrieval.
       - After multiple failed retries, perform a web search.
4. Use RAG to generate the final answer from the most relevant documents.

The pipeline uses LangGraph stateful nodes for initialization, retrieval,
grading, rewriting, web search, and final generation.
"""


# User query
query = """What are the latest Calgary housing market trends, including price changes, sales volume, and rental trends?"""

#query = """What is current mortage rate for 5 year fixed term rate, insured and not insured for TD, BMO, CIBC, Scotiabank,
#RBC and ATB Financial, and mention which Bank gives lower mortgage rate"""


# User query (you can also change this to input() if needed)
source_links = [
    "https://www.crea.ca/housing-market-stats/canadian-housing-market-stats/",
    #"https://rentals.ca/blog/",
    "https://www.ratehub.ca/"
]

# =======================================================================

print("Running RAG pipeline...\n")

pipeline = None  # will be defined later

# Call pipeline AFTER function definitions load
def run_pipeline():
    global pipeline
    pipeline = langraph_rag(source_links)
    inputs = {"query": query}

    for output in pipeline.stream(inputs):
        for node, state in output.items():
            print("\n")
            print(f"Node '{node}':")
            #print("\n---------------------")
            #print(state)
    # Final generation
    print(state["llm_output"])

    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    APP_PASSWORD = os.getenv("APP_PASSWORD")
    RECIPIENTS = os.getenv("RECIPIENTS") # for multiple emails ==> os.getenv("RECIPIENTS").split(",")

    #print(" Automatic send email")
    #send_email_report(EMAIL_ADDRESS, APP_PASSWORD, RECIPIENTS, state["llm_output"])


# =======================================================
# FULL RAG + LANGGRAPH PIPELINE BELOW
# =======================================================

def langraph_rag(source_links):

    import warnings
    warnings.filterwarnings('ignore')
    import os
    from dotenv import load_dotenv
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    # pydantic libabray for defining the expected input and output
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI
    load_dotenv(override=True)
    import re

    # Load documents from the URLs
    raw_documents = [WebBaseLoader(link).load() for link in source_links]
    flattened_docs = [doc for group in raw_documents for doc in group]

    # Initialize a text splitter for chunking the documents
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=100
    )
    
    chunked_documents = chunker.split_documents(flattened_docs)

    # Create a vector store with embeddings
    doc_vectorstore = Chroma.from_documents(
        documents=chunked_documents,
        collection_name="cnn-rpn-knowledge-base",
        embedding=OpenAIEmbeddings(),
    )
    
    # Create a retriever from the vector store
    doc_retriever = doc_vectorstore.as_retriever()

    # -------------------------------------------------------
    # API Keys
    # -------------------------------------------------------
    load_dotenv(override=True)
    #os.environ['OPENAI_API_KEY'] = os.environ("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # -------------------------------------------------------
    # Document Grader
    # -------------------------------------------------------
    # Custom schema for grading document relevance
    class RelevanceScore(BaseModel):
        """Binary relevance indicator for retrieved content."""
    
        relevance: str = Field(
            description="Return strictly 'Yes' if the document contains the information needed to answer the question directly. Otherwise return 'No'."
        )
        justification: str = Field(
            description="A short explanation (one sentence) why you said Yes or No."
        )
    
    
    # Bind LLM with the output schema
    structured_grader = llm.with_structured_output(RelevanceScore)
    
    # Prompt definition
    grading_instruction = """
    You are a document relevance evaluator.
    
    Return Yes only if the document contains direct, explicit information that answers at least one of the questions in the query.
    
    Rules:
    
    - If the document only provides broad or general background → return No.
    - If answering would require assumptions or guessing → return No.
    - If none of the key concepts from the query appear in the document → return No.
    - If the answer cannot be clearly extracted from the document → return No.
    
    Return Yes only when at least one user question can be answered using the document.
    """

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grading_instruction),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    # Composing the grader pipeline
    document_grader = grading_prompt | structured_grader


    # -------------------------------------------------------
    # RAG Pipeline
    # -------------------------------------------------------

    from langchain_core.output_parsers import StrOutputParser
    from langchain import hub
    
    # Helper function to format retrieved documents
    def join_documents(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Load a prebuilt RAG prompt template from the LangChain hub
    rag_prompt_template = hub.pull("rlm/rag-prompt")

    # Construct the RAG pipeline
    # Note: StrOutputParser() ensures clean string output from the LLM
    rag_pipeline = rag_prompt_template | llm | StrOutputParser()


    # -------------------------------------------------------
    # Enhance Questions to Optimize Document Search
    # -------------------------------------------------------   
    # Prompt template for rewriting questions to be more effective for web search
    system_instruction = """You are a query rewriter that improves an input question for optimal web search results. 
    Analyze the question and identify its underlying semantic intent or meaning."""
    
    query_rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            (
                "human",
                "Here is the original query: \n\n {question} \n Please rewrite it to be more effective.",
            ),
        ]
    )
    
    # Construct the pipeline: prompt -> LLM -> output parser
    query_optimizer = query_rewrite_prompt | llm | StrOutputParser()


    # -------------------------------------------------------
    # Using LangGraph to Construct the Graph
    # -------------------------------------------------------   

    from typing import List
    from typing_extensions import TypedDict
    
    # Define a structured dictionary to track the workflow state of the graph process
    class PipelineState(TypedDict):
        """
        A dictionary-style representation of the current state in the graph-based pipeline.
    
        Attributes:
            user_query (str): The original or reformulated question.
            llm_output (str): The response generated by the language model.
            retrieved_docs (List[str]): A collection of documents retrieved based on the query.
            rewrite_count (int): Number of times the query has been rephrased.
            enable_web_search (str): Placeholder indicating if a web search should be triggered
                                     (functionality not implemented in this version).
        """
    
        query: str
        llm_output: str
        retrieved_docs: List[str]
        rewrite_count: int
        enable_web_search: str

    
    # -------------------------------------------------------
    # Function for Graph's Nodes
    # -------------------------------------------------------   

    from langchain.schema import Document
    
    def initialize_state(query_state):
        """
        Initializes values in the query state.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: State dictionary with initialized rewrite_count.
        """
        print("---INITIALIZE QUERY STATE---")
        return {
            "rewrite_count": 0,
        }
    
    
    def fetch_documents(query_state):
        """
        Retrieves relevant documents for the given query.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: An updated state dictionary with a new key 'retrieved_docs'.
        """
        print(query_state)
        print("---FETCH DOCUMENTS---")
    
        query = query_state["query"]
    
        # Perform document retrieval
        relevant_docs = doc_retriever.get_relevant_documents(query)
        return {"retrieved_docs": relevant_docs}
    
    
    def filter_relevant_documents(query_state):
        """
        Evaluates and filters retrieved documents for relevance to the query.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: Updated state with only relevant documents and web search indicator.
        """
        print("---EVALUATE DOCUMENT RELEVANCE---")
        query = query_state["query"]
        documents = query_state["retrieved_docs"]
    
        filtered_results = []
        should_perform_web_search = "No"
    
        for doc in documents:
            score = document_grader.invoke({
                "question": query,
                "document": doc.page_content
            })
            is_relevant = score.relevance
            print(doc.metadata.get('source', 'Unknown'), f'Score: {is_relevant}')
    
            if is_relevant == "Yes":
                print("---DOCUMENT IS RELEVANT---")
                filtered_results.append(doc)
    
        if not filtered_results:
            print("---NO RELEVANT DOCUMENTS FOUND---")
            should_perform_web_search = "Yes"
    
        return {
            "retrieved_docs": filtered_results,
            "perform_web_search": should_perform_web_search
        }
    
    
    def refine_query(query_state):
        """
        Rewrites the input query to improve clarity and search effectiveness.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: Updated state with a refined query and incremented transformation count.
        """
        print("---REFINE QUERY---")
    
        query = query_state["query"]
        rewrite_count = query_state["rewrite_count"] + 1
    
        # Rewrite the query using a question rewriter
        improved_query = query_optimizer.invoke({"question": query})
    
        print("---IMPROVED QUERY---")
        print(improved_query)
    
        return {
            "query": improved_query,
            "rewrite_count": rewrite_count}
    
    
    def generate_response(query_state):
        """
        Generates a response using RAG (Retrieval-Augmented Generation).
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: An updated state dictionary with a new key 'llm_output'.
        """
        print("---GENERATE RESPONSE---")
    
        query = query_state["query"]
        documents = query_state["retrieved_docs"]
    
        # Generate an answer using RAG
        response = rag_pipeline.invoke({
            "context": join_documents(documents),
            "question": query
        })
    
        return {"llm_output": response}
    
    # -------------------------------------------------------
    # Functions for Graph's Edges
    # -------------------------------------------------------   
    from langchain.utilities.tavily_search import TavilySearchAPIWrapper
    from langchain.schema.document import Document

    # You need to get your API key from https://app.tavily.com for web search.
    #os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    
    def web_search(query_state):
        """
        Executes a web search using the given query and returns a list of document objects.
    
        Args:
            query_state: The current query state containing the search question.
    
        Returns:
            Dict[str, Any]: Updated state including the original question and retrieved documents.
        """
        query = query_state["query"]
    
        search_client = TavilySearchAPIWrapper()
        search_results = search_client.results(query=query, max_results=3)
    
        retrieved_docs = [
            Document(page_content=item["content"], metadata={"source": item["url"]})
            for item in search_results
        ]
    
        return {
            "query": query,
            "retrieved_docs": retrieved_docs
        }
    
    
    def decide_next_action(query_state):
        """
        Determines the next step in the workflow: whether to generate an answer
        or rephrase the query again for better document retrieval.
    
        Args:
            query_state (dict): The current state of the query process.
    
        Returns:
            str: The next action to take - either 'if_generate' or 'if_transform_query'.
        """
    
        print("---ASSESSING DOCUMENT RELEVANCE---")
        requires_web_search = query_state["perform_web_search"]
    
        if requires_web_search == "Yes":
            # If the query has already been transformed multiple times with no success,
            # proceed ro apply we search.
            if query_state["rewrite_count"] >= 2:
                print(
                    "---DECISION: MAX REWRITES REACHED AND NO RELEVANT DOCUMENTS FOUND ---"
                )        
                
                return "apply_web_search"
    
            # Still below the rewrite threshold; attempt another reformulation.
            print(
                "---DECISION: NO RELEVANT DOCUMENTS FOUND"
            )
            return "apply_transform_query"
    
        else:
            # Relevant documents are present; move on to answer generation.
            print("---DECISION: RELEVANT DOCUMENTS FOUND → GENERATE---")
            return "apply_generate"

    # -------------------------------------------------------
    # Build the Graph
    # -------------------------------------------------------   
    from langgraph.graph import START, END, StateGraph
    
    # Create a new stateful graph using the defined PipelineState structure
    pipeline_graph = StateGraph(PipelineState)
    
    # Register processing steps (nodes) in the graph
    pipeline_graph.add_node("initialize_state", initialize_state)                   # Step 1: Initialize state
    pipeline_graph.add_node("fetch_documents", fetch_documents)                     # Step 2: Retrieve documents
    pipeline_graph.add_node("filter_relevant_documents", filter_relevant_documents) # Step 3: Grade relevance
    pipeline_graph.add_node("generate_response", generate_response)                 # Step 4: Generate answer
    pipeline_graph.add_node("refine_query", refine_query)                           # Step 5: Refine query if needed
    pipeline_graph.add_node("web_search", web_search)                               # Step 6: Web search
    
    # Define the workflow logic by adding edges between steps (nodes)
    pipeline_graph.add_edge(START, "initialize_state")
    pipeline_graph.add_edge("initialize_state", "fetch_documents")
    pipeline_graph.add_edge("fetch_documents", "filter_relevant_documents")
    
    # Add branching logic after grading based on decision function
    pipeline_graph.add_conditional_edges(
        "filter_relevant_documents",
        decide_next_action,
        {
            "apply_transform_query": "refine_query",
            "apply_web_search": "web_search",
            "apply_generate": "generate_response",
        }
    )
    
    # Handle loopback and termination
    pipeline_graph.add_edge("refine_query", "fetch_documents")
    pipeline_graph.add_edge("web_search", "generate_response")
    pipeline_graph.add_edge("generate_response", END)
    
    # Finalize and compile the pipeline
    retrieval_qa_pipeline = pipeline_graph.compile()


    return retrieval_qa_pipeline

# ==============================================
# Email Sender
# ==============================================
def send_email_report(sender_email, app_password, recipients, content):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg['Subject'] = "RAG Pipeline Report"
    msg['From'] = sender_email
    msg['To'] = recipients  # <-- must be string of addresses
    msg.set_content(content)

    # Gmail SMTP server
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

    print("Email sent!")

# ============================
# MUST ADD THIS or nothing runs
# ============================
if __name__ == "__main__":
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    rag_output = run_pipeline()

