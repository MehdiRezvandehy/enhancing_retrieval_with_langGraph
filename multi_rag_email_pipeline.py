"""
Dynamic Multi-RAG Workflow with LangGraph
Author: Mehdi Rezvandehy

Description:
This script implements a fully automated, multi-step Retrieval-Augmented Generation (RAG) pipeline
using LangGraph and OpenAI LLMs to answer user queries based on multiple web sources. It is designed 
to handle cases where relevant information may not be immediately available and can refine queries 
or perform web searches dynamically.

Key Features:
1. Document Retrieval: Fetches content from provided URLs and splits them into manageable chunks.
2. Relevance Evaluation: Uses an LLM-based structured grader to filter retrieved documents.
3. Query Refinement: Automatically rewrites queries when no relevant documents are found.
4. Web Search Fallback: Performs web searches if the retrieval from provided sources fails.
5. RAG Generation: Produces final answers by combining relevant documents via an LLM-driven RAG pipeline.
6. Response Evaluation: Scores generated answers for completeness and correctness.
7. Multi-Question Handling: Supports multiple queries in one run and synthesizes results into a 
   cohesive summary.
8. Email Notification: Sends the final synthesized report via email in both plain text and HTML.

Workflow:
START → Initialize State → Fetch Documents → Filter Relevant Documents → 
    Conditional:
        - If relevant → Generate Response → Evaluate → END
        - If no relevant documents → Refine Query → Fetch Documents (loop)
        - If repeated failures → Web Search → Generate → Evaluate → END

Execution Options:
- Run Locally: Provide your credentials and API keys in a `.env` file to authenticate services
  (OpenAI API key, email credentials, Tavily API key).
- Automated Scheduled Runs: Use GitHub Actions to run the pipeline on a schedule (daily, weekly,
  hourly). Store credentials in GitHub Secrets to allow the workflow to authenticate and send emails.
"""

from datetime import date

today = date.today()
formatted_date = today.strftime("%B %Y") 

rag_inputs = [
    {
        "question": f"What current Nissan promotions are available in Alberta as of {formatted_date}?",
        "links": ["https://www.stadiumnissan.com/our-promotions.html"]
    },
    {
        "question": f"What Honda vehicle offers are available in Alberta as of {formatted_date}?",
        "links": ["https://www.honda.ca/special-offers/alberta"]
    },
    {
        "question": f"What are Toyota’s latest financing deals in Alberta as of {formatted_date}?",
        "links": ["https://www.shoptoyota.ca/alberta/en"]
    },
    {
        "question": f"What Mazda offers are available in Alberta as of {formatted_date}?",
        "links": ["https://albertamazdaoffers.ca/"]
    },
    {
        "question": f"What are Mercedes-Benz’s current special offers in Alberta as of {formatted_date}?",
        "links": ["https://www.mercedes-benz-countryhills.ca/en/special-offers"]
    }
]

# =======================================================================

print("Running RAG pipeline...\n")

import os
# Load API keys locally
#from dotenv import load_dotenv
#load_dotenv(override=True)
#os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
#os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")
#os.environ["EMAIL_API_KEY"] = os.environ.get("EMAIL_API_KEY")
#EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
#APP_PASSWORD = os.getenv("APP_PASSWORD")
#RECIPIENTS = os.getenv("RECIPIENTS")

# Run by GitHub Actions 
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECIPIENTS = os.getenv("RECIPIENTS") # for multiple emails ==> os.getenv("RECIPIENTS").split(",")

llm_model="gpt-4o-mini"
pipeline = None  # will be defined later

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Call pipeline AFTER function definitions load
def run_pipeline():
    global pipeline

    rag_outputs = []
    for inp in rag_inputs:
        rag = LangGraphRAG (source_links=inp['links'],
                           llm=llm)
        inputs = {
            "query": inp["question"],
        }
        for output in rag.pipeline.stream(inputs):
            for key, value in output.items():
                # Node
                print(f'----------Node "{key}" Completed-------------')
                print("\n")
        # Final generation
        rag_outputs.append(value["llm_output"])
    return rag_outputs


# --------------------------------------------
# LLM
# --------------------------------------------
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=llm_model, temperature=0)

# =======================================================
# FULL RAG + LANGGRAPH PIPELINE BELOW
# =======================================================

class LangGraphRAG:
    """
    End-to-end Retrieval-Augmented Generation pipeline using LangGraph.
    Entire pipeline is constructed in __init__ and exposed through .pipeline
    """

    def __init__(self, source_links, llm):
        import warnings
        warnings.filterwarnings("ignore")

        from dotenv import load_dotenv
        import uuid
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.vectorstores import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.output_parsers import StrOutputParser
        from langchain import hub
        from pydantic import BaseModel, Field
        from typing import List, Any
        from typing_extensions import TypedDict
        from datetime import date

        load_dotenv(override=True)

        # --------------------------------------------
        # Store config
        # --------------------------------------------
        self.source_links = source_links

        # --------------------------------------------
        # Schema for relevance grading
        # --------------------------------------------
        class RelevanceScore(BaseModel):
            relevance: str = Field(description="'Yes' or 'No'")
            justification: str = Field(description="Short justification")

        self.RelevanceScore = RelevanceScore

        # bind structured output
        self.structured_grader = llm.with_structured_output(RelevanceScore)

        grading_instruction = """
        You are a document relevance evaluator.
        Return 'Yes' or 'No' to indicate if the document is relevant to the given question.
        """

        self.grading_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", grading_instruction),
                ("human", "Retrieved document:\n{document}\n\nUser question:\n{question}")
            ]
        )

        self.document_grader = self.grading_prompt | self.structured_grader

        # ------------------------------------------------------------
        # Join documents helper
        # ------------------------------------------------------------
        def join_documents(docs):
            return "\n\n".join(d.page_content for d in docs)

        self.join_documents = join_documents

        # ------------------------------------------------------------
        # RAG Pipeline
        # ------------------------------------------------------------
        rag_prompt_template = hub.pull("rlm/rag-prompt")
        self.rag_pipeline = rag_prompt_template | llm | StrOutputParser()

        # --------------------------------------------
        # Query Rewriter
        # --------------------------------------------
        system_instruction = "Rewrite the query to be more effective for search."

        self.query_rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                ("human", "Original query:\n{question}\nRewrite it.")
            ]
        )
        self.query_optimizer = self.query_rewrite_prompt | llm | StrOutputParser()

        # --------------------------------------------
        # Build retriever
        # --------------------------------------------
        self.doc_retriever = self._build_retriever(source_links)

        # --------------------------------------------
        # Graph State structure
        # --------------------------------------------
        class PipelineState(TypedDict):
            query: str
            doc_retriever: Any
            llm_output: str
            retrieved_docs: List[str]
            refine_query_count: int
            rewrite_count: int
            web_search_count:int
            perform_web_search: str

        self.PipelineState = PipelineState

        
        # --------------------------------------------
        # Build the LangGraph pipeline
        # --------------------------------------------
        from langgraph.graph import StateGraph, START, END
    
        # Create a new stateful graph using the defined PipelineState structure
        pipeline_graph = StateGraph(PipelineState)
        
        # Register nodes
        pipeline_graph.add_node("initialize_state", self.initialize_state)
        pipeline_graph.add_node("fetch_documents", self.fetch_documents)
        pipeline_graph.add_node("filter_relevant_documents", self.filter_relevant_documents)
        pipeline_graph.add_node("generate_response", self.generate_response)
        pipeline_graph.add_node("evaluate_response", self.evaluate_response)
        pipeline_graph.add_node("refine_query", self.refine_query)
        pipeline_graph.add_node("web_search", self.web_search)
        
        # --- GRAPH LOGIC ---
        
        # Start → Initialize
        pipeline_graph.add_edge(START, "initialize_state")
        
        # Initialize → Fetch Documents
        pipeline_graph.add_edge("initialize_state", "fetch_documents")
        
        # Fetch → Filter Relevant
        pipeline_graph.add_edge("fetch_documents", "filter_relevant_documents")
        
        # Conditional logic
        pipeline_graph.add_conditional_edges(
            "filter_relevant_documents",
            self.decide_next_action,
            {
                "apply_transform_query": "refine_query",
                "apply_web_search": "web_search",
                "apply_generate": "generate_response",
            }
        )
        
        # If refine_query → go back to fetch
        pipeline_graph.add_edge("refine_query", "fetch_documents")
        
        # Web search → Generate
        pipeline_graph.add_edge("web_search", "generate_response")
        
        # Generate → Evaluate
        pipeline_graph.add_edge("generate_response", "evaluate_response")
        
        # Evaluate → End or Web Search
        pipeline_graph.add_conditional_edges(
            "evaluate_response",
            self.decide_to_end,
            {
                "apply_end": END,
                "apply_web_search": "web_search",
            }
        )
                

        self.pipeline = pipeline_graph.compile()
        

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _build_retriever(self, source_links):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        import uuid

        raw_documents = [WebBaseLoader(u).load() for u in source_links]
        docs = [d for group in raw_documents for d in group]

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=120
        )
        chunks = splitter.split_documents(docs)

        collection_id = f"collection-{uuid.uuid4()}"
        store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            collection_name=collection_id,
            persist_directory=None
        )
        return store.as_retriever()


    # ============================================================
    # FUNCTION FOR GRAPH'S NODES
    # ============================================================    
    
    def initialize_state(self, query_state):
        """
        Initializes values in the query state.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: State dictionary with initialized refine_query_count and web_search_count.
        """
        print("---INITIALIZE QUERY STATE---")
        return {"refine_query_count": 0, 
                "web_search_count": 0,
                "doc_retriever": None,
               }
    
    def fetch_documents(self, query_state):
        """
        Retrieves relevant documents for the given query.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: An updated state dictionary with a new key 'retrieved_docs'.
        """
        print("---FETCH DOCUMENTS---")
    
        query = query_state["query"]
    
        # Perform document retrieval
        doc_retriever = query_state["doc_retriever"]  
        relevant_docs = self.doc_retriever.get_relevant_documents(query)
        return {"retrieved_docs": relevant_docs}

    
    def filter_relevant_documents(self, query_state):
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
            score = self.document_grader.invoke({
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
    
    
    def refine_query(self, query_state):
        """
        Rewrites the input query to improve clarity and search effectiveness.
    
        Args:
            query_state (dict): The current state of the query workflow.
    
        Returns:
            dict: Updated state with a refined query and incremented transformation count.
        """
        print("---REFINE QUERY---")
    
        query = query_state["query"]
        refine_query_count = query_state["refine_query_count"] + 1
    
        # Rewrite the query using a question rewriter
        improved_query = self.query_optimizer.invoke({"question": query})
    
        print("---IMPROVED QUERY---")
        print(improved_query)
    
        return {
            "query": improved_query,
            "refine_query_count": refine_query_count}
    
    
    def generate_response(self, query_state):
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
        response = self.rag_pipeline.invoke({
            "context": self.join_documents(documents),
            "question": query
        })
    
        return {"llm_output": response}

    
    # ============================================================
    # FUNCTION FOR GRAPH'S EDGES
    # ============================================================
    
    def evaluate_response(self, query_state):
        """
        Args:
            query_state: The current query state containing the search question.
    
        Returns:
            int: score (from 1 to 10) for the agent response
        """

        template_reason = '''
        You are evaluating whether an assistant’s response fully and correctly answers all parts of a user’s question.
        
        Your task:
        - Provide a concise **reason** explaining whether the response answers all questions.
        - Assign a **score** from 1 to 10 based on completeness.
        
        ### User Question
        {query}
        
        ### Assistant's Response
        {llm_output}
        
        ### Output Format (JSON)
        {{"reason": "...", "score": ...}}
        
        '''
        
        print(
            "---ASSESSING RESPONSE: PREDICT SCORE FOR LLM RESPONSE---"
        )  
        query = query_state["query"]
        generate_response = query_state["llm_output"]
        template_prompt = ChatPromptTemplate.from_template(template_reason)
        # message template
        messages = template_prompt.format_messages(query=query, 
                                                   llm_output=generate_response)
        respose = llm.invoke(messages).content
        
    
        return {
            "response_score": json.loads(respose)['score'],
            "llm_output": generate_response
        }
    
    
    def web_search(self, query_state):
        """
        Executes a web search using the given query and returns a list of document objects.
    
        Args:
            query_state: The current query state containing the search question.
    
        Returns:
            Dict[str, Any]: Updated state including the original question and retrieved documents.
        """

        query = query_state["query"]
        web_search_count = query_state["web_search_count"] + 1
        #
        search_client = TavilySearchAPIWrapper()
        search_results = search_client.results(query=query, max_results=3)
    
        retrieved_docs = [
            Document(page_content=item["content"], metadata={"source": item["url"]})
            for item in search_results
        ]
        print(
            "---WEB SEARCH: RETRIEVED DOCS FROM WEB SEARCH---"
        ) 
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "web_search_count": web_search_count
        }
    
    
    def decide_next_action(self, query_state):
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
            # proceed to generate an answer anyway.
            if query_state["refine_query_count"] >= 3:
                print(
                    "---DECISION: MAX REWRITES REACHED AND NO RELEVANT DOCUMENTS FOUND → LETS APPLY WEB SEARCH---"
                )        
    
                return "apply_web_search"
    
            # Still below the rewrite threshold; attempt another reformulation.
            print(
                "---DECISION: NO RELEVANT DOCUMENTS FOUND YET → TRANSFORM QUERY AGAIN---"
            )
            return "apply_transform_query"
    
        else:
            # Relevant documents are present; move on to answer generation.
            print("---DECISION: RELEVANT DOCUMENTS FOUND → GENERATE---")
            return "apply_generate"

        
    def decide_to_end(self, query_state):
        """
        Determines to end the workflow: 
        Args:
            query_state (dict): The current state of the query process.
    
        Returns:
    
        """
    
        print("---ASSESSING to END the WORKFLOW OR NOT ---")
        response_score_value = query_state["response_score"]
    
        if response_score_value >= 6:
            # If the score is bigger than 6, we can end the work. As we do not have
            # ground truth, score above 6 is high enough
    
            print(
                "---DECISION: THE SCORE IS REASONABLE → END---"
            )
    
            return "apply_end"
    
        elif query_state["web_search_count"]<=2:
            # Relevant documents are present; move on to answer generation.
            print("---DECISION: THE SCORE IS LOW → LETS APPLY WEB SEARCH (again)---")
            return "apply_web_search" 
        
        else :
            # Relevant documents are present; move on to answer generation.
            print("---DECISION: THE SCORE IS LOW APPLY MULTIPPLE WEB SEARCH CANNOT FIND AN ANSWER → END---")
            return "apply_end"     


# ==============================================
# Synthesizer
# ==============================================
def aggregate_answers(llm, rag_outputs):
    aggregation_prompt = f"""
    You are an expert synthesizer.
    
    Combine the following question results achieved from RAG into a clear and cohesive final summary separate them. Start
    with date of today {today}
    
    RAG Results:
    {"\n\n".join(
        f"RAG #{i+1}\nQuestion: {item_1['question']}\nAnswer:\n{item_2}"
        for i, (item_1, item_2) in enumerate(zip(rag_inputs, rag_outputs))
    )}
    """

    final = llm.invoke(aggregation_prompt)
    return final


# ==============================================
# Email Sender
# ==============================================
def send_email_report(EMAIL_ADDRESS, APP_PASSWORD, RECIPIENTS, markdown_text, subject):
    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECIPIENTS
    msg["Subject"] = subject

    # Convert markdown → HTML
    html_text = markdown.markdown(markdown_text)

    # Attach both plain and HTML versions
    msg.attach(MIMEText(markdown_text, "plain"))   # fallback
    msg.attach(MIMEText(html_text, "html"))        # formatted version

    # Send via Gmail
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, APP_PASSWORD)
        server.send_message(msg, from_addr=EMAIL_ADDRESS, to_addrs=RECIPIENTS)

    print(f"✅ Email has been sent\n")


# ============================
# MUST ADD THIS or nothing runs
# ============================
if __name__ == "__main__":
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.schema.document import Document
    from langchain.utilities.tavily_search import TavilySearchAPIWrapper
    import json
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import markdown
    from rich.console import Console
    from rich.markdown import Markdown

    outputs = run_pipeline()
    final_answer = aggregate_answers(llm, outputs)
    console = Console()
    console.print(Markdown(final_answer.content))
    print("MULTI LangGraphRAG COMPLETED\n")

    send_email_report(EMAIL_ADDRESS, APP_PASSWORD, RECIPIENTS, final_answer.content, 
                      subject="🚗 Alberta Car Financing Update")
    print("🚀 PIPELINE FINISHED")

    
