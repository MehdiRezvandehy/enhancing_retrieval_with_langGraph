# multi_rag_workflow_with_langgraph

## 🚀 Overview

This repository illustrates how to build advanced multi-RAG workflows by combining:

* **Corrective RAG techniques** — where the system refines its retrieval and answers using self-grading or self-reflection.
* **Multi-step LangGraph workflows** — orchestrating agents or nodes to enhance retrieval relevance and answer quality.
* **Example pipelines** including an email automation RAG pipeline.

The core idea is to explore how LangGraph can orchestrate retrieval, refinement, and generation logic in a graph workflow powered by large language models (LLMs).

---

## 📁 Repository Structure

```
multi_rag_workflow_with_langgraph/
├── .github/                 # CI/CD workflows
├── assets/
│   ├── enhancing_retrieval_langGraph.gif
│   └── … other visuals …
├── enhance_retrieval_langgraph.ipynb         # Primary notebook demonstration
├── enhance_retrieval_langgraph.html          # Exported HTML notebook
├── multi_rag_email_pipeline.py               # Python script example showing an email RAG workflow
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

---

## 📌 Key Components

### 📘 [enhance_retrieval_langgraph.ipynb](https://github.com/MehdiRezvandehy/multi_rag_workflow_with_langgraph/blob/master/multi_rag_workflow_with_langgraph.ipynb)

A **Jupyter Notebook** that:

* Introduces corrective RAG concepts.
* Shows techniques to improve retrieval effectiveness.
* Uses LangGraph to construct an orchestrated RAG pipeline.
* Demonstrates query refinement and web search integration.
* Explain how to run the multi rag workflow on schedule and email automatically. 

This notebook can be a starting point for RAG experimentation and experimentation with LangGraph graph orchestration. ([GitHub][1])

---

### 🧠 [multi_rag_email_pipeline.py](https://github.com/MehdiRezvandehy/multi_rag_workflow_with_langgraph/blob/master/multi_rag_email_pipeline.py)  — What It Does

A **Python script** implementing a multi-agent / RAG-style workflow for processing email data. While this script is more targeted than the main notebook, it demonstrates how you might:

1. **Document Retrieval:** Collects content from URLs and splits it into chunks.
2. **Relevance Filtering:** Uses an LLM grader to keep only useful documents.
3. **Query Refinement:** Rewrites the query if no good matches are found.
4. **Web Search Backup:** Falls back to web search when source retrieval fails.
5. **RAG Generation:** Combines relevant documents to create final answers.
6. **Answer Scoring:** Evaluates responses for quality and completeness.
7. **Multi-Query Support:** Handles multiple questions and generates a unified summary.
8. **Email Reporting:** Sends the final summary by email in text and HTML.

This script applies a use case that collects the latest promotions and special deals for various car brands in Alberta using a RAG workflow, then generates a clean summary report and emails it to your recipients.

It can also be adapted for custom pipelines and integrations.

---

## 📦 Prerequisites

Install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Typical packages in a LangGraph RAG workflow include:

* **langgraph**, **langchain**
* **OpenAI or other LLM providers**
* Vector store and embedding libraries
* Retrieval utilities

(Ensure your `.env` has the necessary API keys for your LLM and vector store services.)

---
## 💡 How It Works

At a high level, the workflows in this repo demonstrate:

1. **Query Pre-processing** — refining or rewriting the user question.
2. **Document Retrieval** — searching a vector database for relevant context.
3. **LLM Generation** — combining retrieval results with LLM reasoning.
4. **Self-Reflection / Correction** — evaluating outputs and improving answer quality through iterative logic.

These steps are orchestrated as a graph using LangGraph, which allows you to define language-agent interactions as connected nodes and workflows. ([LangChain Blog][2])

---

## 🚀 Sending Emails Automatically

To automatically run the pipeline and send emails on a schedule:

1. **Set GitHub Secrets**
   In your repository **Settings → Secrets and variables → Actions**, add your credentials:

   * `EMAIL_ADDRESS` – sender email address
   * `APP_PASSWORD` – app-specific password or SMTP credential
   * `RECIPIENTS` – comma-separated list of email recipients
   * Any API keys required for the RAG workflow (e.g., OpenAI keys)

2. **GitHub Actions Workflow**
   The `.github/workflows/email.yml` file runs the script on a schedule (e.g., daily). It installs dependencies and executes `multi_rag_email_pipeline.py`, sending the report as an email without any manual steps.

   > Typically such workflows use an action like `dawidd6/action-send-mail@…` to send emails via SMTP based on the secrets you configured. ([GitHub][1])

Once configured, the workflow will **run on schedule and send the latest deals automatically**.

---

## 📘 More Details

For full technical walkthrough and how the retrieval + workflow logic works, see the notebook **`enhance_retrieval_langgraph.ipynb`** — it explains the RAG components and how the pipeline is structured.

[1]: https://github.com/dawidd6/action-send-mail?utm_source=chatgpt.com "GitHub - dawidd6/action-send-mail: :gear: A GitHub Action to send an email to multiple recipients"

