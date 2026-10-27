# Agentic AI Chatbot

A modular LangGraph-based chatbot with a Streamlit UI, OpenAI-backed model integration, tool calling, and optional document retrieval.

## Overview

This project demonstrates an agentic AI assistant that can:

- answer general questions with a language model,
- use a calculator tool for arithmetic,
- answer time/date questions,
- search a local knowledge base for Python, AI, LLM, RAG, and LangGraph topics,
- optionally index uploaded PDF/TXT documents and search them with vector retrieval,
- maintain conversation context across turns with LangGraph memory.

## Architecture

- `app.py`: Streamlit frontend and chat UI.
- `agent/graph.py`: LangGraph agent workflow and tool registration.
- `agent/state.py`: Graph state definition.
- `agent/prompts.py`: system prompt used by the agent.
- `tools/`: tool implementations for calculation, knowledge search, time, and document RAG.
- `config/settings.py`: environment-driven configuration.

## Prerequisites

- Python 3.10+
- OpenAI API key with access to a supported chat model
- A terminal with internet access for dependency installation

## Virtual environment setup

```bash
cd agentic-chatbot
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment variables

Copy the example file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
APP_TIMEZONE=UTC
```

## Start the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## Example questions

- "What is 2 + 2 * 3?"
- "What is LangGraph?"
- "What is RAG in AI?"
- "What time is it?"
- "Tell me about Python and LLMs."
- "Search the knowledge base for agentic AI."

## Troubleshooting

- If the app says the API key is missing, set `OPENAI_API_KEY` in `.env` and restart.
- If a tool fails, check the console log for the error details.
- If `streamlit` is not found, install requirements in the active virtual environment.
- If document upload fails, confirm the file is a PDF or TXT file and not empty.
- If tests fail with import errors, run `python -m pytest -q` from the project root.

## Notes

The document retrieval workflow is optional and requires an OpenAI API key because embeddings are generated via the OpenAI API.
