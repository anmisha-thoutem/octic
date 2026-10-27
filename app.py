import logging
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from config.settings import settings
from tools.document_search import upload_and_index_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

st.set_page_config(page_title="Agentic AI Assistant", page_icon="🤖", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-session"


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def get_messages_for_graph() -> list:
    return [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
        for msg in st.session_state.messages
    ]


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = f"chat-session-{len(st.session_state.messages)}"


def handle_user_input(user_input: str) -> None:
    if not user_input or not user_input.strip():
        st.warning("Please enter a message before sending.")
        return

    add_message("user", user_input.strip())

    if not settings.has_openai_key:
        add_message("assistant", "OpenAI API key is missing. Add OPENAI_API_KEY to your .env file and restart the app.")
        st.error("Missing OpenAI API key. Check your environment configuration.")
        return

    try:
        graph = build_graph(allow_unconfigured=False)
        latest_messages = get_messages_for_graph()
        response = graph.invoke(
            {"messages": latest_messages},
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )
        final_message = response["messages"][-1]
        content = getattr(final_message, "content", "")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            content = "I could not generate a response from the model."
        add_message("assistant", str(content))
    except Exception as exc:
        logging.exception("Agent invocation failed")
        add_message("assistant", f"The assistant encountered an error: {exc}")
        st.error(f"Something went wrong: {exc}")


st.title("Agentic AI Assistant")
st.caption("Ask about Python, AI, LLMs, RAG, LangGraph, general knowledge, or math. The agent can use tools when needed.")

with st.sidebar:
    st.header("Available tools")
    st.markdown("- Calculator\n- Knowledge base search\n- Current date/time\n- Document search (upload PDF/TXT)")

    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            result = upload_and_index_document(uploaded_file.name, bytes_data)
            st.success(f"Document indexed: {result['chunks']} chunks from {uploaded_file.name}")
        except Exception as exc:
            st.error(f"Document upload failed: {exc}")

    st.button("Clear chat", on_click=clear_chat)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask the agent a question...")
if prompt:
    handle_user_input(prompt)
    st.rerun()
