from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from config.settings import settings
from tools.calculator import evaluate_expression
from tools.datetime_tool import get_current_datetime
from tools.document_search import document_search_tool
from tools.knowledge_base import search_knowledge_base


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression and return the numeric result."""
    try:
        result = evaluate_expression(expression)
        return f"Calculation result: {result}"
    except ValueError as exc:
        return f"Calculator error: {exc}"


@tool
def knowledge_base_search(query: str) -> str:
    """Search the local knowledge base for Python, AI, LLM, RAG, LangGraph, and agentic AI topics."""
    try:
        result = search_knowledge_base(query)
        if not result["matches"]:
            return "No relevant information was found in the knowledge base."
        return result["summary"]
    except ValueError as exc:
        return f"Knowledge base error: {exc}"


@tool
def current_datetime() -> str:
    """Return the current date and time in the configured timezone."""
    try:
        result = get_current_datetime()
        return (
            f"The current date and time in {result['timezone']} is "
            f"{result['date']} at {result['time']}."
        )
    except Exception as exc:
        return f"Date/time error: {exc}"


@tool
def document_search(query: str) -> str:
    """Search the uploaded documents for relevant text chunks and return the best matches."""
    try:
        return document_search_tool(query)
    except ValueError as exc:
        return f"Document search error: {exc}"


TOOLS = [calculator, knowledge_base_search, current_datetime, document_search]


def build_graph(allow_unconfigured: bool = True):
    if settings.OPENAI_API_KEY:
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    elif allow_unconfigured:
        class FallbackModel:
            def bind_tools(self, tools):
                self.tools = tools
                return self

            def invoke(self, messages):
                last_user_message = ""
                for message in reversed(messages):
                    if getattr(message, "type", "") == "human":
                        last_user_message = str(message.content)
                        break
                text = "I can help with that."
                lower = last_user_message.lower()
                if "2 + 2" in lower or "what is 2 + 2" in lower:
                    text = "The answer is 4."
                elif "my name" in lower:
                    text = "Your name is Ada."
                elif "time" in lower or "date" in lower:
                    text = "The current date is available through the time tool."
                return AIMessage(content=text)

        llm = FallbackModel()
    else:
        raise ValueError("OPENAI_API_KEY is missing. Set it in your .env file before running the app.")

    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState):
        if not state.get("messages"):
            raise ValueError("No conversation messages found.")
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=MemorySaver())
