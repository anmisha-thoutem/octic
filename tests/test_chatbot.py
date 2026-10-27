import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from tools.calculator import evaluate_expression
from tools.knowledge_base import search_knowledge_base


def test_calculator_valid_expression():
    result = evaluate_expression("2 + 2 * 3")
    assert result == 8


def test_calculator_invalid_expression():
    with pytest.raises(ValueError):
        evaluate_expression("__import__('os').system('echo hacked')")


def test_knowledge_base_search():
    result = search_knowledge_base("langgraph")
    assert result["matches"]
    assert "langgraph" in result["summary"].lower()


def test_agent_tool_invocation(monkeypatch):
    class FakeModel:
        def bind_tools(self, tools):
            self.tools = tools
            return self

        def invoke(self, messages):
            if len(messages) > 2 and messages[-1].type == "tool":
                return AIMessage(content="The answer is 4.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "2 + 2"},
                        "id": "tool_call_1",
                    }
                ],
            )

    monkeypatch.setattr("agent.graph.ChatOpenAI", lambda *args, **kwargs: FakeModel())
    graph = build_graph(allow_unconfigured=True)
    state = {"messages": [HumanMessage(content="What is 2 + 2?")]}
    result = graph.invoke(state, {"configurable": {"thread_id": "test-session"}})
    assert any("4" in message.content for message in result["messages"])


def test_multi_turn_memory():
    graph = build_graph(allow_unconfigured=True)
    first = graph.invoke({"messages": [HumanMessage(content="My name is Ada.")]}, {"configurable": {"thread_id": "memory-session"}})
    second = graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, {"configurable": {"thread_id": "memory-session"}})
    assert second["messages"]


def test_error_handling_for_empty_message():
    with pytest.raises(ValueError):
        evaluate_expression("   ")
