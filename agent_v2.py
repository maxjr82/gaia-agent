"""
Multi-tools LangGraph Agent with OpenRouter

This module defines a GAIA benchmark agent that uses multiple tools
(for math, search, PDF/image processing, web scraping, video analysis, etc.)
through LangGraph workflows and an OpenRouter-backed LLM.
"""

import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools.dataset_explorer import analyze_dataset
from tools.file_handler import file_processor
from tools.math_helper import calculator, wolfram_query
from tools.multimedia_analyzer import analyze_image, pdf_text_extractor
from tools.search_engines import (
    arxiv_search,
    web_form_filler,
    web_page_text_extractor,
    web_search,
    wiki_search,
    youtube_video_extractor,
)

# Load environment variables
load_dotenv()

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Aggregate all tools
TOOLS: List = [
    calculator,
    wolfram_query,
    wiki_search,
    web_search,
    arxiv_search,
    analyze_image,
    analyze_dataset,
    file_processor,
    pdf_text_extractor,
    web_form_filler,
    web_page_text_extractor,
    youtube_video_extractor,
]

# Load system prompt
try:
    with open("system_prompt.txt", encoding="utf-8") as fp:
        SYSTEM_PROMPT = fp.read().strip()
except FileNotFoundError:
    logger.error(
        "system_prompt.txt not found. Ensure it exists in the working dir."
    )
    SYSTEM_PROMPT = "You are a helpful assistant."

SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)


def gaia_agent(model: str = "anthropic/claude-3.7-sonnet") -> StateGraph:
    """
    Build and compile the GAIA agent workflow using LangGraph and OpenRouter.

    Args:
        model: Identifier of the LLM model to use with ChatOpenAI.

    Returns:
        A compiled StateGraph ready for invocation.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it to your OpenRouter API key."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
    ).bind_tools(TOOLS)

    def assistant_node(state: MessagesState) -> dict:
        """
        Assistant node: prepends system prompt and invokes the LLM.
        """
        history = state["messages"]
        messages = [SYSTEM_MESSAGE, *history]
        try:
            response = llm.invoke(messages)
            return {"messages": [response]}
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            # Return a fallback reply
            error_msg = "I encountered an error. Please try again."
            fallback = HumanMessage(content=error_msg)
            return {"messages": [fallback]}

    # Build the state graph
    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


if __name__ == "__main__":
    # Example usage
    agent = gaia_agent()
    # question = (
    #    "If a person is 12 years younger than the French president, "
    #    "Emmanuel Macron, which age has this person? Search on the web "
    #    "for the current age of Emmanuel Macron before answering. "
    # )
    question = (
        "What was the actual enrollment count of the clinical "
        "trial on H. pylori in acne vulgaris patients from "
        "Jan-May 2018 as listed on the NIH website?"
    )

    input_payload = {"messages": [HumanMessage(content=question)]}

    response = agent.invoke(input_payload)
    used_tools = set()
    for msg in response.get("messages", []):
        print(f"{msg.type}: {msg.content}\n")
        for line in msg.content.splitlines():
            if line.startswith("[Tool:"):
                tool_name = line.split("]")[0].split(":", 1)[1].strip()
                used_tools.add(tool_name)

    if used_tools:
        print("Tools used during this session:", ", ".join(sorted(used_tools)))
