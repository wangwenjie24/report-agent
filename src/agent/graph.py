"""Define a report agent.

This agent returns a predefined response without using an actual LLM.
"""

from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from agent.configuration import Configuration
from agent.state import SummaryInputState, SummaryOutputState, SummaryState
from agent.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions
from agent.utils import tavily_search, format_sources, deduplicate_and_format_sources

class ResearchQuery(BaseModel):
    query: str = Field(description="The actual search query string")
    aspect: str = Field(description="The specific aspect of the topic being researched")
    rationale: str = Field(description="Brief explanation of why this query is relevant")

class ReflectionQuery(BaseModel):
    knowledge_gap: str = Field(description="Describe what information is missing or needs clarification")
    follow_up_query: str = Field(description="Write a specific question to address this gap")

def generate_query(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a query for web search."""

    configuration = Configuration.from_runnable_config(config)

    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ResearchQuery)
    response = llm.invoke([
        SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content="Generate a query for web search")
    ])
    return {"search_query": response.query}

def web_research(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """Search the web for the query."""
    configuration = Configuration.from_runnable_config(config)
    search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_search_results": [deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)]
    }

def summarize(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize the web search results."""
    configuration = Configuration.from_runnable_config(config)

    existing_summary = state.running_summary
    most_recent_web_research = state.web_search_results[-1]

    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)
    ])

    running_summary = response.content

    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]
    
    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """ Reflect on the summary and generate a follow-up query """
    configuration = Configuration.from_runnable_config(config)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ReflectionQuery)
    response = llm.invoke([
        SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")
    ])
    
    query = response.follow_up_query

    if not query:
        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}
    
    return {"search_query": query}


def finalize_summary(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """Finalize the summary."""
    configuration = Configuration.from_runnable_config(config)

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## 总结\n\n{state.running_summary}\n\n ### 来源:\n{all_sources}"
    return {"running_summary": state.running_summary}

def router(state: SummaryState, config: RunnableConfig) -> Literal["web_research", "finalize_summary"]:
    """ Route the research based on the follow-up query """
    configuration = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configuration.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


# Define a new graph
workflow = StateGraph(state_schema=SummaryState, input=SummaryInputState, output=SummaryOutputState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("generate_query", generate_query)
workflow.add_node("web_research", web_research)
workflow.add_node("summarize", summarize)
workflow.add_node("reflect_on_summary", reflect_on_summary)
workflow.add_node("finalize_summary", finalize_summary)

# Set the entrypoint as `call_model`
workflow.add_edge("__start__", "generate_query")
workflow.add_edge("generate_query", "web_research")
workflow.add_edge("web_research", "summarize")
workflow.add_edge("summarize", "reflect_on_summary")
workflow.add_conditional_edges("reflect_on_summary", router)
workflow.add_edge("finalize_summary", "__end__")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Report Agent"  # This defines the custom name in LangSmith
