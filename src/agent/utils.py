"""Utility functions for the agent."""

from __future__ import annotations
from tavily import TavilyClient
from langsmith import traceable

@traceable
def tavily_search(query: str, include_raw_content=True, max_results: int = 2):
    """
    Perform a web search using Tavily API.

    
    Args:
        query: The search query string
        include_raw_content: Whether to include raw content in results (default: True)
        max_results: Maximum number of results to return (default: 2)
        
    Returns:
        Dict containing search results with format:
        {
            "results": List of search results from Tavily API
        }
    """
    
    client = TavilyClient()
    search_result = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=include_raw_content
    )
    
    return search_result

def format_sources(search_results: dict) -> str:
    """
    Format search results from Tavily into a string with 'title: url' on separate lines.
    
    Args:
        search_results: Raw search results from Tavily API
        
    Returns:
        String with each source on a new line formatted as 'title: url'
    """
    formatted_sources = []
    for result in search_results.get("results", []):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        formatted_sources.append(f"{title}: {url}")
    
    return "\n".join(formatted_sources)

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()