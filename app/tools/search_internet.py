import os
from tavily import TavilyClient
from .utils.tools import tool

from app.settings import settings

client = TavilyClient()

@tool()
def search_internet(query: str):
    """
    Search the internet using the official Tavily API.

    Args:
        query: The search query string
    """    
    if not client.api_key:
        return (True, "Tavily API key not found in environment variables")

    try:
        results = client.search(
            query=query,
            max_results=3,
            include_raw_content="text"
        )

        if not results or not results.get("results"):
            return (True, "No results found")

        formatted = [
            f"{i + 1}. {item['title']} - {item['url']}\n{item.get('content', '').strip()}"
            for i, item in enumerate(results["results"])
        ]
        return (False, "\n\n".join(formatted))

    except Exception as e:
        return (True, f"Search failed: {str(e)}")
