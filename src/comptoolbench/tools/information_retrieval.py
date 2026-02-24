"""Information retrieval tools: web search, knowledge base, database query.

Live mode uses DuckDuckGo (free, no key) for web search.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import requests

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)


def _sim_hash(seed: str, *args: Any) -> int:
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16)


@register_tool
class WebSearch(BaseTool):
    """Search the web for information."""

    name = "web_search"
    schema = ToolSchema(
        name="web_search",
        description="Search the web for information on any topic. Returns a list of relevant results with titles, URLs, and snippets.",
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="query", type="string",
                description="The search query",
            ),
            ToolParameter(
                name="max_results", type="integer",
                description="Maximum number of results (default 5)",
                required=False, default=5,
            ),
        ],
        returns="List of search results with title, URL, and snippet",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        query = kwargs["query"]
        max_results = int(kwargs.get("max_results", 5))

        # Use DuckDuckGo instant answer API (free, no key)
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            # Abstract (main result)
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["AbstractText"][:200],
                    "source": data.get("AbstractSource", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:200],
                        "source": "DuckDuckGo",
                    })

            if not results:
                return self.execute_simulated(**kwargs)

            return {"query": query, "results": results[:max_results], "total": len(results)}

        except Exception:
            return self.execute_simulated(**kwargs)

    _SNIPPET_TEMPLATES = [
        "A comprehensive overview of {q} covering key concepts, recent developments, and practical applications.",
        "Latest research findings on {q} from leading institutions and peer-reviewed publications.",
        "{q}: analysis of current trends, methodologies, and future directions in the field.",
        "Expert guide to {q} with detailed explanations, examples, and best practices for practitioners.",
        "Comparing different approaches to {q} — strengths, limitations, and real-world performance.",
    ]

    _DOMAINS = ["en.wikipedia.org", "arxiv.org", "nature.com", "sciencedirect.com",
                 "medium.com", "github.com", "stackoverflow.com", "nytimes.com",
                 "bbc.com", "reuters.com"]

    def execute_simulated(self, **kwargs: Any) -> Any:
        query = kwargs["query"]
        max_results = int(kwargs.get("max_results", 5))
        h = _sim_hash("search", query)

        results = []
        for i in range(max_results):
            h2 = _sim_hash("search_result", query, i)
            domain = self._DOMAINS[h2 % len(self._DOMAINS)]
            snippet = self._SNIPPET_TEMPLATES[h2 % len(self._SNIPPET_TEMPLATES)].format(q=query)
            source = ["Wikipedia", "News", "Blog", "Academic", "Forum"][h2 % 5]
            results.append({
                "title": f"{query.title()} — {source} {'Overview' if i == 0 else 'Analysis' if i == 1 else 'Guide' if i == 2 else 'Discussion' if i == 3 else 'Report'}",
                "url": f"https://{domain}/article/{h2 % 100000}",
                "snippet": snippet,
                "source": source,
            })

        return {"query": query, "results": results, "total": len(results)}


@register_tool
class DatabaseQuery(BaseTool):
    """Query a structured database."""

    name = "database_query"
    schema = ToolSchema(
        name="database_query",
        description="Query a structured database table. Supports filtering, sorting, and aggregation on predefined datasets (countries, cities, movies, books).",
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="table", type="string",
                description="The table/dataset to query",
                enum=["countries", "cities", "movies", "books"],
            ),
            ToolParameter(
                name="filter_field", type="string",
                description="Field to filter on (e.g. 'population', 'rating', 'year')",
                required=False,
            ),
            ToolParameter(
                name="filter_op", type="string",
                description="Filter operation",
                enum=["equals", "greater_than", "less_than", "contains"],
                required=False,
            ),
            ToolParameter(
                name="filter_value", type="string",
                description="Value to filter by",
                required=False,
            ),
            ToolParameter(
                name="sort_by", type="string",
                description="Field to sort results by",
                required=False,
            ),
            ToolParameter(
                name="limit", type="integer",
                description="Maximum number of results (default 10)",
                required=False, default=10,
            ),
        ],
        returns="Query results as a list of records",
        returns_type="object",
    )

    _DATA: dict[str, list[dict[str, Any]]] = {
        "countries": [
            {"name": "China", "population": 1412, "continent": "Asia", "gdp_trillion": 17.96, "capital": "Beijing"},
            {"name": "India", "population": 1408, "continent": "Asia", "gdp_trillion": 3.74, "capital": "New Delhi"},
            {"name": "United States", "population": 331, "continent": "North America", "gdp_trillion": 26.95, "capital": "Washington D.C."},
            {"name": "Indonesia", "population": 274, "continent": "Asia", "gdp_trillion": 1.32, "capital": "Jakarta"},
            {"name": "Brazil", "population": 214, "continent": "South America", "gdp_trillion": 2.13, "capital": "Brasília"},
            {"name": "Japan", "population": 125, "continent": "Asia", "gdp_trillion": 4.23, "capital": "Tokyo"},
            {"name": "Germany", "population": 84, "continent": "Europe", "gdp_trillion": 4.43, "capital": "Berlin"},
            {"name": "United Kingdom", "population": 67, "continent": "Europe", "gdp_trillion": 3.07, "capital": "London"},
            {"name": "France", "population": 68, "continent": "Europe", "gdp_trillion": 2.78, "capital": "Paris"},
            {"name": "Australia", "population": 26, "continent": "Oceania", "gdp_trillion": 1.69, "capital": "Canberra"},
        ],
        "cities": [
            {"name": "Tokyo", "country": "Japan", "population_million": 37.4, "timezone": "Asia/Tokyo", "avg_temp_c": 16},
            {"name": "Delhi", "country": "India", "population_million": 32.9, "timezone": "Asia/Kolkata", "avg_temp_c": 25},
            {"name": "Shanghai", "country": "China", "population_million": 28.5, "timezone": "Asia/Shanghai", "avg_temp_c": 17},
            {"name": "São Paulo", "country": "Brazil", "population_million": 22.4, "timezone": "America/Sao_Paulo", "avg_temp_c": 20},
            {"name": "Mexico City", "country": "Mexico", "population_million": 21.8, "timezone": "America/Mexico_City", "avg_temp_c": 17},
            {"name": "Cairo", "country": "Egypt", "population_million": 21.3, "timezone": "Africa/Cairo", "avg_temp_c": 22},
            {"name": "Mumbai", "country": "India", "population_million": 21.0, "timezone": "Asia/Kolkata", "avg_temp_c": 27},
            {"name": "Beijing", "country": "China", "population_million": 20.9, "timezone": "Asia/Shanghai", "avg_temp_c": 13},
            {"name": "New York", "country": "United States", "population_million": 18.8, "timezone": "America/New_York", "avg_temp_c": 13},
            {"name": "London", "country": "United Kingdom", "population_million": 9.5, "timezone": "Europe/London", "avg_temp_c": 11},
            {"name": "Paris", "country": "France", "population_million": 11.0, "timezone": "Europe/Paris", "avg_temp_c": 12},
            {"name": "Sydney", "country": "Australia", "population_million": 5.3, "timezone": "Australia/Sydney", "avg_temp_c": 18},
        ],
    }

    def execute_live(self, **kwargs: Any) -> Any:
        return self._query(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._query(**kwargs)

    def _query(self, **kwargs: Any) -> dict[str, Any]:
        table = kwargs["table"]
        data = [row.copy() for row in self._DATA.get(table, [])]

        if not data:
            return {"table": table, "results": [], "count": 0, "error": "Table not found"}

        # Filter
        field = kwargs.get("filter_field")
        op = kwargs.get("filter_op")
        value = kwargs.get("filter_value")

        if field and op and value is not None:
            filtered = []
            for row in data:
                if field not in row:
                    continue
                rv = row[field]
                if op == "equals":
                    if str(rv).lower() == str(value).lower():
                        filtered.append(row)
                elif op == "greater_than":
                    if float(rv) > float(value):
                        filtered.append(row)
                elif op == "less_than":
                    if float(rv) < float(value):
                        filtered.append(row)
                elif op == "contains":
                    if str(value).lower() in str(rv).lower():
                        filtered.append(row)
            data = filtered

        # Sort
        sort_by = kwargs.get("sort_by")
        if sort_by and data and sort_by in data[0]:
            data.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

        # Limit
        limit = int(kwargs.get("limit", 10))
        data = data[:limit]

        return {"table": table, "results": data, "count": len(data)}


@register_tool
class WebPageFetch(BaseTool):
    """Fetch and extract content from a web page."""

    name = "web_page_fetch"
    schema = ToolSchema(
        name="web_page_fetch",
        description="Fetch the content of a web page given its URL. Returns the page title, main text content, and metadata.",
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="url", type="string",
                description="The URL of the web page to fetch (e.g. 'https://example.com/article')",
            ),
            ToolParameter(
                name="extract_mode", type="string",
                description="What to extract from the page",
                enum=["text", "summary", "metadata"],
                default="text",
                required=False,
            ),
        ],
        returns="Page content with title, text, and metadata",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        mode = kwargs.get("extract_mode", "text")
        h = _sim_hash("webpage", url)

        # Generate deterministic page content based on URL
        title = f"Article about {url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()}"
        content = (
            f"This page covers the topic indicated by the URL. "
            f"The main discussion focuses on key developments and recent findings. "
            f"Several experts are quoted providing their analysis. "
            f"The article concludes with a summary of implications and future outlook."
        )

        result: dict[str, Any] = {
            "url": url,
            "title": title,
            "status_code": 200,
        }

        if mode == "text":
            result["content"] = content
            result["word_count"] = len(content.split())
        elif mode == "summary":
            result["summary"] = content[:150] + "..."
        elif mode == "metadata":
            result["author"] = "Staff Writer"
            result["published_date"] = "2026-03-10"
            result["language"] = "en"
            result["word_count"] = 850 + (h % 1500)

        return result


@register_tool
class KnowledgeBaseQuery(BaseTool):
    """Query a structured knowledge base."""

    name = "knowledge_base_query"
    schema = ToolSchema(
        name="knowledge_base_query",
        description="Query a structured knowledge base with factual information about entities, relationships, and concepts. Supports specific questions about people, places, events, and scientific topics.",
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="query", type="string",
                description="The factual question or topic to look up (e.g. 'capital of France', 'population of Japan')",
            ),
            ToolParameter(
                name="domain", type="string",
                description="Knowledge domain to search in",
                enum=["general", "science", "geography", "history", "technology"],
                default="general",
                required=False,
            ),
        ],
        returns="Factual answer with source and confidence",
        returns_type="object",
    )

    _KB_ENTRIES: dict[str, dict[str, Any]] = {
        "capital of france": {"answer": "Paris", "confidence": 1.0, "source": "Geography KB"},
        "capital of japan": {"answer": "Tokyo", "confidence": 1.0, "source": "Geography KB"},
        "capital of germany": {"answer": "Berlin", "confidence": 1.0, "source": "Geography KB"},
        "capital of australia": {"answer": "Canberra", "confidence": 1.0, "source": "Geography KB"},
        "capital of brazil": {"answer": "Brasília", "confidence": 1.0, "source": "Geography KB"},
        "population of japan": {"answer": "125 million (2024 est.)", "confidence": 0.95, "source": "Demographics KB"},
        "population of india": {"answer": "1.44 billion (2024 est.)", "confidence": 0.95, "source": "Demographics KB"},
        "speed of light": {"answer": "299,792,458 meters per second", "confidence": 1.0, "source": "Physics KB"},
        "boiling point of water": {"answer": "100°C (212°F) at standard atmospheric pressure", "confidence": 1.0, "source": "Chemistry KB"},
        "largest ocean": {"answer": "Pacific Ocean (165.25 million km²)", "confidence": 1.0, "source": "Geography KB"},
        "python creator": {"answer": "Guido van Rossum, first released in 1991", "confidence": 1.0, "source": "Technology KB"},
        "transformer architecture": {"answer": "Introduced in 'Attention Is All You Need' (Vaswani et al., 2017) by Google Brain", "confidence": 1.0, "source": "Technology KB"},
    }

    def execute_live(self, **kwargs: Any) -> Any:
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        query = kwargs["query"]
        domain = kwargs.get("domain", "general")
        query_lower = query.lower().strip()

        # Check exact and partial matches
        for key, entry in self._KB_ENTRIES.items():
            if key in query_lower or query_lower in key:
                return {
                    "query": query,
                    "domain": domain,
                    "answer": entry["answer"],
                    "confidence": entry["confidence"],
                    "source": entry["source"],
                    "found": True,
                }

        # Generic response for unknown queries
        h = _sim_hash("kb", query)
        return {
            "query": query,
            "domain": domain,
            "answer": f"Based on available knowledge, {query} relates to a topic with multiple facets. Further research may be needed for a complete answer.",
            "confidence": round(0.4 + (h % 30) / 100, 2),
            "source": f"{domain.title()} Knowledge Base",
            "found": True,
        }


@register_tool
class LookupEntity(BaseTool):
    """Look up information about a specific entity."""

    name = "lookup_entity"
    schema = ToolSchema(
        name="lookup_entity",
        description="Look up detailed information about a specific entity (person, place, concept, etc.) from a knowledge base.",
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="entity", type="string",
                description="The entity to look up, e.g. 'Albert Einstein' or 'Python programming'",
            ),
        ],
        returns="Detailed information about the entity",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        entity = kwargs["entity"]
        # Use Wikipedia API (free, no key)
        try:
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + entity.replace(" ", "_"),
                timeout=10,
                headers={"User-Agent": "CompToolBench/0.1"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "entity": entity,
                    "title": data.get("title", entity),
                    "description": data.get("description", ""),
                    "summary": data.get("extract", "")[:500],
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "source": "Wikipedia",
                }
            return self.execute_simulated(**kwargs)
        except Exception:
            return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        entity = kwargs["entity"]
        h = _sim_hash("entity", entity)
        return {
            "entity": entity,
            "title": entity.title(),
            "description": f"Information about {entity}",
            "summary": f"{entity} is a notable topic. It is known for various contributions and significance in its domain.",
            "url": f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}",
            "source": "Knowledge Base",
        }
