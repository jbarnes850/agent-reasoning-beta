"""
Web search integration for the Agent Reasoning Beta platform.

This module provides:
- Tavily search integration with:
  - General search
  - Context generation for RAG
  - Q&A search
- Result parsing and credibility scoring
- Search result caching
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from urllib.parse import quote_plus

from pydantic import BaseModel, Field
from tavily import TavilyClient

from .tools import measure_performance
from .types import ToolMetrics


class SearchResult(BaseModel):
    """Individual search result with metadata."""

    title: str
    url: str
    snippet: str
    source: str = "tavily"
    credibility_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchCache(BaseModel):
    """Cache for search results."""

    data: Union[List[SearchResult], str]  # Can be list of results or context string
    query: str
    expiry: datetime
    search_type: str


class SearchManager:
    """Manager for web search operations using Tavily."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_duration: timedelta = timedelta(hours=1),
    ):
        self._client = TavilyClient(api_key=api_key) if api_key else None
        self._cache: Dict[str, SearchCache] = {}
        self._cache_duration = cache_duration
        self._metrics = ToolMetrics()

    def set_api_key(self, api_key: str) -> None:
        """Set Tavily API key."""
        self._client = TavilyClient(api_key=api_key)

    @measure_performance
    async def search(
        self, query: str, max_results: int = 5, use_cache: bool = True
    ) -> List[SearchResult]:
        """Perform a general search using Tavily."""
        if not self._client:
            raise RuntimeError("Tavily API key not set")

        cache_key = f"search:{query}:{max_results}"

        if use_cache and cache_key in self._cache:
            cache = self._cache[cache_key]
            if cache.expiry > datetime.utcnow():
                return cache.data  # type: ignore

        # Run in thread pool to avoid blocking
        response = await asyncio.get_event_loop().run_in_executor(
            None, self._client.search, query
        )

        results = []
        for item in response.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="tavily",
                    credibility_score=item.get("score", 0.5),
                )
            )

        self._cache[cache_key] = SearchCache(
            data=results,
            query=query,
            expiry=datetime.utcnow() + self._cache_duration,
            search_type="search",
        )

        return results

    @measure_performance
    async def get_context(self, query: str, use_cache: bool = True) -> str:
        """Get search context for RAG applications."""
        if not self._client:
            raise RuntimeError("Tavily API key not set")

        cache_key = f"context:{query}"

        if use_cache and cache_key in self._cache:
            cache = self._cache[cache_key]
            if cache.expiry > datetime.utcnow():
                return cache.data  # type: ignore

        # Run in thread pool to avoid blocking
        context = await asyncio.get_event_loop().run_in_executor(
            None, self._client.get_search_context, query
        )

        self._cache[cache_key] = SearchCache(
            data=context,
            query=query,
            expiry=datetime.utcnow() + self._cache_duration,
            search_type="context",
        )

        return context

    @measure_performance
    async def qna_search(self, query: str, use_cache: bool = True) -> str:
        """Get a quick answer to a question."""
        if not self._client:
            raise RuntimeError("Tavily API key not set")

        cache_key = f"qna:{query}"

        if use_cache and cache_key in self._cache:
            cache = self._cache[cache_key]
            if cache.expiry > datetime.utcnow():
                return cache.data  # type: ignore

        # Run in thread pool to avoid blocking
        answer = await asyncio.get_event_loop().run_in_executor(
            None, self._client.qna_search, query
        )

        self._cache[cache_key] = SearchCache(
            data=answer,
            query=query,
            expiry=datetime.utcnow() + self._cache_duration,
            search_type="qna",
        )

        return answer

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()

    @property
    def metrics(self) -> ToolMetrics:
        """Get search metrics."""
        return self._metrics
