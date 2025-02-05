"""Define the state structures for the agent."""

from __future__ import annotations
from typing import List, Annotated
import operator
from dataclasses import dataclass, field

@dataclass
class SummaryState:
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    sources_gathered: Annotated[List[str], operator.add] = field(default_factory=list)
    web_search_results: Annotated[List[str], operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)


@dataclass
class SummaryInputState:
    research_topic: str = field(default=None)

@dataclass
class SummaryOutputState:
    running_summary: str = field(default=None)
