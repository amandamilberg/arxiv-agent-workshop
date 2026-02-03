"""
get_papers.py - Fetch research papers from ArXiv

This module handles searching the ArXiv API for papers matching
specific keywords. It's the "input" stage of our pipeline.

Key concepts:
- ArXiv is a free repository of research papers
- We search by keywords and filter by date
- Papers are returned as simple dictionaries (notebook-friendly)
"""

import arxiv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


def get_papers(
    keywords: list[str],
    max_results: int = 20,
    days_back: int = 7
) -> list[dict]:
    """
    Fetch recent papers from ArXiv matching the given keywords.

    This function is designed to be notebook-friendly - it returns data
    instead of printing, so you can easily work with results in Jupyter.

    Args:
        keywords: List of search terms (e.g., ["LLM", "transformers"])
        max_results: Maximum number of papers to return (default: 20)
        days_back: How many days back to search (default: 7)

    Returns:
        List of paper dictionaries with keys:
        - id: ArXiv paper ID (e.g., "2401.12345v1")
        - title: Paper title
        - authors: List of author names
        - abstract: Paper abstract/summary
        - published: Publication date as ISO string
        - url: Link to the paper

    Example:
        >>> papers = get_papers(["large language models"], max_results=5)
        >>> len(papers)
        5
        >>> papers[0].keys()
        dict_keys(['id', 'title', 'authors', 'abstract', 'published', 'url'])
    """

    # Calculate cutoff date for filtering
    cutoff_date = datetime.now() - timedelta(days=days_back)

    # Build search query: "keyword1" OR "keyword2" OR ...
    # Quotes ensure exact phrase matching
    query = " OR ".join([f'"{kw}"' for kw in keywords])

    # Configure the ArXiv search
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # Most recent first
        sort_order=arxiv.SortOrder.Descending
    )

    # Execute search and collect results
    client = arxiv.Client()
    papers = []

    for paper in client.results(search):
        # Only include papers from our time window
        paper_date = paper.published.replace(tzinfo=None)
        if paper_date >= cutoff_date:
            papers.append({
                'id': paper.entry_id.split('/')[-1],  # e.g., "2401.12345v1"
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'published': paper.published.isoformat(),
                'url': paper.entry_id
            })

    return papers


def load_sample_papers(filepath: str = None) -> list[dict]:
    """
    Load pre-saved sample papers for offline demos.

    Use this when you want to:
    - Demo without API calls
    - Test quickly without waiting for ArXiv
    - Have consistent reproducible results

    Args:
        filepath: Path to the sample papers JSON file.
                  If None, uses the default sample_data/papers.json

    Returns:
        List of paper dictionaries (same format as get_papers)

    Example:
        >>> papers = load_sample_papers()
        >>> len(papers)
        20
    """

    if filepath is None:
        # Default to sample_data/papers.json relative to this file's location
        script_dir = Path(__file__).parent.parent
        filepath = script_dir / "sample_data" / "papers.json"

    with open(filepath, 'r') as f:
        papers = json.load(f)

    return papers


def save_papers(papers: list[dict], filepath: str) -> None:
    """
    Save papers to a JSON file for later use.

    Useful for creating your own sample data or caching results.

    Args:
        papers: List of paper dictionaries
        filepath: Where to save the JSON file

    Example:
        >>> papers = get_papers(["LLM"], max_results=20)
        >>> save_papers(papers, "my_papers.json")
    """

    with open(filepath, 'w') as f:
        json.dump(papers, f, indent=2)
