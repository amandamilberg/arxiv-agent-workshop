"""
tools.py - Tools for the AI agent to use

This module contains tools that extend the agent's capabilities beyond
simple text evaluation. These tools allow the agent to:
- Check citation impact of papers
- Create composite rankings
- Download PDFs for deeper analysis

Key concept: Tools turn a "text-in, text-out" LLM into an agent that
can take actions and gather external information.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


# =============================================================================
# Citation Impact Tools
# =============================================================================

def check_citation_impact(paper_title: str, paper_id: str = None) -> dict:
    """
    Check the citation count and impact of a paper using Semantic Scholar API.

    This gives us the "community's opinion" of a paper - how many researchers
    found it valuable enough to cite.

    Args:
        paper_title: The paper's title (used for search)
        paper_id: Optional arXiv ID for more precise lookup

    Returns:
        Dictionary with:
        - citations: Number of citations (0 if not found)
        - influential_citations: Citations that are particularly important
        - year: Publication year
        - found: Whether the paper was found

    Example:
        >>> impact = check_citation_impact("Attention Is All You Need")
        >>> impact['citations']
        85000
    """

    try:
        # Try arXiv ID first if available
        if paper_id:
            arxiv_id = paper_id.replace('v1', '').replace('v2', '').replace('v3', '')
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = "?fields=citationCount,influentialCitationCount,year,title"

            req = urllib.request.Request(
                url + params,
                headers={'User-Agent': 'ArXiv-Workshop/1.0'}
            )

            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    return {
                        'citations': data.get('citationCount', 0) or 0,
                        'influential_citations': data.get('influentialCitationCount', 0) or 0,
                        'year': data.get('year'),
                        'found': True
                    }
            except urllib.error.HTTPError:
                pass  # Fall through to title search

        # Fall back to title search
        encoded_title = urllib.parse.quote(paper_title)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_title}&limit=1&fields=citationCount,influentialCitationCount,year,title"

        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'ArXiv-Workshop/1.0'}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            if data.get('data') and len(data['data']) > 0:
                paper = data['data'][0]
                return {
                    'citations': paper.get('citationCount', 0) or 0,
                    'influential_citations': paper.get('influentialCitationCount', 0) or 0,
                    'year': paper.get('year'),
                    'found': True
                }

    except Exception as e:
        pass  # Return default values on any error

    return {
        'citations': 0,
        'influential_citations': 0,
        'year': None,
        'found': False
    }


def get_citations_batch(papers: list[dict], verbose: bool = True) -> list[dict]:
    """
    Get citation counts for a batch of papers.

    Args:
        papers: List of paper dictionaries with 'title' and 'id' keys
        verbose: Whether to print progress

    Returns:
        List of papers with citation data added

    Example:
        >>> papers_with_citations = get_citations_batch(papers)
        >>> papers_with_citations[0]['citations']
        150
    """

    import time

    results = []

    for i, paper in enumerate(papers):
        if verbose:
            print(f"  [{i+1}/{len(papers)}] Checking: {paper['title'][:50]}...")

        impact = check_citation_impact(paper['title'], paper.get('id'))

        # Add citation data to paper
        paper_with_citations = {**paper, **impact}
        results.append(paper_with_citations)

        # Rate limiting - Semantic Scholar allows ~100 requests/5 min
        time.sleep(0.5)

    if verbose:
        found_count = sum(1 for p in results if p.get('found', False))
        print(f"\n✓ Found citation data for {found_count}/{len(papers)} papers")

    return results


# =============================================================================
# Composite Scoring
# =============================================================================

def create_composite_score(
    llm_score: float,
    citation_count: int,
    llm_weight: float = 0.7,
    citation_weight: float = 0.3,
    max_citations: int = None
) -> float:
    """
    Create a composite score combining LLM relevance and citation impact.

    This addresses the "everything is 7-8" problem by incorporating
    external signals that have higher variance.

    Args:
        llm_score: The LLM's relevance score (0-10)
        citation_count: Number of citations
        llm_weight: Weight for LLM score (default 0.7)
        citation_weight: Weight for citations (default 0.3)
        max_citations: Citation count for normalization. If None, uses 50
                      (appropriate for recent papers). Use 500+ for older papers.

    Returns:
        Composite score (0-10 scale)

    Example:
        >>> create_composite_score(llm_score=8, citation_count=20, max_citations=50)
        8.0  # Weighted combination
    """

    # Default max_citations for recent papers (last 7-30 days)
    if max_citations is None:
        max_citations = 50

    # Normalize citations to 0-10 scale (log scale for better distribution)
    import math

    if citation_count <= 0:
        citation_normalized = 0
    else:
        # Log scale normalizes citations relative to max_citations
        # For max=50: 1 citation = ~0, 5 = ~4.1, 10 = ~5.9, 25 = ~8.2, 50 = 10
        citation_normalized = min(10, (math.log10(citation_count + 1) / math.log10(max_citations + 1)) * 10)

    # Weighted combination
    composite = (llm_score * llm_weight) + (citation_normalized * citation_weight)

    return round(composite, 2)


def add_composite_scores(
    papers: list[dict],
    llm_weight: float = 0.7,
    citation_weight: float = 0.3,
    max_citations: int = None
) -> list[dict]:
    """
    Add composite scores to a list of papers.

    Papers must have 'relevance_score' and 'citations' keys.
    Papers with 'found'=False (not in citation database) use LLM score only.

    Args:
        papers: List of paper dictionaries
        llm_weight: Weight for LLM relevance score
        citation_weight: Weight for citation count
        max_citations: Max citations for normalization. If None, auto-detects
                      from data (uses max found * 1.5, minimum 20)

    Returns:
        Papers with 'composite_score' added, sorted by composite score

    Example:
        >>> ranked = add_composite_scores(papers_with_citations)
        >>> ranked[0]['composite_score']
        9.2
    """

    # Only consider papers that were actually found for max_citations calculation
    found_papers = [p for p in papers if p.get('found', True)]

    # Auto-detect max_citations if not provided
    if max_citations is None:
        if found_papers:
            actual_max = max((p.get('citations', 0) for p in found_papers), default=0)
            # Use 1.5x the actual max (so top paper doesn't get perfect 10)
            # Minimum of 20 to handle papers with very few citations
            max_citations = max(20, int(actual_max * 1.5))
        else:
            max_citations = 20
        print(f"   (Auto-detected max_citations={max_citations} from {len(found_papers)} found papers)")

    # Calculate median citations for unfound papers (neutral value)
    if found_papers:
        sorted_citations = sorted(p.get('citations', 0) for p in found_papers)
        median_citations = sorted_citations[len(sorted_citations) // 2]
    else:
        median_citations = 0

    results = []
    unfound_count = 0

    for paper in papers:
        llm_score = paper.get('relevance_score', 5)
        was_found = paper.get('found', True)

        if was_found:
            # Paper was found - use actual citation count
            citations = paper.get('citations', 0)
        else:
            # Paper NOT found in database - use median (don't penalize)
            citations = median_citations
            unfound_count += 1

        composite = create_composite_score(
            llm_score=llm_score,
            citation_count=citations,
            llm_weight=llm_weight,
            citation_weight=citation_weight,
            max_citations=max_citations
        )

        results.append({**paper, 'composite_score': composite})

    if unfound_count > 0:
        print(f"   ({unfound_count} papers not found in citation DB - using median={median_citations})")

    # Sort by composite score (highest first)
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    return results


# =============================================================================
# PDF Tools
# =============================================================================

def fetch_paper_pdf(paper_id: str, output_dir: str = "./papers") -> Optional[str]:
    """
    Download the PDF for an arXiv paper.

    This allows the agent to "investigate deeply" - reading the full paper
    rather than just the abstract.

    Args:
        paper_id: arXiv paper ID (e.g., "2401.12345" or "2401.12345v1")
        output_dir: Directory to save PDFs

    Returns:
        Path to downloaded PDF, or None if download failed

    Example:
        >>> path = fetch_paper_pdf("2401.12345")
        >>> path
        './papers/2401.12345.pdf'
    """

    # Clean up paper ID
    clean_id = paper_id.split('/')[-1]  # Handle full URLs
    clean_id = re.sub(r'v\d+$', '', clean_id)  # Remove version suffix

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download PDF
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    pdf_path = output_path / f"{clean_id}.pdf"

    try:
        req = urllib.request.Request(
            pdf_url,
            headers={'User-Agent': 'ArXiv-Workshop/1.0'}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            with open(pdf_path, 'wb') as f:
                f.write(response.read())

        return str(pdf_path)

    except Exception as e:
        print(f"  ⚠️ Failed to download {clean_id}: {e}")
        return None


def fetch_papers_batch(
    papers: list[dict],
    output_dir: str = "./papers",
    verbose: bool = True
) -> list[dict]:
    """
    Download PDFs for multiple papers.

    Args:
        papers: List of paper dictionaries with 'id' key
        output_dir: Directory to save PDFs
        verbose: Whether to print progress

    Returns:
        Papers with 'pdf_path' added (None if download failed)
    """

    import time

    results = []

    for i, paper in enumerate(papers):
        if verbose:
            print(f"  [{i+1}/{len(papers)}] Downloading: {paper['id']}...")

        pdf_path = fetch_paper_pdf(paper['id'], output_dir)
        results.append({**paper, 'pdf_path': pdf_path})

        # Be nice to arXiv servers
        time.sleep(1)

    if verbose:
        success_count = sum(1 for p in results if p.get('pdf_path'))
        print(f"\n✓ Downloaded {success_count}/{len(papers)} PDFs")

    return results


# =============================================================================
# Ranking Comparison
# =============================================================================

def compare_rankings(
    papers: list[dict],
    top_n: int = 10
) -> dict:
    """
    Compare different ranking strategies side-by-side.

    Shows how LLM-only, citation-only, and composite rankings differ.

    Args:
        papers: Papers with relevance_score, citations, and composite_score
        top_n: Number of top papers to compare

    Returns:
        Dictionary with three ranked lists for comparison
    """

    # LLM-only ranking
    llm_ranked = sorted(papers, key=lambda x: x.get('relevance_score', 0), reverse=True)[:top_n]

    # Citation-only ranking
    citation_ranked = sorted(papers, key=lambda x: x.get('citations', 0), reverse=True)[:top_n]

    # Composite ranking
    composite_ranked = sorted(papers, key=lambda x: x.get('composite_score', 0), reverse=True)[:top_n]

    return {
        'llm_ranking': llm_ranked,
        'citation_ranking': citation_ranked,
        'composite_ranking': composite_ranked
    }


def print_ranking_comparison(rankings: dict, show_top: int = 5):
    """
    Print a side-by-side comparison of ranking strategies.
    """

    print("=" * 100)
    print(f"{'LLM Ranking':<33} | {'Citation Ranking':<33} | {'Composite Ranking':<33}")
    print("=" * 100)

    for i in range(show_top):
        llm = rankings['llm_ranking'][i] if i < len(rankings['llm_ranking']) else None
        cit = rankings['citation_ranking'][i] if i < len(rankings['citation_ranking']) else None
        comp = rankings['composite_ranking'][i] if i < len(rankings['composite_ranking']) else None

        llm_str = f"{llm['title'][:25]}... ({llm['relevance_score']})" if llm else ""
        cit_str = f"{cit['title'][:25]}... ({cit['citations']})" if cit else ""
        comp_str = f"{comp['title'][:25]}... ({comp['composite_score']})" if comp else ""

        print(f"{i+1}. {llm_str:<30} | {cit_str:<30} | {comp_str:<30}")

    print("=" * 100)
