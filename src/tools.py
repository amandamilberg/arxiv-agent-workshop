"""
tools.py - Tools for the AI agent to use

This module contains tools that extend the agent's capabilities beyond
simple text evaluation. These tools allow the agent to:
- Download PDFs for deeper analysis
- Investigate papers the agent is uncertain about
- Extract text from PDFs for LLM analysis

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
        print(f"  Warning: Failed to download {clean_id}: {e}")
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
# Deep Analysis Tools
# =============================================================================

def extract_pdf_text(pdf_path: str, max_pages: int = 10) -> Optional[str]:
    """
    Extract text content from a PDF file.

    Requires PyMuPDF (fitz) to be installed: pip install pymupdf

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default 10)

    Returns:
        Extracted text content, or None if extraction failed

    Example:
        >>> text = extract_pdf_text("./papers/2401.12345.pdf")
        >>> len(text)
        45000
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  Warning: PyMuPDF not installed. Run: pip install pymupdf")
        return None

    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text_parts.append(page.get_text())

        doc.close()
        return "\n\n".join(text_parts)

    except Exception as e:
        print(f"  Warning: Failed to extract text from {pdf_path}: {e}")
        return None


def get_paper_for_deep_analysis(
    paper: dict,
    output_dir: str = "./papers"
) -> dict:
    """
    Download a paper and extract its text for deep analysis.

    This is the main tool for investigating papers the agent is uncertain about.
    It downloads the PDF and extracts the text so the LLM can read the full paper.

    Args:
        paper: Paper dictionary with 'id' key
        output_dir: Directory to save PDFs

    Returns:
        Paper dictionary with 'pdf_path' and 'full_text' added

    Example:
        >>> paper = {"id": "2401.12345", "title": "Some Paper"}
        >>> result = get_paper_for_deep_analysis(paper)
        >>> len(result['full_text'])
        45000
    """

    # Download the PDF
    pdf_path = fetch_paper_pdf(paper['id'], output_dir)

    result = {**paper, 'pdf_path': pdf_path, 'full_text': None}

    if pdf_path:
        # Extract text from the PDF
        full_text = extract_pdf_text(pdf_path)
        result['full_text'] = full_text

        if full_text:
            print(f"  ✓ Extracted {len(full_text):,} characters from {paper['id']}")
        else:
            print(f"  Warning: Could not extract text from {paper['id']}")

    return result


def get_borderline_papers(evaluated_papers: list[dict], score: int = 7) -> list[dict]:
    """
    Get papers with a borderline score that need deeper investigation.

    Papers scoring exactly 7 are "on the fence" - reading the full paper
    might reveal they're actually an 8-9 (recommend) or a 5-6 (skip).

    Args:
        evaluated_papers: Papers with 'relevance_score'
        score: The borderline score to filter on (default: 7)

    Returns:
        List of papers with the borderline score

    Example:
        >>> borderline = get_borderline_papers(papers)
        >>> len(borderline)
        4
    """

    return [p for p in evaluated_papers if p.get('relevance_score') == score]


# =============================================================================
# Tool Definitions for Agent
# =============================================================================

# These are the tool definitions that can be passed to an LLM for function calling

TOOL_DEFINITIONS = [
    {
        "name": "fetch_paper_pdf",
        "description": "Download the full PDF of an arXiv paper for deeper analysis. Use this when you need more information than the abstract provides to make a relevance judgment.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The arXiv paper ID (e.g., '2401.12345')"
                }
            },
            "required": ["paper_id"]
        }
    },
    {
        "name": "analyze_full_paper",
        "description": "Get the full text of a paper for detailed analysis. Returns the extracted text content from the PDF. Use this when the abstract is unclear or you need to verify specific claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The arXiv paper ID to analyze"
                }
            },
            "required": ["paper_id"]
        }
    }
]


def get_tool_definitions() -> list[dict]:
    """
    Get the tool definitions for function calling.

    Returns:
        List of tool definitions in OpenAI function calling format
    """
    return TOOL_DEFINITIONS
