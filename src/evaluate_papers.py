"""
evaluate_papers.py - Evaluate papers using OpenAI-compatible Batch API

This module handles the AI evaluation of research papers using an
OpenAI-compatible Batch API (works with OpenAI, Azure, or other providers).

Key concepts:
- Batch API: Send many requests at once, get results later (cheaper!)
- Structured output: We ask the model to return JSON
- Team profile: Customizable criteria for relevance scoring

The Batch API is typically 50% cheaper than real-time API calls and
is perfect for non-interactive workloads like paper evaluation.
"""

import json
import time
import re
import os
from openai import OpenAI
from pathlib import Path


# =============================================================================
# TEAM PROFILE - Customize this for your use case!
# =============================================================================
# This defines what makes a paper "relevant" for your team.
# The AI will use this to score papers from 0-10.

TEAM_PROFILE = {
    "focus": """
        The team is building AI-powered applications and wants to stay
        current with the latest research on language models, inference
        optimization, and practical AI engineering.
    """,
    "interests": [
        "Large language model architectures and improvements",
        "Inference optimization and cost reduction",
        "Prompt engineering and techniques",
        "AI agents and tool use",
        "Evaluation methods for LLMs",
    ],
    "avoid": [
        "Pure theoretical papers without practical applications",
        "Incremental benchmark improvements",
        "Papers focused only on training from scratch",
    ]
}


def create_evaluation_prompt(paper: dict, team_profile: dict = None) -> str:
    """
    Create the prompt that asks the model to evaluate a paper.

    This is where the "magic" happens - we give the model:
    1. Context about what the team cares about
    2. The paper's title and abstract
    3. Clear instructions on how to respond

    Args:
        paper: Dictionary with 'title' and 'abstract' keys
        team_profile: Dictionary with 'focus', 'interests', and 'avoid' keys
                     If None, uses the default TEAM_PROFILE

    Returns:
        A formatted prompt string
    """

    if team_profile is None:
        team_profile = TEAM_PROFILE

    # Format the interests and avoid lists as bullet points
    interests_text = "\n".join(f"  - {item}" for item in team_profile['interests'])
    avoid_text = "\n".join(f"  - {item}" for item in team_profile['avoid'])

    prompt = f"""You are evaluating research papers for an AI engineering team.

TEAM PROFILE:
{team_profile['focus'].strip()}

What they find valuable:
{interests_text}

What to avoid recommending:
{avoid_text}

---

PAPER TO EVALUATE:

Title: {paper['title']}

Abstract:
{paper['abstract']}

---

INSTRUCTIONS:
Score this paper's relevance to the team on a scale of 0-10.
- 0-3: Not relevant
- 4-6: Somewhat relevant
- 7: Borderline - might be relevant, needs closer look
- 8-10: Highly relevant, team should read this

Respond with ONLY valid JSON in this exact format:
{{
    "relevance_score": <integer 0-10>,
    "key_insight": "<one sentence explaining the main takeaway>",
    "why_relevant": "<one sentence explaining why this score>"
}}"""

    return prompt


def create_batch_file(
    papers: list[dict],
    model: str,
    output_path: str = "batch_input.jsonl",
    team_profile: dict = None
) -> str:
    """
    Create a JSONL file for the OpenAI-compatible Batch API.

    The Batch API expects a file where each line is a JSON object
    representing one request. This creates the input file.

    Args:
        papers: List of paper dictionaries to evaluate
        model: Model name to use (e.g., "gpt-4o-mini")
        output_path: Where to save the JSONL file
        team_profile: Optional custom team profile

    Returns:
        Path to the created file

    Example:
        >>> create_batch_file(papers, "gpt-4o-mini", "my_batch.jsonl")
        'my_batch.jsonl'
    """

    with open(output_path, 'w') as f:
        for paper in papers:
            # Each request needs a unique ID so we can match responses
            # OpenAI Batch API format
            request = {
                "custom_id": paper['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "user",
                            "content": create_evaluation_prompt(paper, team_profile)
                        }
                    ]
                }
            }
            f.write(json.dumps(request) + '\n')

    return output_path


def submit_batch(client: OpenAI, input_file: str, completion_window: str = "24h") -> str:
    """
    Submit a batch job to the OpenAI-compatible API.

    This uploads the file and starts processing. The batch will
    run in the background - we'll poll for completion.

    Args:
        client: OpenAI API client (or compatible client)
        input_file: Path to the JSONL batch file
        completion_window: SLA for batch completion (e.g., "1h", "24h")

    Returns:
        Batch ID for tracking

    Example:
        >>> client = OpenAI()
        >>> batch_id = submit_batch(client, "batch_input.jsonl")
        >>> print(batch_id)
        'batch_abc123'
    """

    # Upload the file
    with open(input_file, 'rb') as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="batch"
        )

    # Create the batch job
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window
    )

    return batch.id


def wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval: int = 10,
    verbose: bool = True
) -> list[dict]:
    """
    Wait for a batch to complete and return the results.

    This polls the API until the batch is done, then downloads
    and parses all the results.

    Args:
        client: OpenAI API client (or compatible client)
        batch_id: The batch ID to wait for
        poll_interval: Seconds between status checks (default: 10)
        verbose: Whether to print progress updates (default: True)

    Returns:
        List of evaluation results

    Example:
        >>> results = wait_for_batch(client, "batch_abc123")
        >>> len(results)
        20
    """

    while True:
        # Check current status
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        if verbose:
            completed = (batch.request_counts.completed or 0)
            failed = (batch.request_counts.failed or 0)
            total = batch.request_counts.total or 0
            print(f"Status: {status} | Progress: {completed + failed}/{total}")

        if status == "completed":
            break
        elif status in ["failed", "expired", "cancelled"]:
            raise Exception(f"Batch failed with status: {status}")

        time.sleep(poll_interval)

    # Download and parse results
    return get_batch_results(client, batch_id)


def get_batch_results(client: OpenAI, batch_id: str) -> list[dict]:
    """
    Download and parse results from a completed batch.

    Args:
        client: OpenAI API client (or compatible client)
        batch_id: The completed batch ID

    Returns:
        List of evaluation dictionaries with paper_id added

    Example:
        >>> results = get_batch_results(client, "batch_abc123")
        >>> results[0].keys()
        dict_keys(['paper_id', 'relevance_score', 'key_insight', 'why_relevant'])
    """

    batch = client.batches.retrieve(batch_id)

    if not batch.output_file_id:
        raise Exception("No output file available")

    # Download the results file
    result_content = client.files.content(batch.output_file_id)

    results = []

    # Parse each line of the JSONL response
    for line in result_content.text.strip().split('\n'):
        if not line:
            continue

        data = json.loads(line)
        paper_id = data['custom_id']

        # Check if this request succeeded
        if data.get('error'):
            continue

        # Extract the text response from the model
        # OpenAI format: response -> body -> choices[0] -> message -> content
        response_body = data.get('response', {}).get('body', {})
        choices = response_body.get('choices', [])

        if not choices:
            continue

        content = choices[0].get('message', {}).get('content', '')

        # Parse the JSON from the model's response
        evaluation = parse_json_response(content)

        if evaluation:
            evaluation['paper_id'] = paper_id
            results.append(evaluation)

    return results


def parse_json_response(content: str) -> dict | None:
    """
    Extract JSON from the model's response.

    The model usually returns clean JSON, but sometimes includes
    extra text or formatting. This function handles those cases.

    Args:
        content: Raw text response from the model

    Returns:
        Parsed dictionary, or None if parsing failed
    """

    content = content.strip()

    # Try direct parsing first (most common case)
    if content.startswith('{'):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    # Try to find JSON anywhere in the response
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def evaluate_papers_batch(
    papers: list[dict],
    model: str = None,
    team_profile: dict = None,
    api_key: str = None,
    base_url: str = None,
    poll_interval: int = 10,
    completion_window: str = "1h",
    verbose: bool = True
) -> list[dict]:
    """
    High-level function to evaluate papers using the Batch API.

    This is the main function you'll use in notebooks. It handles:
    1. Creating the batch file
    2. Submitting the batch
    3. Waiting for completion
    4. Returning parsed results

    Args:
        papers: List of paper dictionaries to evaluate
        model: Model name (default: from MODEL_NAME env var or "gpt-4o-mini")
        team_profile: Custom team profile (default: uses TEAM_PROFILE)
        api_key: API key (default: from OPENAI_API_KEY env var)
        base_url: Base URL for OpenAI-compatible API (default: from OPENAI_BASE_URL env var)
        poll_interval: Seconds between status checks
        completion_window: SLA for batch completion (e.g., "1h", "24h")
        verbose: Whether to print progress

    Returns:
        List of evaluation dictionaries, each containing:
        - paper_id: The ArXiv paper ID
        - relevance_score: Integer 0-10
        - key_insight: One sentence summary
        - why_relevant: Explanation of the score

    Example:
        >>> from get_papers import get_papers
        >>> papers = get_papers(["LLM"], max_results=5)
        >>> results = evaluate_papers_batch(papers)
        >>> results[0]['relevance_score']
        8
    """

    # Get configuration from environment if not provided
    if model is None:
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # Create the client
    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    if base_url:
        client_kwargs['base_url'] = base_url

    client = OpenAI(**client_kwargs)

    # Create batch file in a temp location
    batch_file = "batch_input.jsonl"
    create_batch_file(papers, model, batch_file, team_profile)

    if verbose:
        print(f"Submitting batch of {len(papers)} papers...")

    # Submit and wait
    batch_id = submit_batch(client, batch_file, completion_window)

    if verbose:
        print(f"Batch ID: {batch_id}")
        print("Waiting for completion...")

    results = wait_for_batch(client, batch_id, poll_interval, verbose)

    # Clean up temp file
    if os.path.exists(batch_file):
        os.remove(batch_file)

    return results


def merge_results_with_papers(
    papers: list[dict],
    evaluations: list[dict]
) -> list[dict]:
    """
    Combine paper data with evaluation results.

    This creates a unified view with both paper metadata and
    relevance scores, sorted by relevance.

    Args:
        papers: List of paper dictionaries
        evaluations: List of evaluation dictionaries

    Returns:
        List of merged dictionaries sorted by relevance_score (highest first)

    Example:
        >>> merged = merge_results_with_papers(papers, evaluations)
        >>> merged[0].keys()
        dict_keys(['id', 'title', 'authors', 'abstract', 'published',
                   'url', 'relevance_score', 'key_insight', 'why_relevant'])
    """

    # Create lookup for evaluations by paper_id
    eval_lookup = {e['paper_id']: e for e in evaluations}

    merged = []
    for paper in papers:
        paper_id = paper['id']
        if paper_id in eval_lookup:
            # Combine paper data with evaluation
            combined = {**paper, **eval_lookup[paper_id]}
            # Remove redundant paper_id (we already have 'id')
            combined.pop('paper_id', None)
            merged.append(combined)

    # Sort by relevance score (highest first)
    merged.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    return merged


def create_reevaluation_prompt(paper: dict, full_text: str, team_profile: dict = None) -> str:
    """
    Create a prompt to re-evaluate a paper using its full text.

    This is used for borderline papers (score=7) where the abstract
    wasn't clear enough to make a confident judgment.
    """

    if team_profile is None:
        team_profile = TEAM_PROFILE

    interests_text = "\n".join(f"  - {item}" for item in team_profile['interests'])

    # Truncate full text if too long (keep first ~8000 chars)
    if len(full_text) > 8000:
        full_text = full_text[:8000] + "\n\n[... truncated for length ...]"

    prompt = f"""You previously evaluated this paper based on its abstract and gave it a score of {paper.get('relevance_score', 7)}.

Your initial assessment: "{paper.get('why_relevant', 'Borderline relevance')}"

Now you have access to the FULL PAPER TEXT. Re-evaluate whether this paper is relevant to the team.

TEAM INTERESTS:
{interests_text}

PAPER TITLE: {paper['title']}

FULL PAPER TEXT:
{full_text}

---

Based on the full paper content, provide a REVISED score and assessment.
The score may go UP (if the paper is more relevant than the abstract suggested) or DOWN (if it's less relevant).

Respond with ONLY valid JSON:
{{
    "revised_score": <integer 0-10>,
    "key_insight": "<what you learned from reading the full paper>",
    "why_revised": "<why the score changed (or stayed the same)>"
}}"""

    return prompt


def create_reevaluation_batch_file(
    papers_with_text: list[dict],
    model: str,
    output_path: str = "reevaluation_batch.jsonl",
    team_profile: dict = None
) -> str:
    """
    Create a JSONL batch file for re-evaluating papers with full text.

    Args:
        papers_with_text: List of paper dicts with 'full_text' key
        model: Model name
        output_path: Where to save the JSONL file
        team_profile: Optional custom team profile

    Returns:
        Path to the created file
    """

    with open(output_path, 'w') as f:
        for paper in papers_with_text:
            if not paper.get('full_text'):
                continue

            request = {
                "custom_id": paper['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "user",
                            "content": create_reevaluation_prompt(
                                paper, paper['full_text'], team_profile
                            )
                        }
                    ]
                }
            }
            f.write(json.dumps(request) + '\n')

    return output_path


def reevaluate_papers_batch(
    papers_with_text: list[dict],
    model: str = None,
    team_profile: dict = None,
    api_key: str = None,
    base_url: str = None,
    poll_interval: int = 10,
    completion_window="1h",
    verbose: bool = True
) -> list[dict]:
    """
    Re-evaluate multiple papers using the Batch API.

    Args:
        papers_with_text: Papers with 'full_text' from PDF extraction
        model: Model name (default: from MODEL_NAME env var)
        team_profile: Custom team profile
        api_key: API key
        base_url: Base URL for API
        poll_interval: Seconds between status checks
        completion_window: SLA for submitting the batch api 
        verbose: Print progress

    Returns:
        List of re-evaluation results with paper_id, revised_score, etc.
    """

    if model is None:
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # Filter to papers that have full text
    papers_to_eval = [p for p in papers_with_text if p.get('full_text')]

    if not papers_to_eval:
        if verbose:
            print("No papers with full text to re-evaluate")
        return []

    # Create client
    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    if base_url:
        client_kwargs['base_url'] = base_url

    client = OpenAI(**client_kwargs)

    # Create batch file
    batch_file = "reevaluation_batch.jsonl"
    create_reevaluation_batch_file(papers_to_eval, model, batch_file, team_profile)

    if verbose:
        print(f"Submitting batch of {len(papers_to_eval)} papers for re-evaluation...")

    # Submit and wait
    batch_id = submit_batch(client, batch_file, completion_window)

    if verbose:
        print(f"Batch ID: {batch_id}")
        print("Waiting for completion...")

    results = wait_for_batch(client, batch_id, poll_interval, verbose)

    # Clean up
    if os.path.exists(batch_file):
        os.remove(batch_file)

    # Parse results - they have revised_score instead of relevance_score
    parsed_results = []
    for r in results:
        parsed_results.append({
            'paper_id': r.get('paper_id'),
            'revised_score': r.get('revised_score'),
            'key_insight': r.get('key_insight'),
            'why_revised': r.get('why_revised')
        })

    return parsed_results


def reevaluate_paper(
    paper: dict,
    full_text: str,
    model: str = None,
    team_profile: dict = None,
    api_key: str = None,
    base_url: str = None
) -> dict:
    """
    Re-evaluate a single borderline paper using its full text.
    For batch processing, use reevaluate_papers_batch() instead.

    Args:
        paper: Paper dict with title, relevance_score, etc.
        full_text: Extracted text from the PDF
        model: Model to use
        team_profile: Team interests
        api_key: API key
        base_url: API base URL

    Returns:
        Dict with revised_score, key_insight, why_revised
    """

    if model is None:
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    if base_url:
        client_kwargs['base_url'] = base_url

    client = OpenAI(**client_kwargs)

    prompt = create_reevaluation_prompt(paper, full_text, team_profile)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    content = response.choices[0].message.content
    result = parse_json_response(content)

    if result:
        return result
    else:
        return {"revised_score": paper.get('relevance_score', 7),
                "key_insight": "Failed to parse response",
                "why_revised": "Error in re-evaluation"}


def get_top_papers(
    merged_results: list[dict],
    min_score: int = 7,
    top_n: int = 10
) -> list[dict]:
    """
    Filter to get the most relevant papers.

    Args:
        merged_results: Output from merge_results_with_papers()
        min_score: Minimum relevance score to include (default: 7)
        top_n: Maximum number of papers to return (default: 10)

    Returns:
        Top N papers meeting the minimum score threshold

    Example:
        >>> top = get_top_papers(merged, min_score=7, top_n=5)
        >>> all(p['relevance_score'] >= 7 for p in top)
        True
    """

    # Filter to papers meeting the minimum score
    relevant = [p for p in merged_results if p.get('relevance_score', 0) >= min_score]

    # Return top N (already sorted by merge_results_with_papers)
    return relevant[:top_n]
