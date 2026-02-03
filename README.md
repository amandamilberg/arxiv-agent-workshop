# Building AI Agents Workshop

A hands-on workshop for building AI agents, starting from a simple batch evaluation system and evolving it into a fully agentic application.

## What You'll Learn

1. **Part 1: Baseline** - A simple batch evaluation pipeline
   - Fetch papers from ArXiv API
   - Evaluate papers using an OpenAI-compatible Batch API
   - Return results for analysis in Jupyter notebooks

2. **Part 2: Adding Tools** *(coming soon)*
   - Define tools for the agent to use
   - Implement tool calling

3. **Part 3: Agentic Loop** *(coming soon)*
   - Build an autonomous agent loop
   - Handle multi-step reasoning

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API key
```

### 2. Run in Jupyter

```bash
jupyter notebook notebooks/workshop.ipynb
```

Or run the scripts directly:

```bash
# Test paper fetching
python src/get_papers.py

# Test evaluation (requires API key)
python src/evaluate_papers.py
```

## Project Structure

```
arxiv-agent-workshop/
├── README.md                 # You're here!
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables
├── notebooks/
│   └── workshop.ipynb        # Interactive workshop notebook
├── src/
│   ├── get_papers.py         # Fetch papers from ArXiv
│   ├── evaluate_papers.py    # Batch evaluation with OpenAI-compatible API
│   └── tools.py              # Tool definitions (Part 2)
└── sample_data/
    └── papers.json           # 20 sample papers for offline demos
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your API key for the OpenAI-compatible provider |
| `OPENAI_BASE_URL` | (Optional) Base URL if using a different provider |
| `MODEL_NAME` | Model to use (default: `gpt-4o-mini`) |

## Requirements

- Python 3.9+
- API key for an OpenAI-compatible service

## Workshop Flow

### Part 1: Understanding the Baseline

The baseline system is intentionally simple:

1. **Fetch** - Get papers from ArXiv matching keywords
2. **Evaluate** - Use the Batch API to score each paper's relevance
3. **Analyze** - Review results in a notebook

This is NOT agentic yet - it's a fixed pipeline with no decision-making.

### Part 2: Making it Agentic

We'll transform this into an agent by:
- Giving the model tools to call (search, fetch, summarize)
- Letting the model decide what to do next
- Adding memory and context management

### Part 3: Advanced Patterns

- Multi-agent collaboration
- Human-in-the-loop workflows
- Error handling and recovery

## Tips for the Workshop

- **Use sample data first**: The `sample_data/papers.json` file lets you test without API calls
- **Read the comments**: The code is heavily commented for learning
- **Experiment**: Try changing the team profile and see how results change

## Troubleshooting

**"No papers found"**
- ArXiv might not have papers from the last 7 days matching your keywords
- Use the sample data: `load_sample_papers()` function

**"API key not found"**
- Make sure you've created `.env` from `.env.example`
- Ensure `OPENAI_API_KEY` is set correctly

**Batch taking too long**
- Batch API can take a few minutes
- For faster iteration, use sample data or reduce the number of papers
