# Building AI Agents Workshop

A hands-on workshop for building AI agents, starting from a simple batch evaluation system and evolving it into a self-improving application with user feedback loops.

## What You'll Learn

1. **Part 1: Baseline** - Batch inference for paper evaluation
   - Fetch papers from arXiv API
   - Evaluate papers using the Batch API (via Doubleword)
   - Rank and filter results based on team relevance

2. **Part 2: Re-evaluating Borderline Papers**
   - Download full PDFs for borderline papers (score = 7)
   - Deep analysis with full paper text
   - Get revised scores based on complete content

3. **Part 3: Gathering User Feedback**
   - Interactive UI for rating recommendations
   - Collect user signals on recommendation quality
   - Track patterns in what the LLM gets right/wrong

4. **Part 4: Closing the Loop** - Self-improving agents
   - Detect systematic errors in evaluations
   - Update prompts based on user feedback
   - Measure improvement over time

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
# Edit .env with your Doubleword API key
```

### 2. Get a Doubleword API Key

1. Sign up at https://app.doubleword.ai
2. Create an API key and save it
3. Add your API key to the `.env` file

### 3. Run the Workshop

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
│   ├── workshop.ipynb        # Interactive workshop notebook
│   ├── get_papers.py         # Paper fetching helper
│   └── batch_input.jsonl     # Generated batch request file
├── src/
│   ├── get_papers.py         # Fetch papers from arXiv
│   ├── evaluate_papers.py    # Batch evaluation and re-evaluation
│   └── tools.py              # Tool definitions (borderline filtering, PDF extraction)
├── papers/                   # Downloaded PDFs for deep analysis
├── feedback/                 # Collected user feedback
└── sample_data/
    └── papers.json           # Sample papers for offline demos
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your Doubleword API key |
| `OPENAI_BASE_URL` | Doubleword API URL (default: `https://api.doubleword.ai/v1`) |
| `MODEL_NAME` | Model to use (default: `=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`) |

## Requirements

- Python 3.9+
- Doubleword API key (sign up at https://app.doubleword.ai)

## Workshop Flow

### Part 1: Understanding the Baseline

The baseline system evaluates papers using batch inference:

1. **Fetch** - Get papers from arXiv matching keywords (LLM agents, prompt engineering, etc.)
2. **Evaluate** - Use the Batch API to score each paper's relevance (0-10)
3. **Rank** - Sort and filter papers by relevance score

You can use either:
- **Option A**: Create batch files manually with `create_batch_file()`
- **Option B**: Use [autobatcher](https://docs.doubleword.ai/batches/autobatcher) for simpler batch processing

### Part 2: Deep Analysis of Borderline Papers

Papers with score 7 are borderline - they might be worth reading. We:
- Download the full PDF
- Extract text content
- Re-evaluate with the complete paper (not just the abstract)
- Get revised scores to make better recommendations

### Part 3: Gathering User Feedback

The LLM's scores cluster around 7-8 (the "politeness problem"). User feedback helps differentiate:
- Interactive widget UI for rating papers
- Collect signals: Not Useful, Okay, Useful, Must Read
- Compare LLM scores vs. actual user value

### Part 4: Self-Improving Agents

Close the feedback loop:
```
LLM Evaluates → User Rates → Find Patterns → Update Prompt → Repeat
```

The agent learns from feedback to improve its own instructions over time.

## Key Concepts

- **Batch API**: 50% cheaper than real-time calls, perfect for non-interactive workloads
- **Team Profile**: Customizable interests/avoid lists that guide evaluation
- **Structured Output**: JSON responses with scores, insights, and reasoning
- **Feedback Loops**: User signals improve future recommendations

## Tips for the Workshop

- **Use sample data first**: The `sample_data/papers.json` file lets you test without API calls
- **Experiment with the team profile**: Try changing interests to see how scores change
- **Watch the score distribution**: Notice how LLM scores cluster (the "politeness problem")
- **Collect real feedback**: The more feedback, the better the improvement signal

## Troubleshooting

**"No papers found"**
- arXiv might not have papers from the last 7 days matching your keywords
- Use the sample data or increase `days_back` parameter

**"API key not found"**
- Make sure you've created `.env` from `.env.example`
- Ensure `OPENAI_API_KEY` is set to your Doubleword key

**Batch taking too long**
- Batch API typically completes in 1-5 minutes
- Use `completion_window="1h"` for faster processing
- For iteration, reduce the number of papers

**PDF extraction fails**
- Some arXiv PDFs may have unusual formatting
- The workshop handles failures gracefully and continues with available papers
