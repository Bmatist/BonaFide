# BonaFide

A web-based tool for analyzing news articles for political bias, framing, and omissions using a multi-agent orchestrated pipeline.

## Features
- **Genre-Aware Scoring**: Calibrated analysis for News, Op-Eds, and Interviews.
- **Multi-Agent Architecture**: Orchestrated analysis pipeline using specialized agents for content extraction, context retrieval, and bias comparison.
- **Ephemeral RAG (Retrieval-Augmented Generation)**: Real-time web search integration via Tavily to ground analysis in verified external facts and counter-perspectives.
- **Omission Analysis**: Detects critical context gaps with relevance and intentionality tagging.
- **Editorial DNA**: Maps articles to specific media archetypes (e.g., State-Aligned, Western-Liberal).
- **Reader Risk**: Highlights potential interpretational consequences for the reader.

## Setup

### Prerequisites
- Python 3.11+
- [Docker](https://www.docker.com/) (Optional)

### Environment Variables
Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Local Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn src.app:app --reload
   ```

## Docker Usage

### Build and Run with Docker

```bash
docker build -t bias-detection-app .
docker run -p 8000:8000 bias-detection-app
```
## License
This project is licensed under the MIT License — see the LICENSE file for details.