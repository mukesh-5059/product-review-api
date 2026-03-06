# Product Review Insights API

This project implements Task 5 from the AI and Programming Hackathon Challenge. It provides a Retrieval-Augmented Generation (RAG) pipeline to extract aspects, sentiment, and supporting evidence from a dataset of Amazon product reviews.

## Implementation Details

The implementation follows the mandatory components specified in the challenge objectives:

1. Text Preprocessing: Reviews are cleaned of HTML tags and normalized. Sentence segmentation is performed using the spaCy en_core_web_sm model to ensure semantic boundaries are respected for the embedding process.
2. Aspect Extraction: A hybrid approach is used. First, K-Means clustering (scikit-learn) groups similar review sentences. Second, an LLM (GPT-4o-mini) analyzes these clusters to identify the top 5-10 distinct aspects.
3. Aspect Sentiment Analysis: Sentiment is calculated based on the metadata ratings (1-5 stars) associated with the retrieved sentences for each aspect. A score between 0.0 and 1.0 is generated based on the ratio of positive to negative reviews.
4. Evidence Selection: Supporting evidence is retrieved using vector similarity search (ChromaDB) with a configurable relevance threshold. All evidence is verbatim from the dataset to prevent hallucination.
5. API Validation: The system uses FastAPI for request handling, input validation, and error management. 

## Repository Structure

product-review-api/
├── main.py                     # Client Gateway API (Proxies requests to RAG Engine)
├── RAG/
│   ├── main.py                 # Core RAG Server (Handles analysis logic)
│   ├── insight_engine.py       # Core Logic (LLM, Sentiment, and Evidence Selection)
│   ├── cluster_aspect_extractor.py # K-Means implementation for theme discovery
│   ├── vector_store.py         # ChromaDB management and sub-batching logic
│   ├── data_manager.py         # spaCy preprocessing and data loading
│   └── index_data.py           # Script to populate the vector database
├── front_end/
│   └── dashboard.py            # Streamlit-based user interface
├── data/
│   └── Clean_reviews.csv       # Preprocessed dataset
├── logs/                       # Server logs (gateway.log and rag_engine.log)
├── requirements.txt            # Project dependencies
└── .env                        # Configuration for OpenRouter API key

## Installation and Setup

1. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

3. Configure environment variables:
   Create a .env file in the root directory and add your key:
   OPENROUTER_API_KEY=your_api_key_here

4. Index the dataset:
   python RAG/index_data.py

## Running the Application

The system requires two backend servers and one frontend application.

1. Start the RAG Engine (Core Server):
   python -m RAG.main
   (Runs on http://localhost:8000)

2. Start the Gateway API (Client Server):
   python main.py
   (Runs on http://localhost:8001)

3. Start the Dashboard (Frontend):
   streamlit run front_end/dashboard.py

## API Documentation

### Get Product Insights
Endpoint: GET /items/{item_id}

Sample Request:
curl http://localhost:8001/items/B003VXFK44

Sample Output:
{
  "product_id": "B003VXFK44",
  "status": "SUCCESS",
  "summary": "This product features a smooth flavor profile suitable for various tastes. However, there are noted issues with packaging labels and return policies.",
  "top_aspects": [
    {
      "aspect": "Flavor Smoothness",
      "category": "Pro",
      "sentiment_score": 0.82,
      "pros_evidence": ["The flavor is smooth and delightful."],
      "cons_evidence": ["A bit too weak for my preference."]
    }
  ]
}

## Error Handling and Logging

- Insufficient Data: If a product has fewer than 5 reviews, the API returns a status of INSUFFICIENT_DATA and refuses to generate a summary to avoid inaccurate results.
- Logging: All internal processes, API requests, and errors are logged to the logs/ directory using Python's standard logging module. Both file and console handlers are implemented.
- Robustness: The Gateway API includes timeouts and error handling for cases where the RAG Engine is unreachable or returns an error.
