import os
import json
import numpy as np
import logging
from openai import OpenAI
from dotenv import load_dotenv
from .cluster_aspect_extractor import ClusterAspectExtractor
from .vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (API Key)
load_dotenv()

class InsightEngine:
    def __init__(self):
        try:
            self.cluster_extractor = ClusterAspectExtractor()
            self.vs = VectorStore()
            
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("OPENROUTER_API_KEY not found in environment.")
                raise ValueError("OPENROUTER_API_KEY not found in .env file")
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.model_id = "openai/gpt-4o-mini"
            logger.info("InsightEngine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize InsightEngine: {e}")
            raise

    def get_full_insights(self, product_id: str):
        """
        Full Task 5 Pipeline with 'Insufficient Data' handling and logging.
        """
        try:
            logger.info(f"Processing insights for Product ID: {product_id}")
            
            # Step 0: Initial data check
            results_raw = self.vs.get_all_for_product(product_id)
            sentences = results_raw.get('documents', [])
            
            if len(sentences) < 5:
                logger.warning(f"Insufficient data for {product_id}: only {len(sentences)} sentences.")
                return {
                    "product_id": product_id,
                    "error": "Not enough data to generate reliable insights. Minimum 5 review sentences required.",
                    "confidence": 0.0
                }

            # Step 1: Extract Top Aspects
            aspects = self.get_top_aspects(product_id)
            if isinstance(aspects, dict) and "error" in aspects:
                return aspects

            results = {
                "product_id": product_id,
                "top_aspects": [],
                "summary": "",
                "confidence": 0.85
            }

            all_evidence_for_summary = []

            for aspect in aspects:
                search_results = self.vs.search(aspect, product_id=product_id, top_k=50)
                
                docs = search_results.get('documents', [[]])[0]
                metas = search_results.get('metadatas', [[]])[0]
                distances = search_results.get('distances', [[]])[0]
                
                pros_evidence = []
                cons_evidence = []
                neutral_evidence = []
                
                RELEVANCE_THRESHOLD = 0.80 
                
                for i in range(len(docs)):
                    if distances[i] > RELEVANCE_THRESHOLD:
                        continue
                    
                    text = docs[i].strip()
                    rating = metas[i].get('rating', 0)
                    
                    if any(e == text for e in pros_evidence + cons_evidence + neutral_evidence):
                        continue

                    if rating >= 4:
                        pros_evidence.append(text)
                    elif rating <= 2:
                        cons_evidence.append(text)
                    else:
                        neutral_evidence.append(text)

                total_pro_con = len(pros_evidence) + len(cons_evidence)
                
                if total_pro_con < 3:
                    category = "Insufficient Data"
                    sentiment_score = 0.0
                else:
                    sentiment_score = len(pros_evidence) / total_pro_con
                    if sentiment_score >= 0.7: category = "Pro"
                    elif sentiment_score <= 0.3: category = "Con"
                    else: category = "Mixed"

                aspect_data = {
                    "aspect": aspect,
                    "category": category,
                    "sentiment_score": round(sentiment_score, 2),
                    "pros_evidence": pros_evidence[:5],
                    "cons_evidence": cons_evidence[:5],
                    "reference_evidence": neutral_evidence[:3] if category == "Insufficient Data" else []
                }
                
                results["top_aspects"].append(aspect_data)
                all_evidence_for_summary.extend(pros_evidence[:2] + cons_evidence[:2])

            # Step 5: Final Summary
            if all_evidence_for_summary:
                summary_text = "\n".join(all_evidence_for_summary[:15])
                prompt = f"Based on these specific review points, provide a concise 2-sentence summary of this product's strengths and weaknesses.\n\nPoints:\n{summary_text}"
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    results["summary"] = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"LLM Summary generation failed: {e}")
                    results["summary"] = "Summary unavailable."

            return results
        except Exception as e:
            logger.error(f"Unexpected error in get_full_insights for {product_id}: {e}")
            return {"error": f"Internal process error: {str(e)}"}

    def get_top_aspects(self, product_id: str):
        """Identifies top aspects with error handling."""
        try:
            representative_sentences = self.cluster_extractor.get_representative_sentences(product_id)
            if not representative_sentences:
                return {"error": "Not enough data for this product."}

            num_reps = len(representative_sentences)
            if num_reps <= 8: specific_count = "2 to 3"
            elif num_reps <= 15: specific_count = "3 to 4"
            else: specific_count = "5 to 6"

            sentences_text = "\n".join([f"- {s}" for s in representative_sentences])
            prompt = f"""
            Identify aspects for this product.
            Include Mandatory: 'Price/Value', 'Delivery/Packaging', and 'Customer Service'.
            Identify {specific_count} additional distinct product-specific aspects.
            NO REDUNDANCY. NO GENERIC SENTIMENT.
            Return ONLY a JSON list of strings.
            Reviews:
            {sentences_text}
            """

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "system", "content": "Return JSON list only."}, {"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            if isinstance(data, list): return data
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list): return val
                return list(data.keys())
            return data
        except Exception as e:
            logger.error(f"Aspect extraction failed for {product_id}: {e}")
            return {"error": "Failed to extract aspects."}

if __name__ == "__main__":
    import sys
    engine = InsightEngine()
    test_ids = ["B002BCD2OG", "B004NE2E9O"]
    
    for test_id in test_ids:
        print("\n" + "="*80)
        insights = engine.get_full_insights(test_id)
        print(json.dumps(insights, indent=2))
