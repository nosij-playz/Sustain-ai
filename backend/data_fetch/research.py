import json
import ollama
from ddgs import DDGS
from typing import List, Dict

class WebSearcher:
    def __init__(self, max_results: int = 8):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict]:
        results = []
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query,
                    max_results=self.max_results
                )
                for item in search_results:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "link": item.get("href", "") # Changed 'source' to 'link'
                    })
        except Exception as e:
            print(f"Search Error: {e}")
        return results

class AIAnalyzer:
    def __init__(self, model_name="gemma4:31b-cloud", fallback_model: str = "gemma4:31b-cloud", enable_fallback: bool = True):
        locked_model = "gemma4:31b-cloud"
        self.model_name = locked_model
        self.fallback_model = locked_model
        self.enable_fallback = enable_fallback
        self.max_tokens = 16000  # Increased for better quality responses

    def study_and_simplify_batch(self, raw_results: List[Dict]):
        """
        Sends ALL results to the AI in one go (Batch Processing) 
        to drastically increase speed and efficiency.
        """
        if not raw_results:
            return [{"error": "No data to analyze"}]

        # 1. Format all results into a single block of text for the AI
        formatted_data = ""
        for i, item in enumerate(raw_results):
            formatted_data += f"--- Source {i+1} ---\nTitle: {item['title']}\nContent: {item['snippet']}\nLink: {item['link']}\n\n"

        # 2. Create a comprehensive prompt for a JSON List output with quality grading
        prompt = (
            f"You are an expert environmental scientist and a friendly teacher. "
            f"I will give you a list of search results. Your task is to study all of them and "
            f"simplify the information so a 15-year-old can understand it. Be friendly and encouraging. "
            f"Grade each explanation for quality (A-F scale based on clarity and actionability).\n\n"
            f"DATA TO STUDY:\n{formatted_data}\n\n"
            f"IMPORTANT: You must return ONLY a JSON list of objects. "
            f"Each object must have exactly these keys: 'title', 'explain', 'source', 'link', 'quality_grade'. "
            f"- 'title': The original title.\n"
            f"- 'explain': Your friendly simplified explanation (concise, 2-3 sentences max).\n"
            f"- 'source': The name of the website/platform (extracted from the link).\n"
            f"- 'link': The original URL.\n"
            f"- 'quality_grade': Grade this explanation A-F (A=excellent clarity, F=poor)."
        )

        print(f"\n🧠 Analyzer is processing {len(raw_results)} sources in one batch (max_tokens: 16k)... Please wait.")

        def run(model_name: str):
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            ai_content = response['message']['content']
            cleaned_json = ai_content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_json)

        try:
            return run(self.model_name)
        except Exception as e:
            print(f"Batch processing error: {e}")
            # If the requested model isn't available, optionally retry with fallback.
            if self.enable_fallback and self.fallback_model and self.model_name != self.fallback_model:
                try:
                    print(f"Retrying batch analysis with fallback model: {self.fallback_model}")
                    return run(self.fallback_model)
                except Exception as e2:
                    print(f"Fallback batch processing error: {e2}")

            # Final fallback: return raw data in the requested format
            return [
                {"title": i['title'], "explain": i['snippet'], "source": "Web", "link": i['link'], "quality_grade": "B"} 
                for i in raw_results
            ]


