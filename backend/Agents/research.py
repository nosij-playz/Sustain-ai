import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
from bs4 import BeautifulSoup

from backend.data_fetch.research import AIAnalyzer, WebSearcher


class ResearchAnalyzer:
    def __init__(self, max_results=10, output_folder="display", model_name: Optional[str] = None, include_images: bool = True, enable_fallback_model: bool = True):
        self.web_searcher = WebSearcher(max_results=max_results)
        if model_name:
            self.ai_analyzer = AIAnalyzer(model_name=model_name, enable_fallback=enable_fallback_model)
        else:
            self.ai_analyzer = AIAnalyzer()
        self.output_folder = output_folder
        self.include_images = include_images
        self._image_cache = {}

        os.makedirs(self.output_folder, exist_ok=True)

    def _get_og_image(self, url: str) -> Optional[str]:
        if not url:
            return None

        if url in self._image_cache:
            return self._image_cache.get(url)

        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "WasteDispoBot/1.0"})
            if resp.status_code >= 400:
                self._image_cache[url] = None
                return None

            soup = BeautifulSoup(resp.text, "lxml")
            tag = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
            image = None
            if tag:
                image = tag.get("content")
            if image and isinstance(image, str):
                image = image.strip()
            if not image:
                image = None

            self._image_cache[url] = image
            return image
        except Exception:
            self._image_cache[url] = None
            return None

    def _enrich_with_images(self, items, limit=3):
        if not self.include_images:
            return items

        if not isinstance(items, list) or not items:
            return items

        to_fetch = []
        for i, item in enumerate(items[:limit]):
            link = (item or {}).get("link")
            if link and isinstance(link, str):
                to_fetch.append((i, link))

        if not to_fetch:
            return items

        def worker(idx, link):
            return idx, self._get_og_image(link)

        with ThreadPoolExecutor(max_workers=min(6, len(to_fetch))) as ex:
            futures = [ex.submit(worker, idx, link) for idx, link in to_fetch]
            for fut in futures:
                idx, img = fut.result()
                if img:
                    items[idx]["image"] = img

        return items

    def run_research(self, query):
        """
        Perform web search + AI analysis.

        Args:
            query (str): Research topic/query

        Returns:
            dict: Structured result
        """
        if not query:
            return {
                "success": False,
                "error": "No query provided."
            }

        try:
            # Step 1: Web Search
            raw_data = self.web_searcher.search(query)

            if not raw_data:
                return {
                    "success": False,
                    "error": "No search results found."
                }

            # Step 2: AI Analysis
            final_json_list = self.ai_analyzer.study_and_simplify_batch(raw_data)

            # Optional: add real preview images from the web (og:image)
            if isinstance(final_json_list, list):
                final_json_list = self._enrich_with_images(final_json_list)

            return {
                "success": True,
                "query": query,
                "results_count": len(final_json_list),
                "report": final_json_list
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def save_report(self, query, filename="research_report.json"):
        """
        Run research and save report to JSON file.
        """
        result = self.run_research(query)

        if result["success"]:
            filepath = os.path.join(self.output_folder, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

            result["saved_to"] = filepath

        return result

    def get_json(self, query):
        """
        Return JSON string output.
        """
        return json.dumps(self.run_research(query), indent=4)