import json
import ollama
import re
import os
import shutil
import time
import sqlite3
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Importing your agents
from backend.Agents.env_live import EnvironmentalDataFetcher
from backend.Agents.summarizer import TechnicalTextSummarizer
from backend.image.explain import Vision 
from backend.Agents.Plotter import AIPlotterApp
from backend.Agents.research import ResearchAnalyzer
from backend.output.Dashboard import DashboardModule

def _load_env_basic(dotenv_path: str = ".env"):
    try:
        with open(dotenv_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        pass

_load_env_basic()

class SessionManager:
    """Handles temporary data storage and session cleanup."""
    def __init__(self, filename=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        storage_dir = os.path.join(root, "Storage")
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.filename = filename or os.path.join(storage_dir, "session_state.json")

    def save(self, data):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def cleanup(self, created_files: Optional[List[str]] = None, purge_session: bool = True):
        print("\n🧹 Cleaning up session data...")
        created_files = created_files or []

        for path in created_files:
            try:
                if path and os.path.exists(path) and os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass

        try:
            if os.path.exists("interface/dashboard.html"):
                os.remove("interface/dashboard.html")
        except Exception:
            pass

        try:
            if os.path.isdir("display"):
                for name in os.listdir("display"):
                    if name.lower().endswith(".png"):
                        try:
                            os.remove(os.path.join("display", name))
                        except Exception:
                            pass
        except Exception:
            pass

        if purge_session:
            try:
                if os.path.exists(self.filename):
                    os.remove(self.filename)
            except Exception:
                pass

        print("✅ Session closed.")


class ChatMemoryStore:
    """SQLite-backed chat memory for preserving recent conversation turns during a run."""
    def __init__(self, db_path: Optional[str] = None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        storage_dir = os.path.join(root, "Storage")
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.db_path = db_path or os.path.join(storage_dir, "chat_memory.sqlite")
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()
        except sqlite3.Error:
            pass
        return connection

    def _initialize(self):
        # Now handled within _connect to ensure table existence on every connection
        pass

    def append_message(self, role: str, content):
        if content is None:
            return

        text = str(content).strip()
        if not text:
            return

        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO chat_messages (role, content) VALUES (?, ?)",
                (role, text),
            )
            connection.commit()
        finally:
            connection.close()

    def append_turn(self, user_text, assistant_text):
        self.append_message("user", user_text)
        self.append_message("assistant", assistant_text)

    def recent_messages(self, limit: int = 12) -> List[Dict[str, str]]:
        if limit <= 0 or not os.path.exists(self.db_path):
            return []

        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT role, content FROM chat_messages ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            connection.close()

        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def cleanup(self):
        for suffix in ("", "-wal", "-shm", "-journal"):
            path = self.db_path + suffix
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


class WasteDispoMaster:
    ENV_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

    def __init__(self, model_name="gemma4:31b-cloud", lite_model=None, default_location="Chittarikkal, Kerala, India"):
        # Hard-lock to a single model across the entire system.
        locked_model = "gemma4:31b-cloud"
        self.model_name = locked_model
        self.lite_model = self.model_name  # kept only for backward compatibility; never used as a different model
        self.max_tokens = 16000  # Increased token limit for better responses
        self.session = SessionManager()
        self.chat_memory = ChatMemoryStore()
        self.system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
        self.master_name = os.getenv("SUSTAINAI_MASTER_NAME", "Lily")

        # Initialize context from saved session or defaults
        saved_data = self.session.load()
        self.context = {
            "location": saved_data.get("location", default_location),
            "knowledge_base": saved_data.get("knowledge_base", {}),
            "is_processing": False,
            "created_files": saved_data.get("created_files", []),
            "last_suggested_actions": saved_data.get("last_suggested_actions", []),
        }
        
        # Track the last cached location for invalidation
        self._last_cached_location = None

        self.agents = {
            "env_agent": {"module": EnvAgentWrapper(), "description": "Real-time weather, soil, and env data.", "type": "DATA"},
            "vision_agent": {"module": VisionAgentWrapper(), "description": "Analyzes images of waste.", "type": "TEXT"},
            "plotter_agent": {"module": PlotterAgentWrapper(), "description": "Creates high-end data visualizations.", "type": "FILE"},
            "search_agent": {"module": ResearchAgentWrapper(), "description": "Deep internet research on waste/chemicals.", "type": "REPORT"},
            "summarizer_agent": {"module": SummarizerAgentWrapper(), "description": "Condenses long agent logs into concise executive summaries.", "type": "TEXT"},
            "classifier_agent": {"module": ClassifierAgentWrapper(), "description": "Classifies waste using local YOLO model.", "type": "IMAGE"},
            "dashboard_agent": {"module": DashboardAgentWrapper(), "description": "Generates a comprehensive Intelligence Command Center.", "type": "FILE"}
        }

    def _get_system_prompt(self):
        agent_desc = "\n".join([f"- {name}: {info['description']}" for name, info in self.agents.items()])
        return (
            f"You are '{self.master_name}', the lead intelligence for {self.system_name}. "
            f"Your goal is to provide a seamless, luxury-grade experience. \n\n"
            f"--- 🛑 STRICT NICHE GUARDRAIL ---\n"
            f"REFUSE all non-environmental queries. Only discuss waste, pollution, soil, and sustainability.\n\n"
            f"--- ⚙️ OPERATIONAL MODE ---\n"
            f"Auto-trigger agents the moment the user mentions environment, waste, pollution, soil, air quality, sustainability, or related requests.\n"
            f"Never ask for permission or say 'Would you like me to...?' or 'Shall I...?'. Execute and report results.\n"
            f"When the user asks for full information, reports, or analysis, run all core agents in one pass.\n"
            f"When the user refers to earlier images, data, or location context, use stored session memory and answer from it instead of asking for the upload again.\n"
            f"IMPORTANT: Reuse cached environmental data for 1 hour when available; do not re-fetch unnecessarily.\n"
            f"IMPORTANT: Never mention file names, image paths, or internal filenames in your responses. If the user refers to an image, describe its content without revealing the file name.\n\n"
            f"--- CONTEXT ---\n"
            f"Current User Location: {self.context['location']}\n\n"
            f"AVAILABLE TOOLS:\n{agent_desc}\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return a JSON list of actions: [ {{ 'intent': 'agent_name', 'parameters': {{ 'key': 'value' }} }} ]\n"
            f"If just chatting, respond as a world-class expert."
        )

    def _build_memory_context(self) -> str:
        kb = self.context.get("knowledge_base", {}) if isinstance(self.context, dict) else {}
        parts = []

        location = self.context.get("location")
        if location:
            parts.append(f"Location memory: {location}")

        cache = kb.get("_cache", {}) if isinstance(kb, dict) else {}
        env_cache = cache.get("env", {}) if isinstance(cache, dict) else {}
        env_result = env_cache.get("result", {}) if isinstance(env_cache, dict) else {}
        if isinstance(env_result, dict) and env_result:
            env_bits = []
            for key in ["temperature", "humidity", "wind_speed", "windspeed_openmeteo", "rain_1h", "precip_mm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"]:
                value = env_result.get(key)
                if value is not None:
                    env_bits.append(f"{key}={value}")
            if env_bits:
                parts.append("Latest environmental cache: " + "; ".join(env_bits[:8]))

        recent_visuals = []
        for key in ["image_analyses", "classifier_results"]:
            items = kb.get(key) if isinstance(kb, dict) else None
            if not isinstance(items, list) or not items:
                continue
            for item in items[-2:]:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or os.path.basename(item.get("file_path", "")) or key
                text = item.get("explanation") or item.get("result") or item.get("text")
                if isinstance(text, dict):
                    text = json.dumps(text, ensure_ascii=False)
                if text:
                    cleaned = str(text).replace("\n", " ").strip()
                    recent_visuals.append(f"{title}: {cleaned[:1200]}")

        if recent_visuals:
            parts.append("Recent image memory:\n- " + "\n- ".join(recent_visuals))

        suggested = self.context.get("last_suggested_actions") or []
        if suggested:
            parts.append(f"Last suggested actions: {json.dumps(suggested, ensure_ascii=False)}")

        if not parts:
            return ""

        return "SESSION MEMORY:\n" + "\n\n".join(parts)

    def _build_chat_messages(self, user_text: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        memory_context = self._build_memory_context()
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        messages.extend(self.chat_memory.recent_messages(limit=12))
        messages.append({"role": "user", "content": user_text})
        return messages

    def _stringify_response(self, response) -> str:
        if isinstance(response, str):
            return response
        try:
            return json.dumps(response, indent=2, ensure_ascii=False)
        except Exception:
            return str(response)

    def _remove_file_paths(self, text: str) -> str:
        """Remove any file paths (e.g., ./backend/display/speech_xxx.jpg) from the text."""
        if not text:
            return text
        # Remove any string that looks like a path ending with an image extension
        pattern = r'[^\s]*\.(?:png|jpg|jpeg|webp)[^\s]*'
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
        # Also remove any leftover slashes or path-like segments
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _extract_image_path(self, text: str) -> Optional[str]:
        """Extract an image file path from the text. More robust than before."""
        if not text:
            return None
        # Look for any non‑space sequence ending with an image extension
        m = re.search(r"([^\s]+\.(?:png|jpg|jpeg|webp))", text, flags=re.IGNORECASE)
        if not m:
            return None
        candidate = m.group(1).strip().strip('"').strip("'")
        # If it's an absolute path or relative that exists, return it
        if os.path.exists(candidate):
            return candidate
        # Otherwise, check if it's a filename that exists in the display folder
        filename = os.path.basename(candidate)
        display_path = os.path.join("./backend/display", filename)
        if os.path.exists(display_path):
            return display_path
        # Also check the root or image folder for demo images
        demo_paths = [filename, os.path.join("image", filename)]
        for p in demo_paths:
            if os.path.exists(p):
                return p
        return None

    def _remember_turn(self, user_text: str, assistant_response) -> str:
        response_text = self._stringify_response(assistant_response)
        # Remove any file paths from the response before storing
        response_text = self._remove_file_paths(response_text)
        self.chat_memory.append_turn(user_text, response_text)
        return response_text

    def cleanup(self):
        self.session.cleanup(created_files=self.context.get("created_files"), purge_session=True)
        self.chat_memory.cleanup()

    def _now(self) -> int:
        return int(time.time())

    def _get_cache(self) -> Dict:
        kb = self.context.setdefault("knowledge_base", {})
        return kb.setdefault("_cache", {})

    def _invalidate_env_cache_if_location_changed(self, new_location: str):
        """Invalidate environment cache if the location has changed."""
        if not new_location:
            return
        
        cache = self._get_cache()
        env_cache = cache.get("env")
        
        if env_cache:
            cached_place = (env_cache.get("place") or "").strip().lower()
            new_place = (new_location or "").strip().lower()
            
            if cached_place and cached_place != new_place:
                print(f"🗑️ Location changed from '{cached_place}' to '{new_place}'. Invalidating env cache.")
                cache.pop("env", None)
                self._last_cached_location = None

    def _get_cached_env(self, place: str):
        # Invalidate cache if location changed
        self._invalidate_env_cache_if_location_changed(place)
        
        cache = self._get_cache()
        env_cache = cache.get("env")
        if not env_cache:
            return None

        if (env_cache.get("place") or "").strip().lower() != (place or "").strip().lower():
            return None

        fetched_at = env_cache.get("fetched_at")
        if not isinstance(fetched_at, int):
            return None

        if self._now() - fetched_at > self.ENV_CACHE_TTL_SECONDS:
            return None

        return env_cache.get("result")

    def _set_cached_env(self, place: str, result: Dict):
        cache = self._get_cache()
        cache["env"] = {
            "place": place,
            "fetched_at": self._now(),
            "ttl_seconds": self.ENV_CACHE_TTL_SECONDS,
            "result": result,
        }
        self._last_cached_location = place

    def _run_env_cached(self, place: str) -> Dict:
        # Invalidate cache if location changed (defensive call)
        self._invalidate_env_cache_if_location_changed(place)
        
        cached = self._get_cached_env(place)
        if cached:
            return {"success": True, "cached": True, **cached}

        res = self.agents["env_agent"]["module"].run({"place": place})
        if isinstance(res, dict) and res.get("success"):
            self._set_cached_env(place, res)
        return res

    def _should_explain_plots(self, user_text: str) -> bool:
        if not user_text:
            return False
        text = user_text.lower()
        return any(k in text for k in ["explain", "insight", "analyze", "analyse", "interpret", "what does this plot", "what does this chart"]) 

    def _is_full_report_request(self, user_text: str) -> bool:
        if not user_text:
            return False
        text = user_text.lower()
        return any(k in text for k in [
            "full report",
            "full data report",
            "detailed report",
            "detailed analysis",
            "full analysis",
            "full information",
            "all data",
            "everything",
            "complete report",
            "collect all the data",
            "complete analysis",
        ]) 

    def _is_affirmative_text(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        return any(k == t or f" {k} " in f" {t} " for k in [
            "yes",
            "yeah",
            "yep",
            "yup",
            "ok",
            "okay",
            "do it",
            "please do",
            "go ahead",
            "sure",
            "affirmative",
            "proceed",
        ])

    def _build_full_bundle(self, user_text: str) -> List[Dict]:
        place = self.context.get("location") or "Unknown"
        query = f"Waste management, pollution, and sustainability updates for {place}. User request: {user_text}".strip()
        return [
            {"intent": "env_agent", "parameters": {"place": place}},
            {"intent": "search_agent", "parameters": {"query": query}},
            {"intent": "plotter_agent", "parameters": {}},
            {"intent": "dashboard_agent", "parameters": {}},
        ]

    def _auto_actions_from_text(self, user_text: str) -> List[Dict]:
        if not user_text:
            return []
        text = user_text.lower()
        if self._is_full_report_request(text):
            return self._build_full_bundle(user_text)

        env_keywords = [
            "environment",
            "environmental",
            "pollution",
            "waste",
            "soil",
            "air quality",
            "weather",
            "climate",
            "sustainability",
            "emissions",
            "recycling",
        ]
        if any(k in text for k in env_keywords):
            return self._build_full_bundle(user_text)
        return []

    def _capture_suggested_actions(self, ai_text: str) -> List[Dict]:
        if not ai_text:
            return []
        t = ai_text.lower()
        if any(k in t for k in ["would you like", "shall i", "do you want", "should i", "i can"]):
            if any(k in t for k in ["environment", "pollution", "waste", "soil", "air quality", "sustainability", "dashboard", "report"]):
                return self._build_full_bundle(ai_text)
        return []

    def _wants_image_analysis(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(k in t for k in ["explain image", "analyze image", "analyse image", "image analysis", "demo image"]) 

    def _wants_classification(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(k in t for k in ["classify", "classification", "identify object", "identify this", "what is this", "detect object", "detect objects"])

    def _classify_image(self, image_path: str) -> Dict:
        return self.agents["classifier_agent"]["module"].run({"image_path": image_path, "conf": 0.3})

    def _analyze_plot_images(self, image_paths: List[str], prompt: str) -> List[Dict[str, str]]:
        if not image_paths:
            return []

        existing_explanations = self.context.setdefault("knowledge_base", {}).setdefault("plot_explanations", [])
        existing_analyses = self.context.setdefault("knowledge_base", {}).setdefault("image_analyses", [])

        existing_paths = {item.get("file_path") for item in existing_explanations if item.get("file_path")}
        existing_paths |= {item.get("file_path") for item in existing_analyses if item.get("file_path")}

        targets = [p for p in image_paths if p and p not in existing_paths]
        if not targets:
            return []

        def run_one(path: str):
            explanation = self.agents["vision_agent"]["module"].run({
                "image_path": path,
                "prompt": prompt,
            })

            try:
                rel = os.path.relpath(path, start=os.getcwd()).replace("\\", "/")
            except Exception:
                rel = str(path).replace("\\", "/")
            dash_src = rel if rel.startswith("../") else f"../{rel}"

            return {
                "file_path": path,
                "explanation": explanation,
                "dash_src": dash_src,
            }

        max_workers = min(4, len(targets))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for item in ex.map(run_one, targets):
                if item and item.get("file_path"):
                    results.append(item)

        for item in results:
            existing_explanations.append({
                "file_path": item["file_path"],
                "explanation": item["explanation"],
            })
            existing_analyses.append({
                "title": f"Plot Analysis: {os.path.basename(item['file_path'])}",
                "file_path": item["file_path"],
                "image": item["dash_src"],
                "explanation": item["explanation"],
            })

        return [
            {"title": os.path.basename(item["file_path"]), "text": item["explanation"]}
            for item in results
        ]

    def _format_plot_explanations(self, explanations: List[Dict[str, str]]) -> str:
        if not explanations:
            return ""
        lines = ["Plot analysis:"]
        for item in explanations:
            title = item.get("title") or "Plot"
            text = item.get("text") or ""
            if text:
                lines.append(f"- {title}: {text}")
        return "\n".join(lines)

    def _analyze_image_list(self, image_list: Optional[List[str]]) -> List[str]:
        if not image_list:
            return []

        analyses = []
        summaries = []
        for image_path in image_list:
            if not image_path or not os.path.exists(image_path):
                continue

            explanation = self.agents["vision_agent"]["module"].run({
                "image_path": image_path,
                "prompt": "Explain this image in detail and relate it to waste, pollution, and sustainability where relevant.",
            })

            # Dashboard is generated under interface/, so use ../ relative src.
            try:
                rel = os.path.relpath(image_path, start=os.getcwd()).replace("\\", "/")
            except Exception:
                rel = str(image_path).replace("\\", "/")
            dash_src = rel if rel.startswith("../") else f"../{rel}"

            analyses.append({
                "title": f"Image Analysis: {os.path.basename(image_path)}",
                "file_path": image_path,
                "image": dash_src,
                "explanation": explanation,
            })
            summaries.append(f"{os.path.basename(image_path)}: {explanation}")

        if analyses:
            self.context.setdefault("knowledge_base", {}).setdefault("image_analyses", []).extend(analyses)
            self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
            self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
            self.session.save(self.context)
        return summaries

    def _build_master_insights(self, kb: Dict) -> List[str]:
        insights: List[str] = []

        env_res = kb.get("env_agent") if isinstance(kb, dict) else None
        if isinstance(env_res, dict) and env_res.get("success"):
            env_data = env_res.get("environmental_data", {}) or {}
            humidity = env_data.get("humidity")
            temp = env_data.get("temperature")
            soil = env_data.get("soil_moisture_0_to_7cm") or env_data.get("soil_moisture_7_to_28cm")
            wind = env_data.get("wind_speed") or env_data.get("windspeed_openmeteo")
            rain = env_data.get("rain_1h") or env_data.get("precip_mm")

            if temp is not None and temp >= 28:
                insights.append(f"Warm conditions ({temp}°C) speed up decomposition; use sealed bins + increase pickup frequency for organics.")
            if humidity is not None and humidity >= 90:
                insights.append(f"Very high humidity ({humidity}%) increases odor/mold risk; keep waste dry and covered.")
            if rain is not None and rain > 0:
                insights.append("Recent precipitation suggests leachate risk; ensure runoff control near waste storage and dumps.")
            if wind is not None and wind <= 1.5:
                insights.append("Low wind can trap odors; improve ventilation around waste holding areas.")
            if soil is not None:
                insights.append(f"Soil moisture reading is {soil}; for composting, avoid anaerobic conditions via mixing/aeration.")

        search_res = kb.get("search_agent") if isinstance(kb, dict) else None
        if isinstance(search_res, dict) and search_res.get("success"):
            insights.append("Research signals below include the latest web sources; prioritize local policy/infrastructure changes for actionability.")

        return insights

    def _cleanup_previous_outputs(self):
        """Deletes only files previously generated by agents (tracked in created_files)."""
        created = self.context.get("created_files") or []
        if not isinstance(created, list) or not created:
            created = []

        kept = []
        for path in created:
            try:
                if path and os.path.exists(path) and os.path.isfile(path):
                    os.remove(path)
                else:
                    kept.append(path)
            except Exception:
                kept.append(path)

        # Reset to what couldn't be deleted
        self.context["created_files"] = kept

        # Also clear old generated visuals/dashboard so each report is clean.
        try:
            if os.path.exists("interface/dashboard.html"):
                os.remove("interface/dashboard.html")
        except Exception:
            pass

        try:
            if os.path.isdir("display"):
                for name in os.listdir("display"):
                    if name.lower().endswith(".png"):
                        try:
                            os.remove(os.path.join("display", name))
                        except Exception:
                            pass
        except Exception:
            pass

    def _plot_env_snapshot_fallback(self, env_res: Dict) -> Dict:
        """Deterministic multi-plot pack from current env metrics (data-scientist style)."""
        try:
            env_data = (env_res or {}).get("environmental_data", {})
            place = (env_res or {}).get("place") or self.context.get("location") or "Unknown"

            os.makedirs("display", exist_ok=True)
            ts = time.strftime("%y%m%d%H%M%S")
            file_paths = []

            def save_fig(fig, filename: str):
                path = os.path.join("display", filename)
                fig.savefig(path, dpi=220, bbox_inches="tight")
                plt.close(fig)
                file_paths.append(path)
                return path

            # --- 1) Core metrics (separate axes per unit) ---
            temp = env_data.get("temperature")
            humidity = env_data.get("humidity")
            wind = env_data.get("wind_speed") or env_data.get("windspeed_openmeteo")
            soil = env_data.get("soil_moisture_0_to_7cm") or env_data.get("soil_moisture_7_to_28cm")
            cloud = env_data.get("cloud_coverage")
            uv = env_data.get("uv_index")
            precip = env_data.get("rain_1h")
            if precip is None:
                precip = env_data.get("precip_mm")

            any_core = any(v is not None for v in [temp, humidity, wind, soil, cloud, uv, precip])
            if not any_core:
                return {"success": False, "error": "No numeric env metrics available to plot."}

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111)
            labels = []
            values = []
            for k, v in [
                ("Temp (°C)", temp),
                ("Humidity (%)", humidity),
                ("Wind (m/s)", wind),
                ("Soil moisture", soil),
                ("Cloud (%)", cloud),
                ("UV", uv),
                ("Precip", precip),
            ]:
                if v is not None:
                    labels.append(k)
                    try:
                        values.append(float(v))
                    except Exception:
                        values.append(None)
            # Filter any non-numeric conversions
            filtered = [(l, val) for l, val in zip(labels, values) if val is not None]
            if filtered:
                labels2 = [l for l, _ in filtered]
                values2 = [v for _, v in filtered]
                ax.bar(labels2, values2)
                ax.set_title(f"Core Environmental Metrics — {place}")
                ax.set_ylabel("Raw value (mixed units)")
                ax.tick_params(axis='x', rotation=18)
                core_path = save_fig(fig, f"core_{ts}.png")
            else:
                plt.close(fig)
                core_path = None

            # --- 2) Soil temperature profile ---
            soil_temp_vars = [
                ("0–7cm", env_data.get("soil_temperature_0_to_7cm")),
                ("7–28cm", env_data.get("soil_temperature_7_to_28cm")),
                ("28–100cm", env_data.get("soil_temperature_28_to_100cm")),
                ("100–255cm", env_data.get("soil_temperature_100_to_255cm")),
            ]
            st = [(lab, v) for lab, v in soil_temp_vars if v is not None]
            if st:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot([lab for lab, _ in st], [float(v) for _, v in st], marker="o")
                ax.set_title(f"Soil Temperature Profile — {place}")
                ax.set_xlabel("Depth")
                ax.set_ylabel("°C")
                save_fig(fig, f"soiltemp_{ts}.png")

            # --- 3) Soil moisture profile ---
            soil_moist_vars = [
                ("0–7cm", env_data.get("soil_moisture_0_to_7cm")),
                ("7–28cm", env_data.get("soil_moisture_7_to_28cm")),
                ("28–100cm", env_data.get("soil_moisture_28_to_100cm")),
                ("100–255cm", env_data.get("soil_moisture_100_to_255cm")),
            ]
            sm = [(lab, v) for lab, v in soil_moist_vars if v is not None]
            if sm:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot([lab for lab, _ in sm], [float(v) for _, v in sm], marker="o")
                ax.set_title(f"Soil Moisture Profile — {place}")
                ax.set_xlabel("Depth")
                ax.set_ylabel("Volumetric (unitless)")
                save_fig(fig, f"soilmoist_{ts}.png")

            # --- 4) Weather detail view ---
            gust = env_data.get("wind_gust")
            vis = env_data.get("visibility") or env_data.get("visibility_km")
            fig, ax = plt.subplots(figsize=(12, 5))
            detail_items = [
                ("Wind", wind),
                ("Gust", gust),
                ("Precip", precip),
                ("Cloud", cloud),
                ("Visibility", vis),
                ("UV", uv),
            ]
            detail_items = [(k, v) for k, v in detail_items if v is not None]
            if detail_items:
                ax.bar([k for k, _ in detail_items], [float(v) for _, v in detail_items])
                ax.set_title(f"Weather Detail Snapshot — {place}")
                ax.set_ylabel("Raw value (mixed units)")
                ax.tick_params(axis='x', rotation=18)
                save_fig(fig, f"weather_{ts}.png")
            else:
                plt.close(fig)

            # --- 5) Flowchart / digraph (system pipeline) ---
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_axis_off()
            ax.set_title("Waste-Dispo Intelligence Pipeline (Flowchart)")

            boxes = {
                "ENV": (0.10, 0.62, "ENV fetch\n(3 APIs)") ,
                "RESEARCH": (0.40, 0.62, "Web research\n+ simplify"),
                "PLOTS": (0.70, 0.62, "Visual analytics\n(multi-charts)"),
                "INSIGHTS": (0.25, 0.25, "Master insights\n(heuristics)"),
                "DASH": (0.60, 0.25, "Dashboard\n(render HTML)"),
            }

            def add_box(key, xywhtxt):
                x, y, text = xywhtxt
                w, h = 0.22, 0.18
                patch = FancyBboxPatch(
                    (x, y), w, h,
                    boxstyle="round,pad=0.02,rounding_size=0.02",
                    linewidth=1.2,
                    edgecolor="#334155",
                    facecolor="#0b1220",
                )
                ax.add_patch(patch)
                ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10, color="#e2e8f0")
                return (x, y, w, h)

            box_pos = {k: add_box(k, v) for k, v in boxes.items()}

            def arrow(src, dst):
                x1, y1, w1, h1 = box_pos[src]
                x2, y2, w2, h2 = box_pos[dst]
                start = (x1 + w1, y1 + h1/2)
                end = (x2, y2 + h2/2)
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#06b6d4"))

            arrow("ENV", "RESEARCH")
            arrow("RESEARCH", "PLOTS")
            # Downstream to insights + dashboard
            ax.add_patch(FancyArrowPatch((0.21, 0.62), (0.31, 0.34), arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#06b6d4"))
            ax.add_patch(FancyArrowPatch((0.82, 0.62), (0.71, 0.34), arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#06b6d4"))
            ax.add_patch(FancyArrowPatch((0.47, 0.25 + 0.09), (0.60, 0.25 + 0.09), arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#06b6d4"))

            save_fig(fig, f"flow_{ts}.png")

            # --- 6) Multiview panel for quick executive glance ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(f"Environmental Multiview — {place}")

            # (a) Core metrics
            ax = axes[0, 0]
            core_small = [("Temp", temp), ("Hum", humidity), ("Wind", wind), ("Soil", soil)]
            core_small = [(k, v) for k, v in core_small if v is not None]
            if core_small:
                ax.bar([k for k, _ in core_small], [float(v) for _, v in core_small])
                ax.set_title("Core")
            else:
                ax.text(0.5, 0.5, "No core metrics", ha="center", va="center")
            # (b) Soil temp
            ax = axes[0, 1]
            if st:
                ax.plot([lab for lab, _ in st], [float(v) for _, v in st], marker="o")
                ax.set_title("Soil Temp")
                ax.tick_params(axis='x', rotation=15)
            else:
                ax.text(0.5, 0.5, "No soil temp", ha="center", va="center")
            # (c) Soil moisture
            ax = axes[1, 0]
            if sm:
                ax.plot([lab for lab, _ in sm], [float(v) for _, v in sm], marker="o")
                ax.set_title("Soil Moisture")
                ax.tick_params(axis='x', rotation=15)
            else:
                ax.text(0.5, 0.5, "No soil moisture", ha="center", va="center")
            # (d) Detail
            ax = axes[1, 1]
            if detail_items:
                ax.bar([k for k, _ in detail_items[:5]], [float(v) for _, v in detail_items[:5]])
                ax.set_title("Detail")
                ax.tick_params(axis='x', rotation=15)
            else:
                ax.text(0.5, 0.5, "No detail metrics", ha="center", va="center")

            plt.tight_layout()
            multiview_path = save_fig(fig, f"multi_{ts}.png")

            primary = multiview_path or core_path or (file_paths[0] if file_paths else None)
            return {"success": True, "file_path": primary, "file_paths": file_paths, "source": "fallback_multi"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_full_report_pipeline(self, user_text: str) -> str:
        self.context["is_processing"] = True
        execution_log = []

        # Clear prior generated outputs so the dashboard reflects only this run.
        self._cleanup_previous_outputs()

        place = self.context.get("location") or "Unknown"

        def run_env():
            cached = self._get_cached_env(place)
            if cached:
                return {"success": True, "cached": True, **cached}

            res = self.agents["env_agent"]["module"].run({"place": place})
            if isinstance(res, dict) and res.get("success"):
                self._set_cached_env(place, res)
            return res

        def run_research():
            research_query = f"Waste management, pollution, and sustainability updates for {place}".strip()
            analyzer = ResearchAnalyzer(
                max_results=6,
                model_name=self.model_name,
                include_images=True,
                enable_fallback_model=False,
            )
            return analyzer.run_research(research_query)

        # 1-2) ENV + RESEARCH (multitask)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_env = ex.submit(run_env)
            fut_research = ex.submit(run_research)
            env_res = fut_env.result()
            search_res = fut_research.result()

        self.context["knowledge_base"]["env_agent"] = env_res
        self.context["knowledge_base"]["search_agent"] = search_res
        execution_log.append({"agent": "env_agent", "result": env_res})
        execution_log.append({"agent": "search_agent", "result": search_res})

        # 3) PLOT (deterministic: reduce agent work + avoid LLM plot failures)
        plot_res = self._plot_env_snapshot_fallback(env_res if isinstance(env_res, dict) else {})

        self.context["knowledge_base"]["plotter_agent"] = plot_res
        execution_log.append({"agent": "plotter_agent", "result": plot_res})

        if isinstance(plot_res, dict):
            for fp in (plot_res.get("file_paths") or []):
                if fp and fp not in self.context["created_files"]:
                    self.context["created_files"].append(fp)
            if plot_res.get("file_path") and plot_res.get("file_path") not in self.context["created_files"]:
                self.context["created_files"].append(plot_res.get("file_path"))

        plot_explanations = []

        # Always explain generated plots using vision.
        if isinstance(plot_res, dict) and plot_res.get("success"):
            plot_paths = plot_res.get("file_paths") or []
            if plot_res.get("file_path"):
                plot_paths = [plot_res.get("file_path")] + plot_paths

            unique_paths = list(dict.fromkeys([p for p in plot_paths if p]))
            plot_explanations.extend(self._analyze_plot_images(
                unique_paths,
                "Explain this chart in detail. Identify axes, key trends, and what it implies about waste/environment. "
                "Provide 3-5 concise insights and 1-2 recommended actions.",
            ))
            for item in plot_explanations:
                execution_log.append({"agent": "vision_agent", "result": {"success": True, "file_path": item.get("title"), "explanation": item.get("text")}})

        # 4) DASHBOARD (+ master insights)
        self.context.setdefault("knowledge_base", {})["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
        dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
        self.context["knowledge_base"]["dashboard_agent"] = dash_res
        execution_log.append({"agent": "dashboard_agent", "result": dash_res})
        if isinstance(dash_res, dict) and dash_res.get("file_path") and dash_res.get("file_path") not in self.context["created_files"]:
            self.context["created_files"].append(dash_res.get("file_path"))

        self.context["is_processing"] = False
        self.session.save(self.context)

        # Deterministic response (avoid an extra synthesis LLM call that can hang).
        env_data = (env_res or {}).get("environmental_data", {}) if isinstance(env_res, dict) else {}
        temperature = env_data.get("temperature")
        humidity = env_data.get("humidity")
        soil = env_data.get("soil_moisture_0_to_7cm") or env_data.get("soil_moisture_7_to_28cm")
        cached_flag = " (cached)" if isinstance(env_res, dict) and env_res.get("cached") else ""

        report_titles = []
        if isinstance(search_res, dict) and search_res.get("success"):
            for item in (search_res.get("report") or [])[:3]:
                t = (item or {}).get("title")
                if t:
                    report_titles.append(t)

        dash_path = None
        if isinstance(dash_res, dict):
            dash_path = dash_res.get("file_path")

        plot_path = plot_res.get("file_path") if isinstance(plot_res, dict) else None

        lines = [
            f"Environmental snapshot for {place}{cached_flag}:",
            f"- Temperature: {temperature if temperature is not None else 'N/A'} °C",
            f"- Humidity: {humidity if humidity is not None else 'N/A'} %",
            f"- Soil moisture: {soil if soil is not None else 'N/A'}",
        ]
        if report_titles:
            lines.append("Top research signals:")
            for t in report_titles:
                lines.append(f"- {t}")
        if plot_path:
            lines.append(f"Plot saved: {plot_path}")
        if dash_path:
            lines.append(f"Dashboard: {dash_path}")

        plot_analysis = self._format_plot_explanations(plot_explanations)
        if plot_analysis:
            lines.append(plot_analysis)

        report_text = "\n".join(lines)
        report_text = self._remove_tags_emojis_and_special_chars(report_text)
        report_text = self._limit_chat_paragraphs(report_text, max_paragraphs=3)
        try:
            summarizer = self.agents.get("summarizer_agent", {}).get("module")
            if summarizer:
                summary_result = summarizer.run({"text": report_text})
                if isinstance(summary_result, dict):
                    summarized = summary_result.get("summary") or summary_result.get("text") or summary_result.get("output")
                    if summarized:
                        summarized, tables = self._parse_summary_output(summarized)
                        self._stage_summary_tables(tables)
                        if tables and self.agents["dashboard_agent"]["module"]:
                            self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
                            dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
                            self.context["knowledge_base"]["dashboard_agent"] = dash_res
                            if isinstance(dash_res, dict) and dash_res.get("file_path") and dash_res.get("file_path") not in self.context["created_files"]:
                                self.context["created_files"].append(dash_res.get("file_path"))
                            self.session.save(self.context)
                        return summarized
                elif isinstance(summary_result, str) and summary_result.strip():
                    summarized, tables = self._parse_summary_output(summary_result)
                    self._stage_summary_tables(tables)
                    if tables and self.agents["dashboard_agent"]["module"]:
                        self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
                        dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
                        self.context["knowledge_base"]["dashboard_agent"] = dash_res
                        if isinstance(dash_res, dict) and dash_res.get("file_path") and dash_res.get("file_path") not in self.context["created_files"]:
                            self.context["created_files"].append(dash_res.get("file_path"))
                        self.session.save(self.context)
                    return summarized
        except Exception as error:
            print(f"Summarizer agent failed in full report pipeline: {error}")

        return report_text

    def _should_run_plotter(self, params: Dict, user_text: str) -> bool:
        # For full reports, allow plotting even if the user didn't provide explicit data.
        if self._is_full_report_request(user_text):
            return True
        query = (params or {}).get("query")
        if query and isinstance(query, str) and query.strip():
            return True
        # Only run plotter if user explicitly asked for a plot/chart and provided data.
        if not user_text:
            return False
        text = user_text.lower()
        asked = any(k in text for k in ["plot", "chart", "graph", "visualize", "visualise"])
        has_numbers = any(ch.isdigit() for ch in text)
        return asked and has_numbers

    def _run_actions(self, actions: List[Dict], user_text: str, force_full_actions: bool = False) -> str:
        # If user is requesting a full dashboard/report, clear previous generated outputs
        if self._is_full_report_request(user_text) or any(a.get("intent") == "dashboard_agent" for a in (actions or [])):
            self._cleanup_previous_outputs()

        # --- Hard Guardrail to intercept over-eager LLM routing ---
        valid_actions = []
        for a in (actions or []):
            intent = a.get("intent")
            
            # If it's a vision tool, enforce strict image presence
            if intent in ["vision_agent", "classifier_agent"]:
                provided_path = a.get("parameters", {}).get("image_path", "")
                has_image = bool(self._extract_image_path(user_text) or self._extract_image_path(provided_path))
                
                if not has_image:
                    print(f"🔀 Guardrail: Intercepted and blocked '{intent}'. No image file was detected.")
                    continue # Drop the action
                    
            valid_actions.append(a)
            
        actions = valid_actions
        
        # If the guardrail stripped all actions (e.g., it only wanted to classify an absent image), fallback.
        if not actions:
            return "Action blocked. To classify or analyze an object, please provide a valid image file."

        # Reduce agent workload unless a full report is explicitly requested.
        if isinstance(actions, list) and not (self._is_full_report_request(user_text) or force_full_actions):
            actions = actions[:1]

        self.context["is_processing"] = True
        execution_log = []
        plot_explanations = []

        # Defer dashboard until the end so it receives the updated knowledge base.
        dashboard_requested = any(a.get("intent") == "dashboard_agent" for a in actions)
        actions = [a for a in actions if a.get("intent") != "dashboard_agent"]

        # Run core data agents first for better downstream prompts.
        priority = {"env_agent": 0, "search_agent": 1, "plotter_agent": 2, "vision_agent": 3}
        try:
            actions.sort(key=lambda a: priority.get(a.get("intent"), 99))
        except Exception:
            pass

        # Parallelize env + research when both are requested.
        env_action = next((a for a in actions if a.get("intent") == "env_agent"), None)
        search_action = next((a for a in actions if a.get("intent") == "search_agent"), None)
        if env_action and search_action:
            actions = [a for a in actions if a.get("intent") not in ("env_agent", "search_agent")]
            place = (env_action.get("parameters") or {}).get("place") or self.context.get("location") or "Unknown"
            query = (search_action.get("parameters") or {}).get("query") or ""

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_env = ex.submit(self._run_env_cached, place)
                fut_search = ex.submit(self.agents["search_agent"]["module"].run, {"query": query})
                env_res = fut_env.result()
                search_res = fut_search.result()

            execution_log.append({"agent": "env_agent", "result": env_res})
            execution_log.append({"agent": "search_agent", "result": search_res})
            self.context["knowledge_base"]["env_agent"] = env_res
            self.context["knowledge_base"]["search_agent"] = search_res
            self.context["location"] = place

        # Store extracted image path for potential use by classifier and vision
        extracted_image = self._extract_image_path(user_text)

        for action in actions:
            intent = action.get("intent")
            params = action.get("parameters", {})

            if intent in self.agents and self.agents[intent]["module"]:
                # ENV caching: reuse env data for 1 hour.
                if intent == "env_agent":
                    place = params.get("place") or self.context.get("location") or "Unknown"
                    res = self._run_env_cached(place)
                    execution_log.append({"agent": intent, "result": res})

                else:
                    # If it's a classifier agent, we also need to trigger vision agent on the same image.
                    # We'll capture the image path from the action parameters or from extracted_image.
                    image_path = params.get("image_path") or extracted_image
                    is_classifier = (intent == "classifier_agent")

                    if intent == "plotter_agent" and (not force_full_actions) and not self._should_run_plotter(params, user_text):
                        res = {
                            "success": False,
                            "error": "Plotter skipped: provide a plot request with actual data points (e.g. '2010=40, 2020=55').",
                        }
                    else:
                        if intent == "plotter_agent" and (not params.get("query")):
                            # Auto-build a sensible plot request for full reports.
                            env_res = self.context.get("knowledge_base", {}).get("env_agent", {})
                            env_data = env_res.get("environmental_data", {}) if isinstance(env_res, dict) else {}
                            temperature = env_data.get("temperature")
                            humidity = env_data.get("humidity")
                            wind = env_data.get("wind_speed") or env_data.get("windspeed_openmeteo")
                            soil = env_data.get("soil_moisture_0_to_7cm") or env_data.get("soil_moisture_7_to_28cm")

                            if any(v is not None for v in [temperature, humidity, wind, soil]):
                                place = (env_res.get("place") if isinstance(env_res, dict) else None) or self.context.get("location")
                                # Synthetic series to enable richer plot variety (6-hour profile)
                                synth = {
                                    "temp_trend_c": [
                                        temperature - 1.2 if temperature is not None else None,
                                        temperature - 0.7 if temperature is not None else None,
                                        temperature - 0.2 if temperature is not None else None,
                                        temperature + 0.3 if temperature is not None else None,
                                        temperature + 0.6 if temperature is not None else None,
                                        temperature + 0.9 if temperature is not None else None,
                                    ],
                                    "humidity_trend_pct": [
                                        humidity + 2 if humidity is not None else None,
                                        humidity + 1 if humidity is not None else None,
                                        humidity if humidity is not None else None,
                                        humidity - 1 if humidity is not None else None,
                                        humidity - 2 if humidity is not None else None,
                                        humidity - 3 if humidity is not None else None,
                                    ],
                                }
                                # Derived risk indices (0-100) for extra plot variants
                                decomp = (
                                    (max(0, min(60, (temperature - 20) * 3)) if temperature is not None else 0)
                                    + (max(0, min(40, (humidity - 50) * 0.8)) if humidity is not None else 0)
                                )
                                odor = (
                                    (max(0, (humidity - 60) * 1.1) if humidity is not None else 0)
                                    + (max(0, (4 - wind) * 8) if wind is not None else 0)
                                )
                                risk_pack = {
                                    "decomposition_risk": int(round(max(0, min(100, decomp)))) if isinstance(decomp, (int, float)) else 0,
                                    "odor_risk": int(round(max(0, min(100, odor)))) if isinstance(odor, (int, float)) else 0,
                                }
                                auto_query = (
                                    "Create a bar chart for current environmental snapshot (one bar per metric): "
                                    f"temperature_c={temperature}, humidity_percent={humidity}, wind_speed={wind}, soil_moisture={soil}. "
                                    f"Title the plot with the place: {place}. "
                                    f"Also include synthetic trend series for the last 6 hours: temp_trend_c={synth['temp_trend_c']}, "
                                    f"humidity_trend_pct={synth['humidity_trend_pct']}. "
                                    f"Include derived risk metrics: {risk_pack}."
                                )
                                params = {**params, "query": auto_query}
                            else:
                                params = {**params, "query": user_text}
                        res = self.agents[intent]["module"].run(params)

                    execution_log.append({"agent": intent, "result": res})

                    # --- If this is a classifier, also run the vision agent on the same image ---
                    if is_classifier and image_path and os.path.exists(image_path):
                        # Check if we already have an analysis for this image to avoid duplicates
                        existing = self.context.get("knowledge_base", {}).get("image_analyses", [])
                        existing_paths = {item.get("file_path") for item in existing if item.get("file_path")}
                        if image_path not in existing_paths:
                            # Run vision agent
                            vision_res = self.agents["vision_agent"]["module"].run({
                                "image_path": image_path,
                                "prompt": "Explain this image in detail and relate it to waste, pollution, and sustainability where relevant."
                            })
                            # Store the analysis
                            try:
                                rel = os.path.relpath(image_path, start=os.getcwd()).replace("\\", "/")
                            except Exception:
                                rel = str(image_path).replace("\\", "/")
                            dash_src = rel if rel.startswith("../") else f"../{rel}"
                            self.context.setdefault("knowledge_base", {}).setdefault("image_analyses", []).append({
                                "title": f"Image Analysis: {os.path.basename(image_path)}",
                                "file_path": image_path,
                                "image": dash_src,
                                "explanation": vision_res,
                            })
                            execution_log.append({
                                "agent": "vision_agent",
                                "result": {"success": True, "file_path": image_path, "explanation": vision_res}
                            })
                            print(f"📸 Auto-triggered vision analysis for classified image: {image_path}")

                # Track created files
                if isinstance(res, dict):
                    file_path = res.get("file_path") or res.get("saved_to")
                    file_paths = res.get("file_paths") or []
                    for fp in [file_path] + file_paths:
                        if fp and fp not in self.context["created_files"]:
                            self.context["created_files"].append(fp)

                    # Always explain generated plots using the vision agent
                    if intent == "plotter_agent" and res.get("success"):
                        plot_paths = res.get("file_paths") or []
                        if res.get("file_path"):
                            plot_paths = [res.get("file_path")] + plot_paths

                        unique_paths = list(dict.fromkeys([p for p in plot_paths if p]))
                        new_explanations = self._analyze_plot_images(
                            unique_paths,
                            "Explain this chart in detail. Identify the axes, key trends, outliers, and what it implies. "
                            "Provide 3-5 concise insights and 1-2 recommended actions related to waste/environment.",
                        )
                        plot_explanations.extend(new_explanations)
                        for item in new_explanations:
                            execution_log.append({"agent": "vision_agent", "result": {"success": True, "file_path": item.get("title"), "explanation": item.get("text")}})

                # Update Knowledge Base
                self.context["knowledge_base"][intent] = res
                if "place" in params:
                    old_location = self.context.get("location")
                    new_location = params["place"]
                    if old_location and old_location != new_location:
                        # Invalidate cache when location is explicitly changed
                        cache = self._get_cache()
                        cache.pop("env", None)
                        self._last_cached_location = None
                        print(f"📍 Location updated from '{old_location}' to '{new_location}'. Cache invalidated.")
                    self.context["location"] = new_location

        # Run dashboard last if requested
        if dashboard_requested and self.agents["dashboard_agent"]["module"]:
            # Keep the dashboard informative even on single-agent runs.
            self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
            dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
            execution_log.append({"agent": "dashboard_agent", "result": dash_res})
            self.context["knowledge_base"]["dashboard_agent"] = dash_res
            if isinstance(dash_res, dict) and dash_res.get("file_path") and dash_res.get("file_path") not in self.context["created_files"]:
                self.context["created_files"].append(dash_res.get("file_path"))

        self.context["is_processing"] = False
        self.context["last_suggested_actions"] = []
        self.session.save(self.context)
        response = self._synthesize_final_response(execution_log, user_text)

        if self.context.get("knowledge_base", {}).get("summary_tables") and self.agents["dashboard_agent"]["module"]:
            self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
            dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
            self.context["knowledge_base"]["dashboard_agent"] = dash_res
            if isinstance(dash_res, dict) and dash_res.get("file_path") and dash_res.get("file_path") not in self.context["created_files"]:
                self.context["created_files"].append(dash_res.get("file_path"))
            self.session.save(self.context)
        return response

    def process_input(self, user_text):
        # Check for explicit location change command
        location_match = re.search(r"(?:change|set|update)\s+location\s+to\s+([^\n,\.]+)", user_text, re.IGNORECASE)
        if location_match:
            new_location = location_match.group(1).strip()
            old_location = self.context.get("location")
            if old_location and old_location.lower() != new_location.lower():
                # Clear cache when location changes
                cache = self._get_cache()
                cache.pop("env", None)
                self._last_cached_location = None
                self.context["location"] = new_location
                self.session.save(self.context)
                print(f"📍 Location explicitly changed from '{old_location}' to '{new_location}'. Cache cleared.")
                return f"✅ Location updated to '{new_location}'. Environmental data cache cleared for the new location."
            elif not old_location:
                self.context["location"] = new_location
                self.session.save(self.context)
                return f"✅ Location set to '{new_location}'."
            else:
                return f"📍 Location is already set to '{old_location}'."

        # Direct object classification path (uses summarizer but strips tables)
        if self._wants_classification(user_text):
            image_path = self._extract_image_path(user_text)
            if not image_path:
                # Try to find a demo image if the user didn't upload one
                for demo in ["3480.webp", os.path.join("image", "3480.webp")]:
                    if os.path.exists(demo):
                        image_path = demo
                        break

            if not image_path:
                return "We could not find the image you uploaded. Please confirm whether this is the image you intended to use."

            # Run classifier
            classification = self._classify_image(image_path)

            # Also run vision agent on the same image (if not already analysed)
            existing = self.context.get("knowledge_base", {}).get("image_analyses", [])
            existing_paths = {item.get("file_path") for item in existing if item.get("file_path")}
            vision_explanation = None
            if image_path not in existing_paths:
                vision_explanation = self.agents["vision_agent"]["module"].run({
                    "image_path": image_path,
                    "prompt": "Explain this image in detail and relate it to waste, pollution, and sustainability where relevant."
                })
                # Remove any file paths from the vision explanation
                vision_explanation = self._remove_file_paths(vision_explanation)

            if isinstance(classification, dict) and classification.get("success"):
                detections = classification.get("detections") or []
                try:
                    rel = os.path.relpath(image_path, start=os.getcwd()).replace("\\", "/")
                except Exception:
                    rel = str(image_path).replace("\\", "/")
                dash_src = rel if rel.startswith("../") else f"../{rel}"

                self.context.setdefault("knowledge_base", {}).setdefault("classifier_results", []).append({
                    "title": f"Classification: {os.path.basename(image_path)}",
                    "file_path": image_path,
                    "image": dash_src,
                    "result": classification,
                })

                # Store vision analysis if we have one
                if vision_explanation:
                    self.context.setdefault("knowledge_base", {}).setdefault("image_analyses", []).append({
                        "title": f"Image Analysis: {os.path.basename(image_path)}",
                        "file_path": image_path,
                        "image": dash_src,
                        "explanation": vision_explanation,
                    })

                # Update master insights and dashboard
                self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
                dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
                if isinstance(dash_res, dict):
                    self.context["knowledge_base"]["dashboard_agent"] = dash_res
                self.session.save(self.context)

                # Build a combined description for the summarizer
                if detections:
                    detected_items = []
                    for item in detections[:8]:
                        label = item.get("label") or "unknown"
                        confidence = item.get("confidence")
                        if confidence is not None:
                            detected_items.append(f"{label} ({confidence*100:.0f}% confidence)")
                        else:
                            detected_items.append(label)
                    classification_text = "I detected: " + ", ".join(detected_items) + "."
                else:
                    classification_text = "I classified the image, but no objects were detected."

                if vision_explanation:
                    combined_text = f"{classification_text} Here's what I see: {vision_explanation}"
                else:
                    combined_text = classification_text

                # Use the summarizer to produce a friendly, conversational answer (with tables stripped)
                raw_summary = self._summarize_for_user(combined_text, context="classification_and_analysis")
                # Also strip any markdown tables that might have slipped through
                cleaned, _ = self._parse_summary_output(raw_summary)
                # Remove any remaining file paths just in case
                cleaned = self._remove_file_paths(cleaned)
                return self._remember_turn(user_text, cleaned)

            return self._remember_turn(user_text, classification.get("error") if isinstance(classification, dict) and classification.get("error") else "Classification failed.")

        # Direct image analysis path (demo/upload).
        if self._wants_image_analysis(user_text):
            image_path = self._extract_image_path(user_text)
            if not image_path:
                for demo in ["3480.webp", os.path.join("image", "3480.webp")]:
                    if os.path.exists(demo):
                        image_path = demo
                        break

            if not image_path:
                return "We could not find the image you uploaded. Please confirm whether this is the image you intended to use."

            explanation = self.agents["vision_agent"]["module"].run({
                "image_path": image_path,
                "prompt": "Explain this image in detail and relate it to waste, pollution, and sustainability where relevant."
            })
            explanation = self._remove_file_paths(explanation)

            # Dashboard is generated under interface/, so use ../ relative src.
            try:
                rel = os.path.relpath(image_path, start=os.getcwd()).replace("\\", "/")
            except Exception:
                rel = str(image_path).replace("\\", "/")
            dash_src = rel if rel.startswith("../") else f"../{rel}"

            self.context.setdefault("knowledge_base", {}).setdefault("image_analyses", []).append({
                "title": f"Image Analysis: {os.path.basename(image_path)}",
                "file_path": image_path,
                "image": dash_src,
                "explanation": explanation,
            })

            # Refresh insights and render dashboard so the analysis is visible.
            self.context["knowledge_base"]["master_insights"] = self._build_master_insights(self.context["knowledge_base"])
            dash_res = self.agents["dashboard_agent"]["module"].run({"knowledge_base": self.context["knowledge_base"]})
            if isinstance(dash_res, dict):
                self.context["knowledge_base"]["dashboard_agent"] = dash_res
            self.session.save(self.context)

            # Summarize the image explanation for user-friendly chat output (strip tables)
            raw = f"Image analysis for {os.path.basename(image_path)}: {explanation}"
            summary = self._summarize_for_user(raw, context="image_analysis")
            cleaned, _ = self._parse_summary_output(summary)
            cleaned = self._remove_file_paths(cleaned)
            return self._remember_turn(user_text, cleaned)

        # Implicit consent: user confirms a previously suggested action.
        if self._is_affirmative_text(user_text) and self.context.get("last_suggested_actions"):
            actions = self.context.get("last_suggested_actions") or []
            return self._remember_turn(user_text, self._run_actions(actions, user_text, force_full_actions=True))

        # Zero-permission auto triggering.
        auto_actions = self._auto_actions_from_text(user_text)
        if auto_actions:
            return self._remember_turn(user_text, self._run_actions(auto_actions, user_text, force_full_actions=True))

        if self._is_full_report_request(user_text):
            return self._remember_turn(user_text, self._run_full_report_pipeline(user_text))

        # 1. Decide Intent
        # Single-model policy: always use self.model_name.
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=self._build_chat_messages(user_text)
            )
        except Exception:
            response = ollama.chat(
                model=self.model_name,
                messages=self._build_chat_messages(user_text)
            )
        ai_content = response['message']['content']

        if "[" in ai_content and "intent" in ai_content:
            try:
                json_str = re.search(r"\[.*\]", ai_content, re.DOTALL).group()
                actions = json.loads(json_str)
                return self._remember_turn(user_text, self._run_actions(actions, user_text, force_full_actions=self._is_full_report_request(user_text)))
            except Exception as e:
                print(f"⚙️ Orchestration Error: {e}")
                return self._remember_turn(user_text, ai_content)
        
        suggested_actions = self._capture_suggested_actions(ai_content)
        if suggested_actions:
            self.context["last_suggested_actions"] = suggested_actions
            self.session.save(self.context)
        return self._remember_turn(user_text, ai_content)

    def _grade_response_quality(self, response_text: str, data_completeness: float = 0.8) -> Dict:
        """
        Grades the quality of a response based on multiple factors.
        Returns a dict with grade and confidence score (0-100).
        """
        score = 50  # base score
        
        # Factor 1: Response length and detail
        word_count = len(response_text.split())
        if word_count > 150:
            score += 20
        elif word_count > 75:
            score += 10
        
        # Factor 2: Presence of actionable insights
        if any(keyword in response_text.lower() for keyword in ["action", "recommendation", "suggest", "improve"]):
            score += 15
        
        # Factor 3: Data completeness
        score += int(data_completeness * 15)
        
        # Factor 4: Specific metrics/numbers
        if any(char.isdigit() for char in response_text):
            score += 10
        
        # Determine grade
        if score >= 85:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 55:
            grade = "C"
        else:
            grade = "D"
        
        return {"grade": grade, "score": min(score, 100), "completeness": data_completeness}

    def _synthesize_final_response(self, logs, original_query):
        data_summary = json.dumps(logs, indent=2)
        
        # Calculate data completeness
        successful_agents = sum(1 for log in logs if isinstance(log.get("result"), dict) and log.get("result", {}).get("success"))
        total_agents = len(logs) if logs else 1
        data_completeness = successful_agents / total_agents if total_agents > 0 else 0
        
        # --- 1. CONDITIONAL SUMMARIZATION (The Threshold) ---
        # Only use the technical summarizer if the data payload is massive (>3000 chars).
        # Otherwise, the Master can easily handle the raw JSON directly.
        is_extreme_data = len(data_summary) > 3000 
        
        backend_context = ""
        
        if is_extreme_data:
            print("📊 Massive data detected. Invoking Technical Summarizer to condense logs...")
            summary_input = f"User request: {original_query}\n\nExecution log:\n{data_summary}"
            try:
                summarizer = self.agents.get("summarizer_agent", {}).get("module")
                if summarizer:
                    summary_result = summarizer.run({"text": summary_input})
                    if isinstance(summary_result, dict):
                        backend_context = summary_result.get("summary") or summary_result.get("text") or summary_result.get("output")
                    elif isinstance(summary_result, str):
                        backend_context = summary_result
            except Exception as error:
                print(f"Summarizer agent failed: {error}")
                backend_context = data_summary # Fallback
        else:
            print("💬 Standard payload. Bypassing summarizer for direct Master interaction.")
            backend_context = data_summary

        # --- 2. HUMANIZATION (The Mouthpiece) ---
        chat_prompt = (
            f"You are {self.master_name}, a highly intelligent, empathetic, and premium environmental assistant for {self.system_name}.\n"
            f"The user asked: '{original_query}'\n\n"
            f"Here is the backend data gathered by our system:\n{backend_context}\n\n"
            f"INSTRUCTIONS FOR YOUR RESPONSE:\n"
            f"1. Be natural, conversational, and pleasing to read. Speak directly to the user as a helpful humanized expert.\n"
            f"2. NEVER mention 'agents', 'JSON', 'execution logs', 'queries', or how the backend works. Be completely seamless.\n"
            f"3. Synthesize the findings into clear insights. If the data lacks specific answers, politely guide the user on what to do next or ask a clarifying question.\n"
            f"4. Keep formatting clean and elegant (short paragraphs, use bullet points only if listing clear actions)."
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': chat_prompt}],
                stream=False
            )
            response_text = response['message']['content']
        except Exception as e:
            print(f"Chat generation failed: {e}")
            response_text = "I've checked our databases for you. We don't have highly specific data on that exact item right now, but how else can I assist you with your sustainability goals?"

        # Parse out any tables for the dashboard, keep the text for chat
        response_text, tables = self._parse_summary_output(response_text)
        self._stage_summary_tables(tables)

        # Keep the chat answer compact
        response_text = self._limit_chat_paragraphs(response_text, max_paragraphs=3)
        # Remove any file paths that may have crept in
        response_text = self._remove_file_paths(response_text)
        
        # Optional: You may want to comment out this grading append for production 
        # so it doesn't ruin the "human" vibe with a sudden "[Data: 80% | Grade: C]" tag.
        quality = self._grade_response_quality(response_text, data_completeness)
        if quality["grade"] in ["C", "D"]:
            pass # Or handle quietly in backend logs instead of showing the user
        
        return response_text

    def _summarize_for_user(self, raw_text: str, context: Optional[str] = None) -> str:
        """Use the summarizer agent to make raw agent outputs user-friendly.
        Falls back to returning the original text on error.
        """
        if not raw_text:
            return ""
        # Remove any file paths before summarization
        raw_text = self._remove_file_paths(raw_text)
        try:
            summarizer = self.agents.get("summarizer_agent", {}).get("module")
            if summarizer:
                payload = {"text": raw_text}
                if context:
                    # For classification, provide a custom prompt to get a friendly, conversational response
                    if context in ["classification", "classification_and_analysis"]:
                        payload["context"] = (
                            "You are a friendly environmental assistant. "
                            "Summarize the following classification and analysis results in a natural, human-like way. "
                            "Explain what the image shows, what objects were detected, and what environmental implications arise. "
                            "Keep it concise, warm, and easy to understand. Do not use bullet points or tables; just a short paragraph."
                        )
                    else:
                        payload["context"] = context
                res = summarizer.run(payload)
                if isinstance(res, dict):
                    summary = res.get("summary") or res.get("text") or res.get("output") or raw_text
                elif isinstance(res, str):
                    summary = res
                else:
                    summary = raw_text
                # Strip any markdown tables from the summary
                cleaned, _ = self._parse_summary_output(summary)
                # Remove any remaining file paths
                cleaned = self._remove_file_paths(cleaned)
                return cleaned
        except Exception:
            pass
        return raw_text
    
    def _clean_response_text(self, text: str) -> str:
        """Remove markdown and special formatting from response."""
        if not text:
            return text
        
        # Remove LaTeX-style math: $...$
        text = re.sub(r'\$[^$]*\$', '', text)
        
        # Remove markdown bold/italic: **text**, *text*, __text__, _text_
        text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]*)\*', r'\1', text)
        text = re.sub(r'__([^_]*)__', r'\1', text)
        text = re.sub(r'_([^_]*)_', r'\1', text)
        
        # Remove markdown headers: ### text -> text
        text = re.sub(r'#{1,6}\s+', '', text)
        
        # Remove markdown code blocks
        text = re.sub(r'```[^`]*```', '', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        
        # Remove markdown links: [text](url) -> text
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
        
        # Clean up spacing
        text = re.sub(r'\n\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    def _remove_tags_emojis_and_special_chars(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'(?<!\w)#\w+', '', text)
        text = re.sub(r'[\U0001F300-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]', '', text)
        text = re.sub(r"[^\w\s.,;:!?()\-/%&'\"\n]", ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def _extract_markdown_tables(self, text: str):
        if not text:
            return [], text

        lines = text.splitlines()
        tables = []
        kept_lines = []
        index = 0

        def looks_like_separator(candidate: str) -> bool:
            return bool(re.match(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$', candidate))

        while index < len(lines):
            current = lines[index]
            next_line = lines[index + 1] if index + 1 < len(lines) else ""

            if "|" in current and looks_like_separator(next_line):
                headers = [cell.strip() for cell in current.strip().strip("|").split("|")]
                rows = []
                index += 2

                while index < len(lines):
                    row_line = lines[index]
                    if "|" not in row_line or not row_line.strip():
                        break
                    row = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
                    if any(cell for cell in row):
                        rows.append(row)
                    index += 1

                if headers and rows:
                    tables.append({"headers": headers, "rows": rows})
                continue

            kept_lines.append(current)
            index += 1

        return tables, "\n".join(kept_lines)

    def _limit_chat_paragraphs(self, text: str, max_paragraphs: int = 3) -> str:
        if not text:
            return text

        paragraphs = [part.strip() for part in re.split(r'\n{2,}', text) if part.strip()]
        if not paragraphs:
            return text.strip()

        if len(paragraphs) > max_paragraphs:
            sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', ' '.join(paragraphs)) if sentence.strip()]
            if sentences:
                per_paragraph = max(1, (len(sentences) + max_paragraphs - 1) // max_paragraphs)
                paragraphs = [
                    ' '.join(sentences[start:start + per_paragraph]).strip()
                    for start in range(0, len(sentences), per_paragraph)
                ]

        return '\n\n'.join(paragraphs[:max_paragraphs])

    def _parse_summary_output(self, summary_text: str):
        tables, text_without_tables = self._extract_markdown_tables(summary_text)
        cleaned_text = self._clean_response_text(text_without_tables)
        cleaned_text = self._remove_tags_emojis_and_special_chars(cleaned_text)
        cleaned_text = self._limit_chat_paragraphs(cleaned_text, max_paragraphs=3)
        return cleaned_text, tables

    def _stage_summary_tables(self, tables):
        kb = self.context.setdefault("knowledge_base", {})
        kb["summary_tables"] = tables or []
        self.session.save(self.context)

    def get_status_update(self):
        """Provides quick status updates while Master is working."""
        if not self.context["is_processing"]:
            return "The system is currently idle and ready for your next command."
        
        return f"{self.master_name} is currently orchestrating multiple agents, synthesizing global research, and rendering your visual layer. Please hold on a moment; perfection takes time."

# ------------------------------------------------------------------
# WRAPPERS
# ------------------------------------------------------------------
class EnvAgentWrapper:
    def __init__(self): from backend.Agents.env_live import EnvironmentalDataFetcher; self.f = EnvironmentalDataFetcher()
    def run(self, p): return self.f.fetch_data(p.get("place", "Unknown"))

class VisionAgentWrapper:
    def __init__(self): from backend.image.explain import Vision; self.a = Vision()
    def run(self, p): return self.a.explain_image(p.get("image_path", ""), prompt=p.get("prompt"))

class PlotterAgentWrapper:
    def __init__(self): from backend.Agents.Plotter import AIPlotterApp; self.p = AIPlotterApp()
    def run(self, p): return self.p.generate_plot(p.get("query", ""))

class ResearchAgentWrapper:
    def __init__(self):
        from backend.Agents.research import ResearchAnalyzer
        self.r = ResearchAnalyzer(max_results=6, model_name="gemma4:31b-cloud", include_images=True, enable_fallback_model=False)
    def run(self, p): return self.r.run_research(p.get("query", ""))

class SummarizerAgentWrapper:
    def __init__(self):
        self.s = TechnicalTextSummarizer()

    def run(self, p):
        text = (p or {}).get("text", "")
        if not isinstance(text, str) or not text.strip():
            return {"success": False, "error": "No text provided for summarization."}

        try:
            summary = self.s.summarize(text)
            return {"success": True, "summary": summary}
        except Exception as error:
            return {"success": False, "error": str(error)}

class ClassifierAgentWrapper:
    def __init__(self):
        from backend.Agents.classifier_agent import ClassifierAgent
        self.c = ClassifierAgent()

    def run(self, p):
        return self.c.run(p or {})

class DashboardAgentWrapper:
    def __init__(self): from backend.output.Dashboard import DashboardModule; self.d = DashboardModule()
    def run(self, p):
        kb = p.get("knowledge_base", {}) or {}
        env_res = kb.get("env_agent") if isinstance(kb.get("env_agent"), dict) else {}
        env_data = env_res.get("environmental_data", {}) if env_res.get("success") else {}
        place = env_res.get("place") or "Unknown"
        updated_at = time.strftime("%I:%M %p").lstrip("0")
        cached_flag = env_res.get("cached", False)

        used_env_keys = set()

        def pick_value(keys: List[str]):
            for key in keys:
                if key in env_data and env_data.get(key) is not None:
                    used_env_keys.add(key)
                    return env_data.get(key), key
            return None, None

        def add_metric(items: List[Dict[str, str]], label: str, value, unit: str = ""):
            if value is None:
                return
            text = f"{value}{unit}" if unit else str(value)
            items.append({"label": label, "value": text})

        atmosphere = []
        soil = []
        weather = []

        temp, _ = pick_value(["temperature"])
        humidity, _ = pick_value(["humidity"])
        pressure, _ = pick_value(["pressure"])
        surface_pressure, _ = pick_value(["surface_pressure"])
        wind, _ = pick_value(["wind_speed", "windspeed_openmeteo"])
        gust, _ = pick_value(["wind_gust"])
        wind_dir, _ = pick_value(["wind_direction"])
        uv, _ = pick_value(["uv_index"])
        visibility, vis_key = pick_value(["visibility_km", "visibility"])
        cloud, _ = pick_value(["cloud_coverage"])
        rain, _ = pick_value(["rain_1h", "precip_mm", "precipitation"])

        add_metric(atmosphere, "Air Temperature", temp, "°C")
        add_metric(atmosphere, "Humidity", humidity, "%")
        add_metric(atmosphere, "Atmospheric Pressure", pressure, " hPa")
        add_metric(atmosphere, "Surface Pressure", surface_pressure, " hPa")
        add_metric(atmosphere, "Wind Speed", wind, " m/s")
        add_metric(atmosphere, "Wind Direction", wind_dir, "°")
        add_metric(atmosphere, "UV Index", uv)
        if visibility is not None:
            add_metric(atmosphere, "Visibility", visibility, " km" if vis_key == "visibility_km" else " m")

        add_metric(weather, "Cloud Cover", cloud, "%")
        add_metric(weather, "Rain (1h)", rain, " mm")
        add_metric(weather, "Wind Gust", gust, " m/s")

        soil_moisture = []
        for key, label in [
            ("soil_moisture_0_to_7cm", "Surface Soil Moisture"),
            ("soil_moisture_7_to_28cm", "Shallow Soil Moisture"),
            ("soil_moisture_28_to_100cm", "Mid Soil Moisture"),
            ("soil_moisture_100_to_255cm", "Deep Soil Moisture"),
        ]:
            val, _ = pick_value([key])
            add_metric(soil_moisture, label, val)

        soil_temp = []
        for key, label in [
            ("soil_temperature_0_to_7cm", "Surface Soil Temperature"),
            ("soil_temperature_7_to_28cm", "Shallow Soil Temperature"),
            ("soil_temperature_28_to_100cm", "Mid Soil Temperature"),
            ("soil_temperature_100_to_255cm", "Deep Soil Temperature"),
        ]:
            val, _ = pick_value([key])
            add_metric(soil_temp, label, val, "°C")

        soil = soil_moisture + soil_temp

        has_env_signals = any(value is not None for value in [temp, humidity, wind, rain]) or bool(soil_moisture)

        def clamp_score(val):
            try:
                return max(0, min(100, int(round(val))))
            except Exception:
                return 0

        decomp_score = 0
        if temp is not None:
            decomp_score += max(0, min(60, (temp - 20) * 3))
        if humidity is not None:
            decomp_score += max(0, min(40, (humidity - 50) * 0.8))

        leachate_score = 0
        if rain is not None:
            leachate_score += rain * 18
        if soil_moisture:
            try:
                leachate_score += float(soil_moisture[0]["value"]) * 100
            except Exception:
                pass

        odor_score = 0
        if humidity is not None:
            odor_score += max(0, (humidity - 60) * 1.1)
        if wind is not None:
            odor_score += max(0, (4 - wind) * 8)

        fire_score = 0
        if temp is not None:
            fire_score += max(0, (temp - 30) * 3)
        if humidity is not None:
            fire_score += max(0, (40 - humidity) * 1.2)
        if rain is not None:
            fire_score -= rain * 8

        overflow_score = 0
        if rain is not None:
            overflow_score += rain * 15
        if humidity is not None:
            overflow_score += humidity * 0.3

        risk_items = []
        if has_env_signals:
            risk_items = [
                ("Decomposition Risk", clamp_score(decomp_score), "Heat + humidity accelerating organic breakdown."),
                ("Leachate Risk", clamp_score(leachate_score), "Rain/soil saturation elevating runoff risk."),
                ("Odor Risk", clamp_score(odor_score), "Humidity and low wind trap odors."),
                ("Fire Risk", clamp_score(fire_score), "Heat and dryness increase combustion risk."),
                ("Overflow Risk", clamp_score(overflow_score), "Precipitation loading storage capacity."),
            ]

        def risk_level(score: int) -> str:
            if score >= 80:
                return "SEVERE"
            if score >= 60:
                return "HIGH"
            if score >= 35:
                return "MODERATE"
            return "LOW"

        risks = [
            {"label": label, "score": score, "level": risk_level(score), "note": note}
            for label, score, note in risk_items
        ]

        overall_risk = max([r["score"] for r in risks], default=0)
        overall_level = risk_level(overall_risk)

        hero = []
        add_metric(hero, "Temperature", temp, "°C")
        add_metric(hero, "Humidity", humidity, "%")
        add_metric(hero, "Wind", wind, " m/s")
        if risk_items:
            hero.append({"label": "Waste Risk", "value": overall_level})

        research_items = []
        search_res = kb.get("search_agent") if isinstance(kb.get("search_agent"), dict) else {}
        for item in (search_res.get("report") or [])[:8]:
            research_items.append({
                "title": item.get("title") or "Research",
                "summary": item.get("explain") or "",
                "link": item.get("link"),
                "image": item.get("image"),
            })

        insights = []
        for ins in (kb.get("master_insights") or []):
            insights.append(ins)

        visuals = []
        for ia in (kb.get("image_analyses") or []):
            if isinstance(ia, dict) and (ia.get("image") or ia.get("file_path")):
                visuals.append({
                    "title": ia.get("title") or "Image Analysis",
                    "image": ia.get("image") or ia.get("file_path"),
                    "description": ia.get("explanation") or "",
                })

        diagnostics = []
        env_meta = {k: v for k, v in env_res.items() if k != "environmental_data"}
        for key in sorted(env_meta.keys()):
            diagnostics.append({"label": key, "value": str(env_meta.get(key))})

        for key in sorted(env_data.keys()):
            if key not in used_env_keys and env_data.get(key) is not None:
                diagnostics.append({"label": key, "value": str(env_data.get(key))})

        payload = {
            "header": {
                "system_name": os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi"),
                "subtitle": "Live Environmental Waste Intelligence",
                "location": place,
                "updated_at": updated_at,
                "cache_status": "HIT" if cached_flag else "MISS",
            },
            "hero": hero,
            "atmosphere": atmosphere,
            "soil": soil,
            "weather": weather,
            "risks": risks,
            "insights": insights,
            "research": research_items,
            "visuals": visuals,
            "tables": kb.get("summary_tables") or [],
            "diagnostics": diagnostics,
        }

        if self.d.generate_dashboard(payload):
            self.d.open_dashboard()
            return {"success": True, "file_path": self.d.output_file}
        return {"success": False}