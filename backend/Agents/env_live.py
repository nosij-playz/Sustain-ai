import os
import json
from backend.data_fetch.envdata import get_env_data, get_coords_from_place


class EnvironmentalDataFetcher:
    def __init__(self, dotenv_path=".env"):
        self._load_environment(dotenv_path)

        self.locationiq_key = os.getenv("LOCATIONIQ_KEY")
        self.owm_key = os.getenv("OWM_KEY")
        self.weatherapi_key = os.getenv("WEATHERAPI_KEY")

    def _load_dotenv_fallback(self, dotenv_path):
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

    def _load_environment(self, dotenv_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)
        except Exception:
            self._load_dotenv_fallback(dotenv_path)

    def _validate_keys(self):
        missing = [
            name for name, value in {
                "LOCATIONIQ_KEY": self.locationiq_key,
                "OWM_KEY": self.owm_key,
                "WEATHERAPI_KEY": self.weatherapi_key,
            }.items() if not value
        ]

        if missing:
            return {
                "success": False,
                "error": f"Missing API key(s): {', '.join(missing)}"
            }

        return None

    def fetch_data(self, place):
        if not place:
            return {
                "success": False,
                "error": "No place provided."
            }

        key_check = self._validate_keys()
        if key_check:
            return key_check

        lat, lon = get_coords_from_place(place, self.locationiq_key)

        if lat is None or lon is None:
            return {
                "success": False,
                "error": "Could not retrieve coordinates."
            }

        env_data = get_env_data(
            lat,
            lon,
            self.owm_key,
            self.weatherapi_key
        )

        return {
            "success": True,
            "place": place,
            "coordinates": {
                "latitude": lat,
                "longitude": lon
            },
            "environmental_data": env_data
        }

    def fetch_json(self, place):
        return json.dumps(self.fetch_data(place), indent=4)