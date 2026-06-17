import requests
from concurrent.futures import ThreadPoolExecutor


def safe_get(url, params=None, headers=None, timeout=15):
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()

        if "application/json" in response.headers.get("Content-Type", ""):
            return response.json()

        return {}

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return {}


def first_valid(lst):
    if not isinstance(lst, list):
        return None

    for value in lst:
        if value is not None:
            return value

    return None


def get_coords_from_place(place_name, locationiq_key):
    """
    Convert place name to latitude/longitude using LocationIQ
    """
    print(f"📍 Fetching coordinates for: {place_name}...")

    url = "https://api.locationiq.com/v1/search"

    params = {
        "key": locationiq_key,
        "q": place_name,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "EnvDataBot/1.0"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()
        results = response.json()

        if results and isinstance(results, list):
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])

            print(f"✅ Found coordinates: {lat}, {lon}")
            return lat, lon

        print("❌ Place not found.")
        return None, None

    except Exception as e:
        print(f"❌ Geocoding failed: {e}")
        return None, None


def get_env_data(lat, lon, owm_key, weatherapi_key):
    data = {}

    hourly_vars = [
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm",
        "surface_pressure",
        "snowfall",
        "snow_depth",
        "shortwave_radiation",
        "temperature_2m",
        "precipitation",
        "windspeed_10m",
        "windgusts_10m"
    ]

    def fetch_openweather():
        print("🌦 Fetching OpenWeatherMap data...")
        return safe_get(
            "https://api.openweathermap.org/data/2.5/weather",
            {
                "lat": lat,
                "lon": lon,
                "units": "metric",
                "appid": owm_key
            }
        )

    def fetch_weatherapi():
        print("🌤 Fetching WeatherAPI data...")
        return safe_get(
            "https://api.weatherapi.com/v1/current.json",
            {
                "key": weatherapi_key,
                "q": f"{lat},{lon}"
            }
        )

    def fetch_openmeteo():
        print("🌍 Fetching Open-Meteo data...")
        return safe_get(
            "https://api.open-meteo.com/v1/ecmwf",
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(hourly_vars),
                "current_weather": True,
                "timezone": "auto"
            }
        )

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_owm = ex.submit(fetch_openweather)
        fut_wapi = ex.submit(fetch_weatherapi)
        fut_om = ex.submit(fetch_openmeteo)

        res_owm = fut_owm.result()
        res_wapi = fut_wapi.result()
        res_om = fut_om.result()

    if res_owm:
        main = res_owm.get("main", {})
        wind = res_owm.get("wind", {})
        clouds = res_owm.get("clouds", {})
        rain = res_owm.get("rain", {})

        data.update({
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),
            "pressure": main.get("pressure"),
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "wind_deg": wind.get("deg"),
            "wind_gust": wind.get("gust"),
            "cloud_coverage": clouds.get("all"),
            "rain_1h": rain.get("1h", 0),
            "rain_3h": rain.get("3h", 0),
            "visibility": res_owm.get("visibility")
        })

    if res_wapi:
        current = res_wapi.get("current", {})

        data.update({
            "uv_index": current.get("uv"),
            "visibility_km": current.get("vis_km"),
            "wind_dir": current.get("wind_dir"),
            "wind_kph": current.get("wind_kph"),
            "gust_kph": current.get("gust_kph"),
            "precip_mm": current.get("precip_mm")
        })

    if res_om:
        current_weather = res_om.get("current_weather", {})
        hourly = res_om.get("hourly", {})

        data["temperature_openmeteo"] = current_weather.get("temperature")
        data["windspeed_openmeteo"] = current_weather.get("windspeed")

        for var in hourly_vars:
            data[var] = first_valid(hourly.get(var, []))

    return data


