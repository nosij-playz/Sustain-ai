from backend.Agents.env_live import EnvironmentalDataFetcher

fetcher = EnvironmentalDataFetcher()
result = fetcher.fetch_data("Chittarikkal, Kerala, India")
print(result)