from backend.Agents.Plotter import AIPlotterApp

plot_app = AIPlotterApp()

user_query = "Create a bar chart for plastic waste growth: 1990-10M, 2000-20M, 2010-40M"

result = plot_app.generate_plot(user_query)

print(result["message"])

if result.get("file_path"):
	print(f"Saved plot: {result['file_path']}")