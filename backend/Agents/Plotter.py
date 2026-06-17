from backend.output.Ai_Plotter import AIPlotter


class AIPlotterApp:
    def __init__(self):
        self.plotter = AIPlotter()

    def generate_plot(self, input_text):
        """
        Generate and execute plotting code from given input text.
        
        Args:
            input_text (str): Natural language description of the plot.
            
        Returns:
            dict: Status and message
        """
        file_paths = self.plotter.generate_plots_from_text(input_text)

        if file_paths:
            return {
                "success": True,
                "message": f"Plots created successfully in '{self.plotter.display_folder}'",
                "file_path": file_paths[0],
                "file_paths": file_paths,
            }

        return {
            "success": False,
            "message": "Failed to execute generated plotting code.",
            "file_path": None,
            "file_paths": [],
        }