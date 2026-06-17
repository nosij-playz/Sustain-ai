from backend.display.Dashboard import DashboardModule
    # Sample JSON input as requested
sample_data = [
        {
            "title": "Plastic Waste in Oceans",
            "content": "Over 8 million tons of plastic enter the ocean every year, affecting marine life.",
            "link": "https://www.nationalgeographic.com",
            "image": "https://via.placeholder.com/150"
        },
        {
            "title": "Soil Carbon Sequestration",
            "content": "Improving soil health can trap gigatons of carbon, helping fight climate change.",
            "link": None,
            "image": None
        },
        {
            "title": "The Danger of PVC",
            "content": "Polyvinyl Chloride releases toxic dioxins during production and disposal.",
            "link": "https://www.epa.gov",
            "image": "https://via.placeholder.com/150"
        }
    ]

dash = DashboardModule()
    
if dash.generate_dashboard(sample_data):
    dash.open_dashboard()
else:
    print("Failed to generate dashboard.")
