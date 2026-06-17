from backend.Agents.Master import WasteDispoMaster
import os

if __name__ == "__main__":
    default_location = "Chittarikkal, Kerala, India"
    system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
    master_name = os.getenv("SUSTAINAI_MASTER_NAME", "Lily")

    print(f"🚀 Starting {system_name} Command Center...")

    upload_image = ["disposable-garbage-bags-613.jpg","3480.webp"]
    location = default_location
    master = WasteDispoMaster(default_location=location)

    upload_summaries = master._analyze_image_list(upload_image)
    if upload_summaries:
        print(f"\n{master_name} (Image Analysis):")
        for summary in upload_summaries:
            print(f"- {summary}")

    print(
        f"{master_name} is Online. "
        "You can initiate autonomous environmental analysis, request live ecosystem intelligence, "
        "trigger research workflows, or generate a fully interactive decision-support dashboard."
    )

    try:
        while True:
            user_in = input("\nYou: ")
            if user_in.lower() == "exit":
                break

            if any(word in user_in.lower() for word in ["status", "processing", "working", "update"]):
                print(f"\n{master_name} (Status): {master.get_status_update()}")
            else:
                print(f"\n{master_name}: {master.process_input(user_in)}")
    finally:
        master.cleanup()
