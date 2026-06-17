from backend.Agents.Master import WasteDispoMaster
from backend.output.tts_and_sst import speak, listen
import os


def is_status_query(text):
    keywords = ["status", "processing", "working", "update"]
    return any(word in text.lower() for word in keywords)


def is_exit_query(text):
    keywords = ["stop", "exit", "bye", "goodbye", "good bye", "shutdown", "quit", "close"]
    return any(word in text.lower() for word in keywords)


def main():
    default_location = "Chittarikkal, Kerala, India"
    system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
    master_name = os.getenv("SUSTAINAI_MASTER_NAME", "Lily")

    print(f"🚀 Starting {system_name} Command Center...")
    
    master = WasteDispoMaster(default_location=default_location)

    welcome_message = (
        f"hello {master_name} is online. "
        "You can initiate autonomous environmental analysis, "
        "request live ecosystem intelligence, "
        "trigger research workflows, "
        "or generate a fully interactive decision support dashboard."
    )

    print(f"\n{master_name}: {welcome_message}")
    speak(welcome_message)

    try:
        while True:
            print("\n🎤 Listening...")
            user_in = listen()

            if not user_in:
                import time
                time.sleep(0.5)  # Prevent rapid retries
                continue

            print(f"\nYou (Voice): {user_in}")

            if is_exit_query(user_in):
                goodbye = f"Alright. {master_name} signing off. Goodbye."
                print(f"\n{master_name}: {goodbye}")
                speak(goodbye)
                break

            if is_status_query(user_in):
                response = master.get_status_update()
                print(f"\n{master_name} (Status): {response}")
                speak(response)
                continue

            print(f"\n🧠 {master_name} processing...")
            response = master.process_input(user_in)

            print(f"\n{master_name}: {response}")
            speak(response)

    finally:
        print("\n🧹 Cleaning session...")
        master.cleanup()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()