from backend.output.tts_and_sst import speak, listen

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main():
    # Greeting
    speak("Hi! I'm online. What would you like to talk about?")

    while True:
        user_input = listen()

        if user_input:
            # Check for exit keywords
            if any(word in user_input.lower() for word in ["stop", "exit", "goodbye", "bye"]):
                speak("Alright, I'm heading out. See you later!")
                break
            
            # SIMPLE LOGIC: This is where you would normally connect an AI like GPT.
            # For now, it just reflects your input.
            response = f"You said {user_input}. That's really interesting, tell me more!"
            speak(response)

if __name__ == "__main__":
    main()

