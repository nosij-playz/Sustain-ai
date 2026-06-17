import ollama
import os
import re

class Vision:
    def __init__(self, model_name="gemma4:31b-cloud"):
        """
        Initializes the Vision Analyzer with a specific model.
        """
        self.model_name = "gemma4:31b-cloud"

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

    def explain_image(self, image_path, prompt=None):
        """
        Passes an image path and returns the model's response as a string.
        """
        # 1. Verify if the image file exists
        if not os.path.exists(image_path):
            return f"Error: The file at {image_path} was not found."

        # Default prompt if none is provided
        if prompt is None:
            prompt = "Please explain this image in detail. Be concise. What is happening, and what are the key objects or people visible?"

        try:
            # 2. Call the Ollama chat interface
            response = ollama.chat(
                model=self.model_name, 
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path] 
                    },
                ],
            )
            
            # 3. Clean and return the text content
            response_text = response['message']['content']
            return self._clean_response_text(response_text)

        except Exception as e:
            return f"An error occurred while processing the image: {str(e)}"