import re
import time
from ollama import chat


class TechnicalTextSummarizer:
    def __init__(
        self,
        model="gpt-oss:120b-cloud",
        chunk_size=10000,
        max_workers=1,
        final_summary_words=1500,
        retry_attempts=3,
        retry_delay=5
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.final_summary_words = final_summary_words
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def ingest_text(self, text):
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if not text.strip():
            raise ValueError("Input text is empty")

        return text

    def preprocess_text(self, text):
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t+", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(.)\1{15,}", r"\1", text)
        return text.strip()

    def split_into_chunks(self, text):
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                for i in range(0, len(paragraph), self.chunk_size):
                    chunks.append(paragraph[i:i + self.chunk_size])

                continue

            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def call_model(self, prompt):
        for attempt in range(self.retry_attempts):
            try:
                response = chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an elite technical synthesis engine. "
                                "Study technical input deeply. "
                                "Preserve analytical relationships, engineering logic, observations, "
                                "metrics, causal dependencies, conceptual findings, technical explanations, "
                                "architecture details, and research insights. "
                                "Remove redundancy while maximizing technical information density."
                                "Generate a structured technical summary that captures the essence of the input with precision and depth in Natural English."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                return response["message"]["content"]

            except Exception as error:
                if attempt == self.retry_attempts - 1:
                    raise error

                print(f"Retrying model call {attempt + 1}/{self.retry_attempts}")
                time.sleep(self.retry_delay)

    def summarize_chunk(self, chunk, chunk_number):
        prompt = (
            f"This is chunk {chunk_number} from a very large technical dataset.\n\n"
            f"Analyze this content deeply.\n"
            f"Extract technical insights, analytical findings, relationships, explanations, "
            f"important details, patterns, observations, and architecture knowledge.\n"
            f"Create a dense structured technical summary preserving maximum information.\n\n"
            f"{chunk}"
        )

        return self.call_model(prompt)

    def summarize_all_chunks(self, chunks):
        summaries = []

        for index, chunk in enumerate(chunks):
            print(f"Processing chunk {index + 1} of {len(chunks)}")
            summary = self.summarize_chunk(chunk, index + 1)
            summaries.append(summary)

        return "\n\n".join(summaries)

    def create_final_summary(self, combined_summary):
        prompt = (
            f"The following content contains multiple technical summaries derived from a very large dataset.\n\n"
            f"Merge all information intelligently.\n"
            f"Remove duplicate information.\n"
            f"Preserve technical depth.\n"
            f"Preserve analytical relationships.\n"
            f"Maintain explanation quality.\n"
            f"Generate one coherent technical summary of approximately {self.final_summary_words} words.\n\n"
            f"{combined_summary}"
        )

        return self.call_model(prompt)

    def summarize(self, raw_text):
        print("Validating input")
        text = self.ingest_text(raw_text)

        print("Preprocessing text")
        cleaned_text = self.preprocess_text(text)

        print("Splitting into chunks")
        chunks = self.split_into_chunks(cleaned_text)

        print(f"Total chunks created: {len(chunks)}")

        print("Starting chunk analysis")
        chunk_summaries = self.summarize_all_chunks(chunks)

        print("Generating final summary")
        final_summary = self.create_final_summary(chunk_summaries)

        return final_summary


