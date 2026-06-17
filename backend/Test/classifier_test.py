import os
import sys

# Ensure project root is on sys.path so Tests work regardless of cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.Agents.classifier_agent import ClassifierAgent


if __name__ == '__main__':
    agent = ClassifierAgent()
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3480.webp"))
    res = agent.run({"image_path": image_path, "conf": 0.3})
    print(res)
