from backend.image.explain import Vision

analyzer = Vision()
path = "3480.webp"
result = analyzer.explain_image(path)
print("\n--- AI Response ---\n")
print(result)
