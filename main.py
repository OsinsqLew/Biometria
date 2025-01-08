import itertools
import numpy as np
import pandas as pd
from deepface import DeepFace

def euclidean_distance(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return np.linalg.norm(embedding1 - embedding2)

# Lista modeli i backendów do porównania
models = ["VGG-Face", "Facenet", "OpenFace", "ArcFace"]
detector_backends = ["opencv", "mtcnn", "retinaface"]

photos = ["photos/photo-12-1.jpeg", "photos/photo-12-3.jpeg"]
photo_embd = {}

# Lista wyników
results = []

# Porównanie wyników dla różnych modeli i backendów
for model in models:
    for backend in detector_backends:
        print(f"\nUsing Model: {model}, Detector Backend: {backend}")
        
        photo_embd = {}
        
        try:
            # Generowanie embeddingów dla zdjęć
            for photo in photos:
                photo_embd[photo] = DeepFace.represent(
                    photo, 
                    model_name=model, 
                    detector_backend=backend,
                    enforce_detection=False  # W przypadku problemów z detekcją
                )[0]["embedding"]
            
            # Porównanie par zdjęć
            pairs = itertools.combinations(photos, 2)
            for photo1, photo2 in pairs:
                cosine_sim = np.dot(photo_embd[photo1], photo_embd[photo2]) / (
                    np.linalg.norm(photo_embd[photo1]) * np.linalg.norm(photo_embd[photo2])
                )
                euc_dist = euclidean_distance(photo_embd[photo1], photo_embd[photo2])
                
                # Zapisanie wyników do tabeli
                results.append({
                    "Model": model,
                    "Algorithm": backend,
                    "Photo 1": photo1,
                    "Photo 2": photo2,
                    "Cos Sim": cosine_sim,
                    "Euci": euc_dist
                })
        except Exception as e:
            print(f"Error with model {model} and backend {backend}: {e}")

# Tworzenie DataFrame z wynikami
df = pd.DataFrame(results)

# Zapisanie wyników do pliku CSV
df.to_csv("reverse-60.csv", index=False)


print("\n--- Results saved to csv file ---")
