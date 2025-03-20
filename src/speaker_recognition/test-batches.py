import faiss
from utils import get_embeddings_from_files

audios_dir = "audios"
audio_files = ["diego-1.wav", "johannes-1.wav", "gio-1.wav", "mauro-1.wav"]

embeddings = get_embeddings_from_files(audio_files, audios_dir)

dim = len(embeddings[0])
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings)

new_audio = get_embeddings_from_files("diego-2.wav", audios_dir)
distances, indices = faiss_index.search(new_audio, len(audio_files))

# Print the most similar audios.
for i, index in enumerate(indices[0]):
    distance = distances[0][i]
    print(f"{i+1}: {audio_files[index]:<20}Cosine similarity: {distance:.2f}")
