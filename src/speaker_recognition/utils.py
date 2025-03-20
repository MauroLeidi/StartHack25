import torch
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize

# Adjust according to the model.
EMBEDDING_DIMS: int = 192

speaker_recognition = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


def get_embeddings_from_files(audio_paths: str | list[str], audios_dir=None):
    if len(audio_paths) == 0:
        return torch.tensor([])
    elif isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    file_path = [Path(path) for path in audio_paths]
    loaded_waveforms = pad_sequence([
        speaker_recognition.load_audio(wav, savedir=audios_dir) for wav in audio_paths
    ], batch_first=True)
    embeddings = speaker_recognition.encode_batch(loaded_waveforms).squeeze(1)
    return normalize(embeddings, p=2, dim=1)


def initialize_faiss_database(preload_audios: bool = True):
    """Initialize the FAISS database used to perform speaker identification.

    Args:
        preload_audios (:obj:`bool`, defaults to `True`):
            Whether to initialize the database with some audios already indexed.
    """
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMS)
    if preload_audios:
        audios_dir = "audios"
        audio_files = ["diego-1.wav", "johannes-1.wav", "gio-1.wav", "mauro-1.wav"]
        embeddings = get_embeddings_from_files(audio_files, audios_dir)
        faiss_index.add(embeddings)

    new_audio = get_embeddings_from_files("diego-2.wav", audios_dir)
    distances, indices = faiss_index.search(new_audio, len(audio_files))

    # Print the most similar audios.
    for i, index in enumerate(indices[0]):
        distance = distances[0][i]
        print(f"{i+1}: {audio_files[index]:<20}Cosine similarity: {distance:.2f}")
