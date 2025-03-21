"""Vector database powered by FAISS.

See `https://github.com/facebookresearch/faiss`__.
"""

import sys
from collections import Counter
from pathlib import Path

import faiss
import soundfile as sf
import torch

from .embedder import AudioEmbedder

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import reduce_noise

# Adjust according to the model.
SIMILARITY_THRESHOLD: float = 0.0
EMBEDDING_DIMS: int = 192

SPEAKER_ID_PREFIX: str = "speaker"


class VectorDB:

    def __init__(
        self,
        dims: int = EMBEDDING_DIMS,
        similarity_threshold=SIMILARITY_THRESHOLD,
        preload_audios: bool = True,
    ):
        """Initialize the FAISS database used to perform speaker identification.

        Args:
            dims (:obj:`int`, defaults to `EMBEDDING_DIMS`):
                The dimensions of the embeddings that will be indexed.
            preload_audios (:obj:`bool`, defaults to `True`):
                Whether to initialize the database with some audios already indexed.
        """
        self.dims = dims
        self.similarity_threshold = similarity_threshold
        self.index = faiss.IndexFlatIP(self.dims)
        self.embedder = AudioEmbedder()
        self.speakers: list[str] = []
        if preload_audios:
            audios_dir = "src/speaker_recognition/audios"
            audio_files = ["diego-4.wav", "mauro-2.wav"]
            file_paths = [Path(f"{audios_dir}/{path}") for path in audio_files]
            file_numpy = [sf.read(file) for file in file_paths]
            noised_reduced = [reduce_noise(*audio)[1] for audio in file_numpy]
            embeddings = self.embedder.embed_from_files(noised_reduced)
            self.index.add(embeddings)
            _ = [self.add_speaker() for _ in audio_files]

    def classify_speaker(self, speaker_embeddings: torch.Tensor) -> list[str]:
        similarities, indices = self.index.search(speaker_embeddings, 4)
        print("SIMS:", similarities)
        print("INDICES:", indices)
        potential_speakers: list[tuple[str, float]] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= self.similarity_threshold:
                potential_speakers.append((self.speakers[idx], sim))
        print("POTENTIAL SPEAKERS:", potential_speakers)
        if potential_speakers:
            # Order by descending similarity.
            potential_speakers = sorted(
                potential_speakers, key=lambda x: x[1], reverse=True
            )
            return [x[0] for x in potential_speakers]
        return [self.add_speaker()]

    def add_speaker(self) -> None:
        if not self.speakers:
            speaker = f"{SPEAKER_ID_PREFIX}-00"
        else:
            last_speaker_count: int = int(self.speakers[-1].split("-")[-1])
            speaker = f"{SPEAKER_ID_PREFIX}-{last_speaker_count+1:02d}"
        self.speakers.append(speaker)
        return speaker
