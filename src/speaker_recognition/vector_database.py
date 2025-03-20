"""Vector database powered by FAISS.

See `https://github.com/facebookresearch/faiss`__.
"""

import torch
import faiss
from .embedder import AudioEmbedder

# Adjust according to the model.
SIMILARITY_THRESHOLD: float = 0.55
EMBEDDING_DIMS: int = 192

SPEAKER_ID_PREFIX: str = "speaker"

class VectorDB:
    
    def __init__(
        self,
        dims: int = EMBEDDING_DIMS,
        similarity_threshold = SIMILARITY_THRESHOLD,
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
            audio_files = ["diego-1.wav", "johannes-1.wav", "gio-1.wav", "mauro-1.wav"]
            embeddings = self.embedder.embed_from_files(audio_files, audios_dir)
            self.index.add(embeddings)
            _ = [self._add_speaker() for _ in audio_files]

    def classify_speaker(self, speaker_embeddings: torch.Tensor):
        similarities, indices = self.index.search(speaker_embeddings, 1)
        print("SIMS:", similarities)
        print("INDICES:", indices)
        # if sim >= self.similarity_threshold:
        #     print("SPEAKER:", self.speakers[idx])

    def _add_speaker(self) -> None:
        if len(self.speakers) == 0:
            self.speakers.append(f"{SPEAKER_ID_PREFIX}-00")
        else:
            last_speaker_count: int = int(self.speakers[-1].split("-")[-1])
            self.speakers.append(f"{SPEAKER_ID_PREFIX}-{last_speaker_count+1:02d}")
