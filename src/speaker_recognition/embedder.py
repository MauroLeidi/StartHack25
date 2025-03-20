"""Audio embeddings."""

import io
import torch
import torchaudio
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize


class AudioEmbedder:
    """Audio Embedder."""

    def __init__(self, model: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.embedder = SpeakerRecognition.from_hparams(
            source=model,
            savedir=f"pretrained_models/{model.split('/')[-1]}"
        )

    def embed_from_raw(self, audios: bytes | list[bytes]):
        if isinstance(audios, bytes):
            audios = [audios]
        loaded_waveforms = pad_sequence([
            self.load_raw_audio(audio) for audio in audios
        ], batch_first=True)
        return self._loaded_waveforms2embeddings(loaded_waveforms)

    def embed_from_files(self, audio_paths: str | list[str], audios_dir=None):
        if len(audio_paths) == 0:
            return torch.tensor([])
        elif isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        file_path = [Path(path) for path in audio_paths]
        loaded_waveforms = pad_sequence([
            self.embedder.load_audio(wav, savedir=audios_dir) for wav in audio_paths
        ], batch_first=True)
        return self._loaded_waveforms2embeddings(loaded_waveforms)

    def load_raw_audio(self, audio: bytes):
        signal, sr = torchaudio.load(io.BytesIO(audio), channels_first=False)
        return self.embedder.audio_normalizer(signal, sr)

    def _loaded_waveforms2embeddings(self, loaded_waveforms: list):
        embeddings = self.embedder.encode_batch(loaded_waveforms).squeeze(1)
        return self._normalize(embeddings)

    @classmethod
    def _normalize(cls, embeddings):
        return normalize(embeddings, p=2, dim=1)
