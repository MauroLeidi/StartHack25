"""Audio embeddings."""

import io
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence


class AudioEmbedder:
    """Audio Embedder."""

    def __init__(self, model: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.embedder = SpeakerRecognition.from_hparams(
            source=model, savedir=f"pretrained_models/{model.split('/')[-1]}"
        )

    def embed_from_raw(self, audios: bytes | list[bytes]):
        if isinstance(audios, bytes):
            audios = [audios]
        loaded_waveforms = [torch.from_numpy(audio) for audio in audios]
        return self._loaded_waveforms2embeddings(loaded_waveforms)

    def embed_from_files(self, noised_reduced):
        loaded_waveforms = [torch.from_numpy(nr) for nr in noised_reduced]
        return self._loaded_waveforms2embeddings(loaded_waveforms)

    def load_raw_audio(self, audio: bytes):
        signal, sr = torchaudio.load(io.BytesIO(audio), channels_first=False)
        return self.embedder.audio_normalizer(signal, sr)

    def _loaded_waveforms2embeddings(self, loaded_waveforms: list):
        embeds = []
        for waveform in loaded_waveforms:
            embeds.append(self.embedder.encode_batch(waveform).squeeze(1))
        embeds = torch.stack(embeds).squeeze(1)
        return self._normalize(embeds)

    @classmethod
    def _normalize(cls, embeddings):
        return normalize(embeddings, p=2, dim=1)
