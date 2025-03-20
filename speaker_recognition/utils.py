import torch
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize

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
