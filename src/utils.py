import io
from pyannote.audio import Pipeline
import whisper
from pyannote.core import Segment, Annotation
import numpy as np
import soundfile as sf
import torch

def process_audio_array(waveform):
    """
    Process audio bytes to perform diarization and speech-to-text, returning a dictionary
    with speaker segments and transcriptions.
    
    Args:
        audio_bytes (bytes): Audio data as bytes
    
    Returns:
        dict: Dictionary with segments containing start time, end time, speaker, and text
    """
    # Create BytesIO object from audio bytes
    
    # Initialize diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"
    )
    
    # Apply diarization (using the BytesIO object)
    diarization = pipeline({
        "waveform": torch.tensor(waveform).unsqueeze(0),
        "sample_rate": 16000
    })

    
    # Initialize speech recognition model
    model = whisper.load_model("tiny.en")
    
    # Transcribe audio (using the same BytesIO object)
    asr_result = model.transcribe(waveform)
    
    # Process results
    timestamp_texts = get_text_with_timestamp(asr_result)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization)
    final_result = merge_sentence(spk_text)
    
    # Convert to dictionary format
    result_dict = {
        "segments": []
    }
    
    for seg, spk, text in final_result:
        segment = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": spk,
            "text": text
        }
        result_dict["segments"].append(segment)
    
    return result_dict

def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

PUNC_SENT_END = ['.', '?', '!']

def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text