import io
import os

import noisereduce as nr
import numpy as np
import requests
import soundfile as sf
import torch
import whisper
from dotenv import load_dotenv
from funasr import AutoModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from scipy.signal import butter, lfilter
from df.enhance import enhance, init_df, load_audio, save_audio
import librosa

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def reduce_noise_with_deepfilternet(audio, sr):
    MODEL, DF_STATE, _ = init_df()
    # resample audio to 48kHz
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)    
     # Convert resampled_audio to PyTorch tensor
    tensor_audio = torch.from_numpy(resampled_audio).float().unsqueeze(0)
    # print(tensor_audio.shape)
    enhanced = enhance(MODEL, DF_STATE, tensor_audio)
    # back to numpy array
    enhanced = enhanced.squeeze().numpy()
    # resample back to original rate
    enhanced = librosa.resample(enhanced, orig_sr=48000, target_sr=sr)
    return enhanced


def perform_speaker_diarization():
    """
    Process audio bytes to perform diarization and speech-to-text, returning a dictionary
    with speaker segments and transcriptions.

    Returns:
        dict: Dictionary with segments containing start time, end time, speaker, and text
    """
    # Create BytesIO object from audio bytes

    # Initialize diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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",
    )

    # Apply diarization (using the BytesIO object)
    diarization = pipeline("./wav/noise_reduce_out.wav")

    # Initialize speech recognition model
    model = whisper.load_model("small.en")

    # Transcribe audio (using the same BytesIO object)
    asr_result = model.transcribe("./wav/noise_reduce_out.wav")

    # Process results
    timestamp_texts = get_text_with_timestamp(asr_result)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization)
    final_result = merge_sentence(spk_text)

    # Convert to dictionary format
    result_dict = {"segments": []}

    for seg, spk, text in final_result:
        segment = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": spk,
            "text": text,
        }
        result_dict["segments"].append(segment)

    return result_dict


def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res["segments"]:
        start = item["start"]
        end = item["end"]
        text = item["text"]
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = "".join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = [".", "?", "!"]


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


def extract_speakers_chunks(segments_results, waveform):
    # Create a dictionary to hold the speaker's chunks
    speaker_chunks = {}

    sampling_rate = 16000
    # Process the segments to group them by speaker and chunk them
    for segment in segments_results["segments"]:
        speaker = segment["speaker"]
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]

        # Calculate start and end sample indices based on the sampling rate
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)

        # Extract the chunk from the waveform
        chunk = waveform[start_sample:end_sample]

        # If the speaker is already in the dictionary, append the chunk to their list, else create a new entry
        if speaker in speaker_chunks:
            speaker_chunks[speaker]["texts"].append(text.strip())
            speaker_chunks[speaker]["chunks"].append(chunk)
        else:
            speaker_chunks[speaker] = {
                "texts": [text.strip()],
                "chunks": [chunk],
            }
    return speaker_chunks


def highpass_filter(audio, sr, cutoff=100, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, audio)


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def bandpass_filter(
    audio,
    sr,
    lowcut=300,
    highcut=3400,
    order=5,
):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", analog=False)
    return lfilter(b, a, audio)


def convert_wav_ndarray_to_bytearray(wav_ndarray, sr):
    buffer = io.BytesIO()
    sf.write(buffer, wav_ndarray, sr, format="WAV")
    return buffer.getvalue()


def reduce_noise(audio, sampling_rate, use_deep=False):
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)  # Average left and right channels
    print("Audio shape:", audio.shape)
    # save wav file for debugging as input.wav to /wav
    sf.write("./wav/noise_reduce_input.wav", data=audio, samplerate=sampling_rate)
    # apply highpass_filter, remove low frequencies (humming etc.)
    audio = highpass_filter(audio=audio, sr=sampling_rate)
    print("Audio shape after highpass:", audio.shape)
    # optionally apply bandpass_filter
    # audio = bandpass_filter(audio, sr=sampling_rate)
    # normalize audio, emphasisze voices
    #audio = normalize_audio(audio)
    #print("Audio shape after normalize:", audio.shape)
    # reduce noise
    # if not use_deep:
    enhanced_audio = nr.reduce_noise(
        y=audio, sr=sampling_rate, n_fft=512, prop_decrease=0.9
    )
    # else:
    # enhanced_audio = reduce_noise_with_deepfilternet(audio=audio, sr=sampling_rate)
    print("Audio shape after reduce noise:", enhanced_audio.shape)
    # save wav file for debugging as output.wav to /wav
    sf.write("./wav/noise_reduce_out.wav", enhanced_audio, samplerate=sampling_rate)
    # back to bytes
    return (
        convert_wav_ndarray_to_bytearray(enhanced_audio, sr=sampling_rate),
        enhanced_audio,
    )


def recognise_emotion_speech(wav_file_path):
    """Functions that recognise speech emotion from a given file audio, and returns one of the following values:
    0: angry 1: disgusted 2: fearful 3: happy 4: neutral 5: other 6: sad 7: surprised 8: unknown


    Returns:
        _type_: _description_
    """
    model = AutoModel(model="iic/emotion2vec_plus_base")
    res = model.generate(
        wav_file_path,
        output_dir="./outputs",
        granularity="utterance",
        extract_embedding=False,
    )

    labels = res[0]["labels"]
    scores = res[0]["scores"]

    # Find the label with the maximum score
    index_max_score = scores.index(max(scores))
    max_label = labels[index_max_score]
    return index_max_score, max_label


def call_openai_api(prompt):
    """Helper function to call OpenAI API"""
    URL = "https://api.openai.com/v1/chat/completions"

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating persona: {str(e)}"
