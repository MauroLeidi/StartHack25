import io
import json
import os
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import streamlit as st
from backend import create_socket

# Create wav directory if it doesn't exist
os.makedirs("./wav", exist_ok=True)


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_chunks = []
        self.recording_thread = None
        self.socket = None
        self.accumulated_audio = None
        self.sample_rate = 8000  # Changed to 8000Hz

    def create_wav_data(self, audio_data):
        """Create WAV data with proper headers"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        return wav_buffer.getvalue()

    def record_chunks(self):
        is_first_chunk = True  # Track if this is the first chunk
        while self.recording:
            # Record for 0.5 seconds
            chunk = sd.rec(
                int(self.sample_rate * 0.5),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
            )
            sd.wait()
            self.audio_chunks.append(chunk)

            # Accumulate chunks
            if len(self.audio_chunks) > 0:
                self.accumulated_audio = np.concatenate(self.audio_chunks)

                # Send data - first chunk with WAV headers, subsequent chunks as raw bytes
                if self.socket:
                    try:
                        if is_first_chunk:
                            # First chunk with WAV headers
                            wav_data = self.create_wav_data(self.accumulated_audio)
                            self.socket.send(wav_data)
                            is_first_chunk = False
                        else:
                            # Subsequent chunks as raw bytes
                            raw_data = chunk.tobytes()
                            self.socket.send(raw_data)
                    except Exception as e:
                        print(f"Error sending audio: {e}")

            time.sleep(0.01)  # Small delay to prevent high CPU usage

    def start_recording(self):
        try:
            self.recording = True
            self.audio_chunks = []
            self.accumulated_audio = None
            # Create and connect socket
            self.socket = create_socket()
            self.recording_thread = threading.Thread(
                target=self.record_chunks, daemon=True
            )
            self.recording_thread.start()
        except Exception as e:
            self.recording = False
            st.error(f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)

        # Send the final accumulated audio
        if self.socket and self.accumulated_audio is not None:
            try:
                wav_data = self.create_wav_data(self.accumulated_audio)
                self.socket.send(wav_data)
                print("Sent final audio data")
            except Exception as e:
                st.error(f"Error sending final audio: {str(e)}")

        # Close session and get final transcription
        final_text = None
        if self.socket:
            try:
                final_text = self.socket.close_session()
            except Exception as e:
                st.error(f"Error closing session: {str(e)}")

        return self.accumulated_audio, final_text


# Initialize the recorder
if "recorder" not in st.session_state:
    st.session_state.recorder = AudioRecorder()
    st.session_state.button_state = False
    st.session_state.audio_bytes = None
    st.session_state.final_text = None

st.title("Audio Recorder")

st.subheader("Recording")
if st.button("Record/Stop", key="record_button"):
    st.session_state.button_state = not st.session_state.button_state

    if st.session_state.button_state:
        # Start recording
        st.session_state.recorder.start_recording()
        st.warning("Recording... Click button again to stop.")
    else:
        # Stop recording and get final transcription
        final_audio, final_text = st.session_state.recorder.stop_recording()
        if final_audio is not None:
            st.success("Recording stopped!")

            # Create WAV data for playback
            wav_data = st.session_state.recorder.create_wav_data(final_audio)
            st.session_state.audio_bytes = wav_data
            st.audio(st.session_state.audio_bytes, format="audio/wav")

            # Show transcription if available
            if final_text:
                st.session_state.final_text = final_text
                st.info(f"Transcription: {final_text}")
