<h1 align="center">
  <br>
  StartHack25 Speaker Recognition and Transcription System
  <br>
</h1>

<h4 align="center">An advanced voice recognition system for multi-speaker environments with contextual memory</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#components">Components</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#contributing">Contributing</a>
</p>

## Key Features

* **Multi-Speaker Recognition** - Identifies different speakers in a conversation using voice embeddings
* **Single-Message Multi-Speaker Compatibility** - Each message can contain voice from multiple clients, the voices and content will be separated and treated accordingly.
* **Noise Reduction** - Advanced audio processing to isolate main voices from background noise
* **Real-Time Transcription** - Live speech-to-text conversion as audio is being recorded
* **Speaker Diarization** - Separates and labels audio by individual speakers
* **Contextual Memory** - Maintains conversation history and preferences for each identified speaker
* **Privacy-Focused** - Built with data security and user privacy as core principles


## Components

The system architecture consists of multiple specialized modules:

![image](https://github.com/user-attachments/assets/de6b26a1-9dc3-4013-9260-e546dec7405c)

### Speaker Recognition Module

Leverages SpeechBrain's ECAPA-VOXCELEB model to generate unique voice embeddings and a FAISS vector database for fast speaker matching.

### Audio Processing Pipeline

```
Raw Audio → Noise Reduction → High-pass Filtering → Normalization → Speaker Diarization → Transcription
```

### Memory Management System

Stores and retrieves conversation context for personalized interactions with returning users.

### API Server

Provides REST and WebSocket endpoints for real-time audio processing and transcription.

## Project Structure

```
└── StartHack25
    └── src
        ├── speaker_recognition
        │   ├── __init__.py
        │   ├── embedder.py          # Generate voice embeddings
        │   ├── test-batches.py
        │   ├── utils.py
        │   └── vector_database.py   # FAISS vector database for speaker matching
        ├── frontend.py              # Streamlit web interface
        ├── memoryprocessing.py      # Conversation memory management
        ├── openai_api_example.py
        ├── relay.py                 # Main API server
        ├── utils.py
        └── websocket_adapter.py     # WebSocket client for real-time audio streaming
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/StartHack25.git
cd StartHack25

# Option 1: Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Option 2: Using Poetry
pipx install poetry
poetry install
poetry shell

# Set up environment variables
# Create a .env file with the following variables:
# OPENAI_API_KEY=your_openai_api_key
# AZURE_SPEECH_KEY=your_azure_speech_key
# AZURE_SPEECH_REGION=your_azure_region
```

### Model Download

The required speech recognition models will download automatically on first run, but you can pre-download them with:

```bash
python -c "from speechbrain.inference.speaker import SpeakerRecognition; SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec-ecapa-voxceleb')"
```

## Usage

### Starting the System

1. In a web browser, not Safari, start the web interface: https://voiceoasis.azurewebsites.net/


2. Launch the API server:
```bash
cd src
python relay.py
```

3. Start the interaction, by speaking with the virtual waiter.


### Using the Web Interface

1. Click the microphone button to start recording
2. Speak clearly for best recognition results. Note that if you are not in our Database you will get a a new profile, but we only store the embedding, for provacy considerations.
3. The system will:
   - Identify how many people and who is speaking, separate the content based on the speaker
   - Transcribe speech in real-time
   - Maintain context from previous interactions
4. Click the microphone button again to stop recording

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chats/<chat_session_id>/sessions` | POST | Create a new audio session |
| `/chats/<chat_session_id>/sessions/<session_id>/wav` | POST | Upload an audio chunk |
| `/chats/<chat_session_id>/sessions/<session_id>` | DELETE | Close a session |
| `/chats/<chat_session_id>/set-memories` | POST | Store conversation memory |
| `/chats/<chat_session_id>/get-memories` | GET | Retrieve conversation memory |

### API Documentation

For complete API documentation:
1. Run the relay server: `python ./src/relay.py`
2. Open [API Docs](http://localhost:5000/apidocs/) in your browser

## Use Cases

### Restaurant Digital Waiter

The system can identify multiple customers at a table, understand their orders, and remember their preferences from previous visits.

### Meeting Transcription

Accurately capture and attribute statements in multi-person meetings, **even if multiple clients are speaking in the same message**, generating speaker-labeled transcripts.

### Smart Home Assistant

Recognize different family members and provide personalized responses based on individual preferences and history.

## Performance Considerations

- Audio processing is optimized to run in real-time (< 0.5s per sample) on consumer hardware
- For larger deployments, consider scaling the vector database with a distributed FAISS implementation
- GPU acceleration significantly improves speaker embedding generation performance

## Troubleshooting

### Common Issues

- **Microphone not detected**: Ensure browser permissions are granted
- **WebSocket connection fails**: Check if the relay server is running and accessible

### Logs

Check the terminal running the relay server for detailed logs and error messages.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- SpeechBrain team for the ECAPA-VOXCELEB models
- OpenAI for Whisper transcription technology
- StartHack25 organizers and mentors
