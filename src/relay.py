import numpy as np
from typing import Any, Dict
#from df.enhance import enhance, init_df, load_audio, save_audio
import uuid
import json
#import torch
from scipy.io import wavfile
import os
from dotenv import load_dotenv
import numpy as np
from flask import Flask, request, jsonify
#import azure.cognitiveservices.speech as speechsdk
from flask_cors import CORS
from flasgger import Swagger
from flask_sock import Sock
from scipy.io.wavfile import write
import soundfile as sf
from openai import OpenAI
import io
import librosa
from utils import extract_speakers_chunks, reduce_noise, perform_speaker_diarization
import torch
from speaker_recognition.vector_database import VectorDB
from speaker_recognition.embedder import AudioEmbedder
from memoryprocessing import load_or_create_summary_persona

# torch only use cpu

if torch.cuda.is_available():
    device = torch.device("cuda")
device = torch.device("cpu")

# Load environment variables
load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

MEMORY_FILE = "memories.json"

sessions = {}

# Vector database used to store the audio embeddings to perform speaker
# identification.
db = VectorDB(preload_audios=True)
audio_embedder = AudioEmbedder()
messages = {}

#MODEL, DF_STATE, _ = init_df()

# def reduce_noise_with_deepfilternet(audio, sr):
#   # resample audio to 48kHz
#   resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)    
#   # Convert resampled_audio to PyTorch tensor
#   tensor_audio = torch.from_numpy(resampled_audio).float().unsqueeze(0)
#   # print(tensor_audio.shape)
#   enhanced = enhance(MODEL, DF_STATE, tensor_audio)
#   # back to numpy array
#   enhanced = enhanced.squeeze().numpy()
#   # resample back to original rate
#   enhanced = librosa.resample(enhanced, orig_sr=48000, target_sr=sr)
#   return enhanced

menu = """
Menu:
Starters:
1. Fried Shrimp with Avocado Tartare - $12.00
2. Gazpacho Soup with Basil and Ginger - $8.50
3. Zucchini and Burrata with Cherry Tomatoes - $9.00
Main Courses:
1. Grilled Salmon with Lemon Butter Sauce - $22.00
2. Herb-Crusted Chicken with Roasted Vegetables - $18.50
3. Vegan Stir-fried Tofu with Vegetables and Rice - $14.00
4. Classic Beef Burger with Fries - $15.00
5. Spaghetti Aglio e Olio with Chili Flakes - $13.00
Desserts:
1. Chocolate Lava Cake - $7.00
2. Lemon Sorbet - $5.00
3. Tiramisu - $6.00
Drinks:
1. Sparkling Water - $3.50
2. Fresh Lemonade - $4.00
3. Red Wine (Glass) - $7.00
4. White Wine (Glass) - $7.00
5. Espresso - $2.50
6. Iced Tea - $3.00
Beer:
1. Lager - $5.00
2. Pale Ale - $5.50
3. IPA - $6.00
4. Stout - $6.00
"""


def convert_wav_ndarray_to_bytearray(wav_ndarray, sr):
    buffer = io.BytesIO()
    sf.write(buffer, wav_ndarray, sr, format='WAV')
    return buffer.getvalue()


def transcribe_whisper(audio_recording):
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = 'audio.wav'  # Whisper requires a filename with a valid extension
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        #language = ""  # specify Language explicitly
    )
    print(f"openai transcription: {transcription.text}")
    return transcription.text
    

def transcribe_preview(session):
    if session["audio_buffer"] is not None:
        text = transcribe_whisper(session["audio_buffer"])
        # send transcription
        ws = session.get("websocket")
        if ws:
            message = {
                "event": "recognizing",
                "text": text,
                "language": session["language"]
            }
            ws.send(json.dumps(message))


def load_wav_file():
  file_path = "./wav/input.wav"
  audio, sampling_rate = sf.read(file_path)
  return audio, sampling_rate


def perform_speaker_identification(diarization: Dict[Any, Any], audio:Any):
  return None

def load_memories():
    """Load stored memories from the file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memories(memories):
    """Save memories to the file."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=4)

@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """
    Open a new voice input session and start continuous recognition.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - language
          properties:
            language:
              type: string
              description: Language code for speech recognition (e.g., en-US)
    responses:
      200:
        description: Session created successfully
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Unique identifier for the voice recognition session
      400:
        description: Language parameter missing
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    session_id = str(uuid.uuid4())

    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    language = body["language"]

    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": language,
        "websocket": None  # will be set when the client connects via WS
    }

    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk received successfully
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    audio_data = request.get_data()  # raw binary data from the POST body
    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] = sessions[session_id]["audio_buffer"] + audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data
    transcribe_preview(sessions[session_id])
    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """
    Close the session (stop recognition, close push stream, cleanup).
    
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to close
    responses:
      200:
        description: Session successfully closed
        schema:
          type: object
          properties:
            status:
              type: string
              example: session_closed
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: Session not found
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    if sessions[session_id]["audio_buffer"] is not None:
        # TODO: OUR NEW FLOW, WHATEVER IS DONE HERE IS NOT REALTIME YET
        messages[chat_session_id] = []
        # NOISE REDUCTION
        # -------------------
        with io.BytesIO(sessions[session_id]["audio_buffer"]) as buffer:
          audio, sampling_rate = sf.read(buffer)
          # audio, sampling_rate = sf.read("src/speaker_recognition/audios/diego-1.wav")
        sessions[session_id]["audio_buffer"], enhanced_audio = reduce_noise(audio, sampling_rate)

        # SPEAKER DIARIZATION
        # -------------------
        diarization = perform_speaker_diarization()
        speaker_chunks = extract_speakers_chunks(diarization, enhanced_audio)

        # SPEAKER IDENTIFICATION
        # -------------------
        print("SPEAKER CHUNKS:", diarization)
        print("NUM SPEAKERS:", len(speaker_chunks))

        audio_messages = []
        for chunks in speaker_chunks.values():
          print("CHUNKSSSSS:", chunks['chunks'])
          speaker_embeddings = audio_embedder.embed_from_raw(chunks['chunks'])
          speaker_id = db.classify_speaker(speaker_embeddings)
          audio_messages.append(
            {
              'speaker_id': speaker_id,
              'message': '. '.join(chunks['texts']),
            }
          )
        messages[chat_session_id].append(audio_messages)
        print("MESSAGES:", messages)
        #current_speaker = perform_speaker_identification(diarization, sessions[session_id]["audio_buffer"])
        # save audiofile as wav for debug
        # with open(f"audio_{session_id}.wav", "wb") as f:
        #     f.write(sessions[session_id]["audio_buffer"])
        text = transcribe_whisper(sessions[session_id]["audio_buffer"])
        # send transcription
        ws = sessions[session_id].get("websocket")
        if ws:
          message = {
              "event": "recognized",
              "text": text,
              "language": sessions[session_id]["language"]
          }
          ws.send(json.dumps(message))
    # # Remove from session store
    sessions.pop(session_id, None)
    return jsonify({"status": "session_closed"})


@sock.route(path="/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """
    WebSocket endpoint for clients to receive STT results.

    This WebSocket allows clients to connect and receive speech-to-text (STT) results
    in real time. The connection is maintained until the client disconnects. If the 
    session ID is invalid, an error message is sent, and the connection is closed.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the chat session.
      - name: session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the speech session.
    responses:
      400:
        description: Session not found.
      101:
        description: WebSocket connection established.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    # Store the websocket reference in the session
    sessions[session_id]["websocket"] = ws

    # Keep the socket open to send events
    # Typically we'd read messages from the client in a loop if needed
    while True:
        # If the client closes the socket, an exception is thrown or `ws.receive()` returns None
        msg = ws.receive()
        if msg is None:
            break


@app.route('/chats/<chat_session_id>/set-memories', methods=['POST'])
def set_memories(chat_session_id):
    """
    Set memories for a specific chat session.

    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            chat_history:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The chat message text.
              description: List of chat messages in the session.
    responses:
      200:
        description: Memory set successfully.
        schema:
          type: object
          properties:
            success:
              type: string
              example: "1"
      400:
        description: Invalid request data.
    """
    chat_history = request.get_json()
    
    messages_robot = [msg['text'] for idx, msg in enumerate(chat_history) if idx % 2 == 1]

    # Iterate through robot messages and append them as individual turns
    for i, _ in enumerate(messages_robot):
      robot_turn = [
          {
              "speaker_id": "robot",
              "message": messages_robot[i],
          }
      ]
      messages[chat_session_id].append(robot_turn)

    unique = set(msg["speaker_id"] for turn in messages[chat_session_id] for msg in turn)
    unique.discard("robot")
    numspeakers = len(unique)

    # TODO preprocess data (chat history & system message)
    # Generate a prompt using 1. the messages and 2. who is saying what 3. A general description of the past interactions with the given person
    # messages are alternating (robot,person,robot,person,robot,person ...)
    # Generate the structured prompt
    memory = f"""You are a waiter currently serving a table with {numspeakers} different clients. 
    Here is the conversation so far:
    """

    for turn in messages[chat_session_id]:
      for message in turn:
        if message["speaker_id"] == 'robot':
          speaker_role = 'You (Waiter)'
        else:
          speaker_role = f"Client {message['speaker_id'].split('-')[1]}"
        memory += f"{speaker_role} said: {message['message']}\n"

    # we have to check if we stored personal information about the id of the last speaker
    # summaries = load_or_create_summary_persona()
    # Ensure summaries DataFrame is not empty before checking
    # if not summaries.empty and ids[-1] in summaries['user_id'].values:
      # Retrieve the description
    #   description = summaries.loc[summaries['user_id'] == ids[-1], 'summary_persona'].iloc[0]

      # Append it to the prompt
    #   memory += f" A short description of the last client who spoke you, that can help you decide you what to say next: {description}"

    # Check if the file exists and load the existing data
    data = load_memories()

    # Create or update the session in the data
    data[chat_session_id] = {
        "chat_session_id": chat_session_id,
        "memory": memory
    }
    print("SESSIONNNNN", data[chat_session_id])
    save_memories(data)
    return jsonify({"success": "1"})


@app.route('/chats/<chat_session_id>/get-memories', methods=['GET'])
def get_memories(chat_session_id):
    """
    Retrieve stored memories for a specific chat session.
    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
    responses:
      200:
        description: Successfully retrieved memories for the chat session.
        schema:
          type: object
          properties:
            memories:
              type: string
              description: The stored memories for the chat session.
      400:
        description: Invalid chat session ID.
      404:
        description: Chat session not found.
    """
    data = load_memories()
    if chat_session_id in data:
      current_session_data = data[chat_session_id]
      # get memory from the database for current user
      memory = current_session_data['memory']
    else:
      memory = 'No memories available for the current session.'
    return jsonify({"memories": memory})


if __name__ == "__main__":
    # In production, you would use a real WSGI server like gunicorn/uwsgi
    app.run(debug=True, host="0.0.0.0", port=5000)
