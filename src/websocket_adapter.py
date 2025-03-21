import json
import threading
import time
import uuid

import requests
import websocket


class AudioSocket:
    def __init__(self):
        self.ws = None
        self.ws_thread = None
        self.is_ready = threading.Event()
        self.final_transcription_received = threading.Event()
        self.final_text = None
        self.chat_id = str(uuid.uuid4())
        self.session_id = None
        self.base_url = "http://localhost:5000"

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        data = json.loads(message)
        print(f"Received: {data}")

        # Check if this is the final transcription
        if data.get("event") == "recognized":
            print(f"Final transcription: {data['text']}")
            self.final_text = data["text"]
            self.final_transcription_received.set()

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.is_ready.clear()

    def on_open(self, ws):
        print("WebSocket connection established")
        self.is_ready.set()

    def create_session(self, language="en-US"):
        """Create a new voice input session"""
        response = requests.post(
            f"{self.base_url}/chats/{self.chat_id}/sessions",
            json={"language": language},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to create session: {response.text}")
        self.session_id = response.json()["session_id"]
        return self.session_id

    def send_audio_chunk(self, wav_data):
        """Send an audio chunk to the API"""
        if not self.session_id:
            raise Exception("No active session")

        response = requests.post(
            f"{self.base_url}/chats/{self.chat_id}/sessions/{self.session_id}/wav",
            data=wav_data,
            headers={"Content-Type": "audio/wav"},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to send audio chunk: {response.text}")
        return True

    def close_session(self):
        """Close the session and wait for final transcription"""
        if not self.session_id:
            return None

        # Reset the final transcription event
        self.final_transcription_received.clear()
        self.final_text = None

        # Send session close request
        response = requests.delete(
            f"{self.base_url}/chats/{self.chat_id}/sessions/{self.session_id}"
        )
        if response.status_code != 200:
            raise Exception(f"Failed to close session: {response.text}")

        print("Waiting for final transcription...")
        # Wait indefinitely for final transcription (no timeout)
        self.final_transcription_received.wait()
        print("Received final transcription, closing connection...")

        # Store the final text
        final_text = self.final_text

        # Now we can safely close the WebSocket
        if self.ws:
            self.ws.close()
            self.is_ready.clear()

        self.session_id = None
        return final_text

    def connect(self):
        """Connect to the WebSocket server"""
        # First create a session
        self.create_session()

        # Then connect to WebSocket
        self.ws = websocket.WebSocketApp(
            f"ws://localhost:5000/ws/chats/{self.chat_id}/sessions/{self.session_id}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )

        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait for connection to be ready
        if not self.is_ready.wait(timeout=5):
            raise Exception("WebSocket connection timeout")

        return self

    def send(self, data):
        """Send data through the WebSocket"""
        if not self.ws or not self.is_ready.is_set():
            raise Exception("WebSocket not connected")
        return self.send_audio_chunk(data)


def create_socket():
    """Create and connect a new AudioSocket"""
    socket = AudioSocket()
    return socket.connect()
