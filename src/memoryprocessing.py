import os

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from utils import call_openai_api

# File paths for storing memory and user summaries
summary_persona_file_path = "../data/summary_personas.csv"


# Function to initialize or load the summary persona DataFrame
def load_or_create_summary_persona():
    if os.path.exists(summary_persona_file_path):
        return pd.read_csv(summary_persona_file_path)
    else:
        return pd.DataFrame(columns=["user_id", "summary_persona"])


# Function to create or update the summary persona (Implementation missing)
def generate_summary_persona(messages):
    """
    Generate personas based on conversation data, grouping each human speaker with robot responses.

    Args:
        messages: List of conversation exchanges

    Returns:
        Dictionary of speaker_ids and their generated personas
    """
    # Identify all unique speakers excluding the robot
    human_speakers = set()
    for exchange in messages:
        for message_obj in exchange:
            if message_obj["speaker_id"] != "robot":
                human_speakers.add(message_obj["speaker_id"])

    # Organize conversations by human speaker + robot responses
    speaker_conversations = {}
    for speaker_id in human_speakers:
        speaker_conversations[speaker_id] = []

    # Reconstruct conversation flows for each human speaker with robot
    robot_responses = []
    current_speakers = set()

    for exchange in messages:
        # Get speakers in this exchange
        exchange_speakers = set()
        for message_obj in exchange:
            if message_obj["speaker_id"] != "robot":
                exchange_speakers.add(message_obj["speaker_id"])

        # Add human messages to their respective conversations
        for message_obj in exchange:
            if message_obj["speaker_id"] != "robot":
                speaker_conversations[message_obj["speaker_id"]].append(
                    {
                        "speaker_id": message_obj["speaker_id"],
                        "message": message_obj["message"],
                    }
                )
            else:
                # This is a robot response - store it
                robot_responses.append(
                    {
                        "speaker_id": "robot",
                        "message": message_obj["message"],
                        "responding_to": exchange_speakers,  # Which speakers the robot is responding to
                    }
                )

                # Add this robot response to all speakers from the previous exchange
                for speaker_id in current_speakers:
                    speaker_conversations[speaker_id].append(
                        {"speaker_id": "robot", "message": message_obj["message"]}
                    )

        # Update current speakers for the next iteration
        if exchange_speakers:
            current_speakers = exchange_speakers

    # Generate personas for each human speaker
    personas = {}

    # Load the Jinja template
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("summary_persona.jinja")

    for speaker_id, conversation in speaker_conversations.items():
        # Render the template with the speaker-robot conversation
        prompt = template.render(speaker_id=speaker_id, conversation=conversation)

        response = call_openai_api(prompt)
        personas[speaker_id] = response

    # save personas to csv with
    data = []
    for user_id, summary_persona in personas.items():
        data.append({"user_id": user_id, "summary_persona": summary_persona})

    # Create DataFrame with specified columns
    df = pd.DataFrame(data, columns=["user_id", "summary_persona"])

    # Save to CSV
    df.to_csv(summary_persona_file_path, index=False)
    return personas
