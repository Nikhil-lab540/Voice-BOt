import streamlit as st
import whisper
from gtts import gTTS
import tempfile
from groq import Groq
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Simple Voice Chatbot", page_icon="🎤")
st.title("🎙️ Voice Chatbot")
st.write("Speak to the AI and get audio responses back")

# -------------------------------
# Groq client
# -------------------------------
client = Groq(api_key="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")

# -------------------------------
# Load Whisper model
# -------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# -------------------------------
# Init session state
# -------------------------------
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# -------------------------------
# Audio input or upload
# -------------------------------
st.subheader("🎤 Record Your Voice")
audio_bytes = st.audio_input("Click to record your voice", key="voice_input")

st.subheader("📁 Or Upload an Audio File")
uploaded_audio = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

# Determine audio source
audio_source = None
audio_label = ""
if audio_bytes:
    if len(audio_bytes.getvalue()) < 2000:
        st.warning("⚠️ Mic recording too short or blank. Try again.")
    else:
        audio_source = audio_bytes.read()
        audio_label = "🎤 Recorded Audio"
elif uploaded_audio:
    audio_source = uploaded_audio.read()
    audio_label = "📁 Uploaded Audio"

# -------------------------------
# Process audio
# -------------------------------
if audio_source:
    with st.status("Processing...", expanded=True) as status:
        st.write("🔁 Saving audio to file...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_source)
            temp_audio_path = temp_audio.name

        st.write(f"🎧 {audio_label}:")
        st.audio(temp_audio_path)

        # -------------------------------
        # Plot waveform using librosa
        # -------------------------------
        st.write("📊 Audio Waveform Preview:")
        try:
            y, sr = librosa.load(temp_audio_path, sr=None)
            fig, ax = plt.subplots(figsize=(6, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Waveform")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot waveform: {e}")

        # -------------------------------
        # Transcribe
        # -------------------------------
        st.write("📝 Transcribing with Whisper...")
        transcription = whisper_model.transcribe(temp_audio_path, fp16=False)["text"]

        if not transcription.strip():
            st.warning("⚠️ No speech detected. Try again with clearer audio.")
            status.update(label="No speech detected", state="error")
        else:
            st.write(f"🗣️ You said: `{transcription}`")

            st.write("🤖 Generating AI response...")
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": transcription}],
                model="llama3-8b-8192"
            )
            response_text = response.choices[0].message.content

            st.write("🔊 Converting response to speech...")
            tts = gTTS(response_text)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_response:
                tts.save(temp_audio_response.name)
                response_audio_path = temp_audio_response.name

            st.session_state.conversation.append({
                "user": transcription,
                "ai": response_text
            })

            status.update(label="✅ Done!", state="complete")

            st.subheader("AI Response:")
            st.write(response_text)
            st.audio(response_audio_path)

# -------------------------------
# Show conversation history
# -------------------------------
if st.session_state.conversation:
    st.subheader("📜 Conversation History")
    for i, exchange in enumerate(st.session_state.conversation):
        st.markdown(f"**You:** {exchange['user']}")
        st.markdown(f"**AI:** {exchange['ai']}")
        if i < len(st.session_state.conversation) - 1:
            st.divider()

# -------------------------------
# Clear button
# -------------------------------
if st.button("🧹 Clear Conversation"):
    st.session_state.conversation = []
    st.rerun()
