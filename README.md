Multilingual Voice Assistant

A multilingual desktop voice assistant built with Python and Tkinter, featuring live speech recognition, language detection, and AI-generated responses using the Gemma model.

🎯 Features

🎧 Toggle-based microphone control (press once to listen, press again to stop)

🌐 Detects languages including English, Spanish, French, and Haitian Creole

🤖 AI-powered responses using Gemma (via ollama)

🧠 Custom-trained language classifier fallback

🔈 Text-to-speech playback using gTTS and pygame

🮡 Beautiful GUI built with Tkinter and themed with ttk

🚀 Getting Started

Prerequisites

Install the required Python packages using pip:

pip install tkinter gTTS pygame pycld2 langdetect speechrecognition scikit-learn joblib

Also, install and configure Ollama and the Gemma 1B model:

ollama run gemma3:1b

Files Needed

Ensure the following files exist in your project folder:

custom_lang_model.pkl — Trained classifier for language detection

custom_vectorizer.pkl — Fitted TF-IDF vectorizer

Run the App

Run the application with:

python your_script_name.py

(Replace your_script_name.py with the filename containing the app code.)

🛠 How It Works

Speech Recognition — Captures your voice via microphone using Google Speech Recognition API.

Language Detection — Uses a custom ML model, pycld2, or langdetect as fallback.

AI Response — Generates a reply via ollama running the gemma3:1b model.

Text-to-Speech — Uses gTTS to vocalize the assistant's response.

🎨 UI Overview

🎧 Mic Button: Start/stop listening

📃 Chat Window: Displays conversation history

🌐 Language Label: Detected language and confidence

🔄 Status Indicator: Shows listening/processing status

📝 Tech Stack

Tkinter, ttk — User interface

speech_recognition — Microphone input

scikit-learn, joblib — Custom language model

pycld2, langdetect — Language detection fallbacks

subprocess — Interface with ollama

gTTS, pygame — Speech synthesis and playback

threading, queue — Background task management

📂 Project Structure

.
├── custom_lang_model.pkl
├── custom_vectorizer.pkl
├── your_script_name.py
├── temp/
│   └── response.mp3

⚠️ Notes

Ensure ollama and gemma3:1b are installed and runnable.

Internet connection is required for speech recognition and gTTS.

👩‍💻 Author

Built by Stephany Rodriguez and Percy Valera
