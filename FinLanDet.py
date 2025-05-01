import tkinter as tk
from tkinter import messagebox, ttk
import speech_recognition as sr
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import pycld2 as cld2
from langdetect import detect, DetectorFactory
import threading
import pygame
import time
import os
from gtts import gTTS
import queue

# Set deterministic seed for langdetect
DetectorFactory.seed = 0
# Initialize pygame mixer for audio playback
pygame.mixer.init()

class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Voice Assistant")
        self.root.geometry("700x500")
        self.root.configure(bg="#f0f4f8")

        # Define color palette
        self.primary_color = "#6366f1"
        self.secondary_color = "#8b5cf6"
        self.bg_color = "#f0f4f8"
        self.text_color = "#1e293b"
        self.accent_color = "#4f46e5"

        # Attempt to load custom language model and vectorizer
        try:
            self.clf = joblib.load("custom_lang_model.pkl")
            self.vectorizer = joblib.load("custom_vectorizer.pkl")
        except Exception as e:
            print(f"Error loading models: {e}")
            messagebox.showerror("Model Error", f"Could not load language models: {e}")
            self.clf, self.vectorizer = None, None

        # Initialize UI and states
        self.setup_ui()
        self.conversation = []
        self.is_listening = False
        self.current_language = "en"

        # Create job queue for sequential processing
        self.job_queue = queue.Queue()
        threading.Thread(target=self.process_queue_worker, daemon=True).start()

    # Worker thread to handle queued tasks
    def process_queue_worker(self):
        while True:
            job_func, args = self.job_queue.get()
            try:
                job_func(*args)
            except Exception as e:
                print(f"Error processing job: {e}")
            self.job_queue.task_done()

    # UI setup using Tkinter widgets
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Style definitions
        style = ttk.Style()
        style.configure("TFrame", background=self.bg_color)
        style.configure("TButton", background=self.primary_color, foreground="white", font=("Helvetica", 12, "bold"), padding=10)
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground=self.primary_color, background=self.bg_color)
        style.configure("Status.TLabel", font=("Helvetica", 10), foreground=self.accent_color, background=self.bg_color)

        # Header section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        title_label = ttk.Label(header_frame, text="\U0001F30D Multilingual Voice Assistant", style="Header.TLabel")
        title_label.pack(side=tk.LEFT)
        self.language_label = ttk.Label(header_frame, text="", font=("Helvetica", 12, "bold"), foreground=self.secondary_color, background=self.bg_color)
        self.language_label.pack(side=tk.RIGHT)

        # Chat display area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.chat_display = tk.Text(chat_frame, wrap=tk.WORD, font=("Helvetica", 11), background="white", foreground=self.text_color, relief=tk.FLAT, height=15)
        self.chat_display.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.chat_display.config(state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_display.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.chat_display.config(yscrollcommand=scrollbar.set)

        # Status indicator area
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_label = ttk.Label(status_frame, text="Ready", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        self.animation_canvas = tk.Canvas(status_frame, width=30, height=30, background=self.bg_color, highlightthickness=0)
        self.animation_canvas.pack(side=tk.RIGHT)
        self.animation_obj = None

        # Microphone control button
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.mic_button_frame = tk.Frame(control_frame, background=self.primary_color, relief=tk.RAISED, borderwidth=0)
        self.mic_button_frame.pack(pady=10)
        self.mic_button = tk.Button(self.mic_button_frame, text="\U0001F399 Start Listening", command=self.toggle_listening, font=("Helvetica", 12, "bold"), background=self.primary_color, foreground="white", activebackground=self.accent_color, activeforeground="white", width=20, height=2, relief=tk.FLAT, borderwidth=0, cursor="hand2")
        self.mic_button.pack(padx=2, pady=2)
        self.mic_button.bind("<Enter>", lambda e: self.on_button_hover(True))
        self.mic_button.bind("<Leave>", lambda e: self.on_button_hover(False))

    # Button hover animation
    def on_button_hover(self, is_hovering):
        color = self.secondary_color if is_hovering else self.primary_color
        self.mic_button.config(background=color)
        self.mic_button_frame.config(background=color)

    # Update GUI status and animation state
    def update_status(self, message, is_listening=False):
        self.status_label.config(text=message)
        if self.animation_obj:
            self.animation_canvas.delete(self.animation_obj)
        if is_listening:
            self.animate_recording()
        else:
            self.animation_obj = self.animation_canvas.create_oval(5, 5, 25, 25, fill="gray", outline=self.bg_color)

    # Animate pulsing red circle when listening
    def animate_recording(self):
        if self.animation_obj:
            self.animation_canvas.delete(self.animation_obj)
        size_factor = abs(int(5 * (1 + 0.3 * (time.time() % 2))))
        self.animation_obj = self.animation_canvas.create_oval(10-size_factor, 10-size_factor, 20+size_factor, 20+size_factor, fill="#ef4444", outline=self.bg_color)
        if self.is_listening:
            self.root.after(100, self.animate_recording)

    # Toggle microphone listening state
    def toggle_listening(self):
        if self.is_listening:
            self.is_listening = False
            self.mic_button.config(text="\U0001F399 Start Listening")
            self.update_status("Ready")
        else:
            self.is_listening = True
            self.mic_button.config(text="\u23F9\uFE0F Stop Listening")
            self.update_status("\U0001F399 Listening...", is_listening=True)
            self.job_queue.put((self.listen_to_speech, ()))

    # Append message to chat display
    def add_message(self, text, sender="user"):
        self.chat_display.config(state=tk.NORMAL)
        if self.chat_display.get("1.0", tk.END).strip():
            self.chat_display.insert(tk.END, "\n\n")
        tag = "user_tag" if sender == "user" else "assistant_tag"
        prefix = "You: " if sender == "user" else "Assistant: "
        self.chat_display.insert(tk.END, prefix, tag)
        self.chat_display.insert(tk.END, text)
        self.chat_display.tag_configure("user_tag", foreground=self.primary_color, font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("assistant_tag", foreground=self.secondary_color, font=("Helvetica", 11, "bold"))
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    # Capture and recognize speech input
    def listen_to_speech(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)
                self.root.after(0, lambda: self.update_status("Processing speech..."))
                try:
                    text = recognizer.recognize_google(audio)
                    self.root.after(0, lambda: self.process_speech(text))
                except sr.UnknownValueError:
                    fallback = "I'm sorry, I didn't catch that. Can you please repeat?"
                    self.root.after(0, lambda: self.add_message(fallback, "assistant"))
                    self.root.after(0, lambda: self.speak_text(fallback, "en"))
                    self.root.after(0, lambda: self.update_status("Ready"))
                    self.root.after(0, lambda: self.toggle_listening())
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.toggle_listening())

    # Language detection pipeline
    def detect_language(self, text):
        try:
            if self.clf and self.vectorizer:
                x = self.vectorizer.transform([text])
                lang_code = self.clf.predict(x)[0]
                if lang_code in ["en", "fr", "es", "hat"]:
                    return "Custom Model", lang_code, "100%", text
            is_reliable, _, details = cld2.detect(text)
            return "CLD2", details[0][1], f"{details[0][2]}%", text
        except:
            try:
                lang_code = detect(text)
                return "Langdetect", lang_code, "Unknown", text
            except:
                return "Default", "en", "Unknown", text

    # Analyze speech input and queue response generation
    def process_speech(self, text):
        if not text:
            fallback = "I'm sorry, I didn't catch that. Can you please repeat?"
            self.add_message(fallback, "assistant")
            self.update_status("Ready")
            self.toggle_listening()
            self.speak_text(fallback, "en")
            return
        self.add_message(text, "user")
        detector, lang_code, confidence, _ = self.detect_language(text)
        self.current_language = lang_code
        lang_name = self.get_language_name(lang_code)
        self.language_label.config(text=f"\U0001F310 {lang_name} ({lang_code}) - {confidence}")
        self.update_status(f"Detected {lang_name} via {detector}. Getting response...")
        self.job_queue.put((self.get_gemma_response, (text, lang_code)))

    # Map ISO language code to readable name
    def get_language_name(self, lang_code):
        language_map = {"en": "English", "es": "Spanish", "fr": "French", "hat": "Haitian Creole", "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi"}
        return language_map.get(lang_code, f"Unknown ({lang_code})")

    # Generate a response using Gemma model via subprocess
    def get_gemma_response(self, text, lang_code):
        try:
            prompts = {
                "en": f"You are a helpful multilingual assistant. Respond in English: '{text}'",
                "es": f"Eres un asistente multiling\u00fce. Responde en espa\u00f1ol: '{text}'",
                "fr": f"Tu es un assistant multilingue. R\u00e9ponds en fran\u00e7ais: '{text}'",
                "hat": f"Ou se yon asistan miltileng. Reponn an krey\u00f2l: '{text}'"
            }
            prompt = prompts.get(lang_code, f"You are a helpful assistant. Respond to: '{text}'")
            result = subprocess.run(["ollama", "run", "gemma3:1b"], input=prompt.encode("utf-8"), capture_output=True, timeout=30)
            gemma_response = result.stdout.decode("utf-8").strip()
            self.root.after(0, lambda: self.add_message(gemma_response, "assistant"))
            self.root.after(0, lambda: self.update_status("Ready"))
            self.root.after(0, lambda: self.speak_text(gemma_response, lang_code))
            self.root.after(3000, lambda: self.toggle_listening())
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Gemma error: {e}"))
            self.root.after(0, lambda: self.toggle_listening())

    # Use gTTS to convert text to speech and play it
    def speak_text(self, text, lang_code):
        try:
            tts_lang_code = "fr" if lang_code == "hat" else lang_code
            if not os.path.exists("temp"):
                os.makedirs("temp")
            tts = gTTS(text=text, lang=tts_lang_code, slow=False)
            tts.save("temp/response.mp3")
            pygame.mixer.music.load("temp/response.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Text-to-speech error: {e}")

# Launch the app if run directly
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    root.mainloop()
