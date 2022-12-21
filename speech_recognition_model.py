import numpy as np
import speech_recognition as sr
from abc import ABC


class SpeechRecognitionModel(ABC):
    def __init__(self):
        pass

    def predict(self, audio_path: str):
        pass


class GoogleRecognizer(SpeechRecognitionModel):
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def predict(self, audio_path: str):
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source, duration=10, offset=20)

        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return ""
