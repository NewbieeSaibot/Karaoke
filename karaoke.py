from gui import TkinterGUI


class Karaoke:
    def __init__(self, files_path: str):
        self.gui = TkinterGUI(files_path)
        self.un_mix_model = None  # Demucs3
        self.recognition_model = None  # SpeechRecognition

    def run(self):
        self.gui.run()
