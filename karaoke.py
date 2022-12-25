from lyrics_aligner import LyricsAligner
from lyrics_extractor import LyricsExtractor
from music_player import MusicPlayer, PyGameMusicPlayer
from un_mix_model import UnMixModel, Demucs3
from speech_recognition_model import GoogleRecognizer, SpeechRecognitionModel
import os
import threading


class Karaoke:
    """
    This class is an abstraction to the karaoke functionalities.
    Main functionality is to realize the orchestration of machine learning models to initialize a single
    run.
    """
    def __init__(self, base_path: str, music_player: MusicPlayer = PyGameMusicPlayer,
                 un_mix_model: UnMixModel = Demucs3,
                 speech_recognition_model: SpeechRecognitionModel = GoogleRecognizer):
        self.base_path = base_path
        self.music_list = self.__get_music_list()

        self.un_mix_model = un_mix_model(base_path)
        self.recognition_model = speech_recognition_model()
        self.lyrics_aligner = LyricsAligner(base_path)
        self.lyrics_extractor = LyricsExtractor()
        self.music_player = music_player()

    def start_karaoke(self, selected_music_name: str):
        """Do every necessary logic to start to sing a new music.
        1. If necessary calculate the separated tracks with unmix model.
        2. If necessary align the lyrics with the vocal audio.
        3. Play the music with no vocal.
        4. Show the Lyrics in the GUI.
        """
        vocal_audio_path, no_vocal_audio_path = self.un_mix_model.predict(selected_music_name)
        aligned_lyrics = self.__get_aligned_lyrics(selected_music_name, vocal_audio_path)
        self.__play_music(no_vocal_audio_path)
        self.__show_lyrics(aligned_lyrics)

    def __get_aligned_lyrics(self, music_name: str, vocal_audio_path: str):
        lyric = self.lyrics_extractor.extract()
        aligned_words = self.lyrics_aligner.align(lyric, vocal_audio_path, music_name)
        return aligned_words

    def __show_lyrics(self, aligned_lyrics: str):
        # interact with GUI
        pass

    def __get_music_list(self):
        return os.listdir(self.base_path)

    def __play_music(self, music_path: str):
        x = threading.Thread(target=self.music_player.play, args=(music_path,))
        x.start()
