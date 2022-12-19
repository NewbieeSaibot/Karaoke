from abc import ABC
from playsound import playsound
import simpleaudio
import pydub
import vlc


class MusicPlayer(ABC):
    def __init__(self):
        pass

    def play(self, audio):
        pass


class PlaySoundMusicPlayer(MusicPlayer):
    def __init__(self):
        pass

    def play(self, audio):
        # playsound(audio)
        '''
        sound = pydub.AudioSegment.from_wav(audio)
        playback = simpleaudio.play_buffer(

            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        '''
        vlc.MediaPlayer(audio)
        vlc.play()
