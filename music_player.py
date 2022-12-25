from abc import ABC
from pygame import mixer


class MusicPlayer(ABC):
    def __init__(self):
        pass

    def play(self, audio):
        pass


class PyGameMusicPlayer(MusicPlayer):
    def __init__(self):
        mixer.init()

    def play(self, music_path: str):
        # Load audio file
        mixer.music.load(music_path)

        print("music started playing....")

        # Set preferred volume
        mixer.music.set_volume(0.8)

        # Play the music
        mixer.music.play()

        # Infinite loop
        while True:
            print("------------------------------------------------------------------------------------")
            print("Press 'p' to pause the music")
            print("Press 'r' to resume the music")
            print("Press 'e' to exit the program")

            # take user input
            userInput = input(" ")

            if userInput == 'p':

                # Pause the music
                mixer.music.pause()
                print("music is paused....")
            elif userInput == 'r':

                # Resume the music
                mixer.music.unpause()
                print("music is resumed....")
            elif userInput == 'e':

                # Stop the music playback
                mixer.music.stop()
                print("music is stopped....")
                break
