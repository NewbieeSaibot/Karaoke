from karaoke import Karaoke
from music_player import PlaySoundMusicPlayer


def main():
    vlc = PlaySoundMusicPlayer()
    vlc.play("C:\\Users\\tobia\\Desktop\\musicas\\karaoke\\nightjar.wav")
    # karaoke = Karaoke("C:\\Users\\tobia\\Desktop\\musicas\\karaoke")
    # karaoke.run()


if __name__ == '__main__':
    main()
