from karaoke import Karaoke
from gui import TkinterGUI


def main():
    music_files_path = "C:\\Users\\tobia\\Desktop\\musicas\\karaoke"

    karaoke = Karaoke(music_files_path)
    gui = TkinterGUI(karaoke, music_files_path)
    gui.run()


if __name__ == '__main__':
    main()
