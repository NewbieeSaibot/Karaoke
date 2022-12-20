from karaoke import Karaoke
from gui import TkinterGUI


def main():
    files_path = "C:\\Users\\tobia\\Desktop\\musicas\\karaoke"
    karaoke = Karaoke(files_path)
    gui = TkinterGUI(karaoke, files_path)
    gui.run()


if __name__ == '__main__':
    main()
