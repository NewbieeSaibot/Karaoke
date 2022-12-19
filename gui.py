import os
import tkinter as tk
from abc import ABC


class GUI(ABC):
    def __init__(self):
        pass

    def run(self):
        pass


class TkinterGUI(GUI):
    def __init__(self, files_path: str):
        self.files_path = files_path

    def run(self):
        window = tk.Tk()
        window.geometry("800x600")
        window.title("Karaoke")
        # label = tk.Label(window, text="Welcome to AI Karaoke!!").pack()
        list_items = tk.Variable(value=self.__get_music_list())
        listbox = tk.Listbox(window,
                             listvariable=list_items,
                             height=4,
                             selectmode=tk.EXTENDED)
        listbox.pack(expand=False, fill=tk.BOTH)
        window.mainloop()

    def __get_music_list(self):
        return os.listdir(self.files_path)
