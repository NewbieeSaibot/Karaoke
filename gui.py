from abc import ABC
import tkinter as tk


class GUI(ABC):
    def __init__(self):
        pass

    def run(self):
        pass


class TkinterGUI(GUI):
    def __init__(self, karaoke, files_path: str):
        self.karaoke = karaoke
        self.files_path = files_path
        self.music_list = karaoke.music_list

    def run(self):
        window = tk.Tk()
        window.geometry("800x600")
        window.title("Karaoke")
        # label = tk.Label(window, text="Welcome to AI Karaoke!!").pack()
        list_items = tk.Variable(value=self.music_list)
        self.listbox = tk.Listbox(window,
                                  listvariable=list_items,
                                  height=4,
                                  selectmode=tk.EXTENDED)
        self.listbox.pack(expand=False, fill=tk.BOTH)

        play_button = tk.Button(window, text="Start", command=self.play_function)
        play_button.pack(expand=False, fill=tk.BOTH)

        lyrics_box = tk.Text(
            window,
            height=30,
            width=100,
            font=(None, 15),
            state='disabled'
        )
        lyrics_box.pack(expand=False, fill=tk.BOTH)
        self.karaoke.lyrics_box_reference = lyrics_box

        window.mainloop()

    def play_function(self):
        selected_items_number = self.listbox.curselection()
        if len(selected_items_number) == 1:
            music_name = self.listbox.get(selected_items_number[0])
            music_name = music_name.split(".")[0]
            self.karaoke.start_karaoke(music_name)
