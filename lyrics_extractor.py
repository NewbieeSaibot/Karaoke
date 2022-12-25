import os


class LyricsExtractor:
    def __init__(self):
        pass

    def remove_special_characters(self, text: str):
        special_charcaters = ["!", "?", ";", ",", ".", "(", ")", "[", "]", "{", "}"]
        for spec in special_charcaters:
            text = text.replace(spec, "")
        return text

    def extract(self, music_name: str = "nightjar") -> str:
        base_path = "C:\\Users\\tobia\\Desktop\\musicas\\karaoke"
        if music_name == "nightjar":
            file = open(os.path.join(base_path, music_name, music_name + ".txt"), "r")
            lines = file.readlines()
            lyric = ""
            for line in lines:
                if len(line.strip()) > 0:
                    lyric += self.remove_special_characters(line.strip().upper()).replace(" ", "|") + "|"
            return lyric[:-1]
        else:
            return ""
