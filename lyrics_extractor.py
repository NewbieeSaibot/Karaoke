import os


class LyricsExtractor:
    def __init__(self):
        pass

    def remove_special_characters(self, text: str):
        special_charcaters = ["!", "?", ";", ",", ".", "(", ")", "[", "]", "{", "}"]
        for spec in special_charcaters:
            text = text.replace(spec, "")

        text = text.replace("Ç", "C")

        text = text.replace("Á", "A")
        text = text.replace("À", "A")
        text = text.replace("Ã", "A")
        text = text.replace("Â", "A")

        text = text.replace("É", "E")
        text = text.replace("È", "E")
        text = text.replace("Ê", "E")

        text = text.replace("Í", "I")
        text = text.replace("Ì", "I")
        text = text.replace("Î", "I")

        text = text.replace("Ó", "O")
        text = text.replace("Ò", "O")
        text = text.replace("Õ", "O")
        text = text.replace("Ô", "O")

        text = text.replace("Ú", "U")
        text = text.replace("Ù", "U")
        text = text.replace("Û", "U")
        print(text)
        return text

    def extract(self, music_name: str = "") -> str:
        base_path = "C:\\Users\\tobia\\Desktop\\musicas\\karaoke"

        file = open(os.path.join(base_path, music_name, music_name + ".txt"), "r", encoding='utf8')
        lines = file.readlines()
        lyric = ""
        for line in lines:
            if len(line.strip()) > 0:
                lyric += self.remove_special_characters(line.strip().upper()).replace(" ", "|") + "|"
        return lyric[:-1]
