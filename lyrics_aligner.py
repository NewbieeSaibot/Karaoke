import numpy as np
import torch
import torchaudio
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
import librosa
import soundfile
import os
import pickle


class LyricsAligner:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def align(self, lyrics: str, vocal_audio_path: str, music_name: str):
        # Does the aligned lyric already exists?
        if os.path.exists(os.path.join(self.base_path, music_name, music_name + ".lrc")):
            file = open(os.path.join(self.base_path, music_name, music_name + ".lrc"), "rb")
            aligned_lyric = pickle.load(file)
            file.close()
            return aligned_lyric

        wave, sr = librosa.load(vocal_audio_path)
        waves, offset_times = self.__separate_waves(wave, sr)
        if not os.path.exists(os.path.join(self.base_path, music_name, "tmp")):
            os.mkdir(os.path.join(self.base_path, music_name, "tmp"))

        for i in range(len(waves)):
            soundfile.write(os.path.join(self.base_path, music_name, "tmp", str(i) + ".wav"), waves[i], sr)
        cuts = [-1]
        for i in range(len(lyrics)):
            if lyrics[i] == "|":
                cuts.append(i)
        cuts.append(len(lyrics))

        final_alignment = []
        last_index = 0
        offset_list = []
        for i in range(len(waves)):
            for j in range(len(cuts) - (last_index + 1)):
                sublyric = lyrics[cuts[last_index] + 1:cuts[last_index + j + 1]]
                curr_align = self.align_lyrics_with_audio(sublyric, os.path.join(self.base_path, music_name, "tmp", str(i) + ".wav"),
                                                          verbose_plot=False)
                if j != 0:
                    if curr_align[j]['initial_percentage'] < final_alignment[-1]['final_percentage'] and \
                            len(waves[i]) / sr - curr_align[j]['final_percentage'] < 1:
                        last_index += j
                        break
                    else:
                        final_alignment.append(curr_align[j])
                        offset_list.append(i)
                else:
                    final_alignment.append(curr_align[j])
                    offset_list.append(i)

        for i in range(len(final_alignment)):
            final_alignment[i]["initial_percentage"] += offset_times[offset_list[i]]
            final_alignment[i]["final_percentage"] += offset_times[offset_list[i]]

        # save aligned lyric
        file = open(os.path.join(self.base_path, music_name, music_name + ".lrc"), "wb")
        pickle.dump(final_alignment, file)
        file.close()

        return final_alignment

    def get_amp_envelope(self, wave: np.ndarray, window_size: int):
        amp_envelope = np.zeros(int(len(wave) / window_size))
        rectified_wave = np.abs(wave)
        for i in range(0, len(amp_envelope)):
            amp_envelope[i] = np.sum(rectified_wave[i*window_size:(i + 1)*window_size])
        return amp_envelope

    def detect_separation_points(self, amp_envelope: np.ndarray, sample_rate: int, window_size: int):
        amp_envelope = (amp_envelope - np.min(amp_envelope)) / np.max(amp_envelope)
        noise_level = 0.05  # np.percentile(amp_envelope, [50])
        threshold = 6 * noise_level
        start_points, end_points = [], []
        mode = 0
        for i in range(1, len(amp_envelope)):
            if mode == 0:
                # Look for a start point
                if amp_envelope[i] > threshold and amp_envelope[i] - \
                   amp_envelope[i - 1] > 0:
                    start_points.append(i)
                    mode = 1
            else:
                # Look for end point
                if amp_envelope[i] < noise_level and amp_envelope[i] - \
                   amp_envelope[i - 1] < 0:
                    end_points.append(i)
                    mode = 0

        separation_points = []
        for i in range(len(end_points)):
            separation_points.append((window_size*start_points[i]/sample_rate - 0.5, window_size*end_points[i]/sample_rate))
        return separation_points

    def __separate_waves(self, wave: np.ndarray, sample_rate: int):
        minimum_considered_frequency = 20
        time_of_complete_wave_at_minimum_frequency = 1/minimum_considered_frequency
        frames_per_wave_at_min_freq = 8 * int(time_of_complete_wave_at_minimum_frequency * sample_rate)
        amp_envelope = self.get_amp_envelope(wave, frames_per_wave_at_min_freq)
        separation_points = self.detect_separation_points(amp_envelope, sample_rate, frames_per_wave_at_min_freq)

        separated_waves = []
        offset_times = []
        for separation_point in separation_points:
            ini = int(max(0, separation_point[0] * sample_rate))
            end = int(min(len(wave), separation_point[1] * sample_rate))
            separated_waves.append(wave[ini:end])
            offset_times.append(separation_point[0])

        return separated_waves, offset_times

    def align_lyrics_with_audio(self, lyrics: str, audio_path: str, verbose_plot: bool = False,
                                verbose_settings: bool = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose_settings:
            print(torch.__version__)
            print(torchaudio.__version__)
            print(device)

        # Set initial configs of plots and random seed of pytorch
        if verbose_plot:
            matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
        torch.random.manual_seed(0)

        SPEECH_FILE = torchaudio.utils.download_asset(audio_path)
        # Generate frame-wise label probability
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        labels = bundle.get_labels()
        with torch.inference_mode():
            waveform, sample_rate = torchaudio.load(SPEECH_FILE)
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        if verbose_plot:
            print(labels)
            plt.imshow(emission.T)
            plt.colorbar()
            plt.title("Frame-wise class probability")
            plt.xlabel("Time")
            plt.ylabel("Labels")
            plt.show()

        # transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
        # transcript = "DON'T|GO|WASTIN'|MY|TIME|OUTTA|YOUR|MIND|ARE|YOU|OUT|OF|YOUR|MIND|I|AIN'T|MAKIN'|YOU|MINE|I|WANNA|SEE|YOU|GONE"
        transcript = lyrics
        dictionary = {c: i for i, c in enumerate(labels)}

        tokens = [dictionary[c] for c in transcript]
        if verbose_plot:
            print(list(zip(transcript, tokens)))

        def get_trellis(emission, tokens, blank_id=0):
            num_frame = emission.size(0)
            num_tokens = len(tokens)

            # Trellis has extra diemsions for both time axis and tokens.
            # The extra dim for tokens represents <SoS> (start-of-sentence)
            # The extra dim for time axis is for simplification of the code.
            trellis = torch.empty((num_frame + 1, num_tokens + 1))
            trellis[0, 0] = 0
            trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
            trellis[0, -num_tokens:] = -float("inf")
            trellis[-num_tokens:, 0] = float("inf")

            for t in range(num_frame):
                trellis[t + 1, 1:] = torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
            return trellis

        trellis = get_trellis(emission, tokens)

        if verbose_plot:
            plt.imshow(trellis[1:, 1:].T, origin="lower")
            plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
            plt.colorbar()
            plt.show()

        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emission, tokens, blank_id=0):
            # Note:
            # j and t are indices for trellis, which has extra dimensions
            # for time and tokens at the beginning.
            # When referring to time frame index `T` in trellis,
            # the corresponding index in emission is `T-1`.
            # Similarly, when referring to token index `J` in trellis,
            # the corresponding index in transcript is `J-1`.
            j = trellis.size(1) - 1
            t_start = torch.argmax(trellis[:, j]).item()

            path = []
            for t in range(t_start, 0, -1):
                # 1. Figure out if the current position was stay or change
                # Note (again):
                # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
                # Score for token staying the same from time frame J-1 to T.
                stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
                # Score for token changing from C-1 at T-1 to J at T.
                changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

                # 2. Store the path with frame-wise probability.
                prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
                # Return token index and time index in non-trellis coordinate.
                path.append(Point(j - 1, t - 1, prob))

                # 3. Update the token
                if changed > stayed:
                    j -= 1
                    if j == 0:
                        break
            else:
                raise ValueError("Failed to align")
            return path[::-1]

        path = backtrack(trellis, emission, tokens)
        if verbose_plot:
            for p in path:
                print(p)

        def plot_trellis_with_path(trellis, path):
            # To plot trellis with path, we take advantage of 'nan' value
            trellis_with_path = trellis.clone()
            for _, p in enumerate(path):
                trellis_with_path[p.time_index, p.token_index] = float("nan")
            plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")

        if verbose_plot:
            plot_trellis_with_path(trellis, path)
            plt.title("The path found by backtracking")
            plt.show()

        # Merge the labels
        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float

            def __repr__(self):
                return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

            @property
            def length(self):
                return self.end - self.start

        def merge_repeats(path):
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(
                    Segment(
                        transcript[path[i1].token_index],
                        path[i1].time_index,
                        path[i2 - 1].time_index + 1,
                        score,
                    )
                )
                i1 = i2
            return segments

        segments = merge_repeats(path)
        if verbose_plot:
            for seg in segments:
                print(seg)

        def plot_trellis_with_segments(trellis, segments, transcript):
            # To plot trellis with path, we take advantage of 'nan' value
            trellis_with_path = trellis.clone()
            for i, seg in enumerate(segments):
                if seg.label != "|":
                    trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

            fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
            ax1.set_title("Path, label and probability for each label")
            ax1.imshow(trellis_with_path.T, origin="lower")
            ax1.set_xticks([])

            for i, seg in enumerate(segments):
                if seg.label != "|":
                    ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
                    ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

            ax2.set_title("Label probability with and without repetation")
            xs, hs, ws = [], [], []
            for seg in segments:
                if seg.label != "|":
                    xs.append((seg.end + seg.start) / 2 + 0.4)
                    hs.append(seg.score)
                    ws.append(seg.end - seg.start)
                    ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
            ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

            xs, hs = [], []
            for p in path:
                label = transcript[p.token_index]
                if label != "|":
                    xs.append(p.time_index + 1)
                    hs.append(p.score)

            ax2.bar(xs, hs, width=0.5, alpha=0.5)
            ax2.axhline(0, color="black")
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(-0.1, 1.1)

        if verbose_plot:
            plot_trellis_with_segments(trellis, segments, transcript)
            plt.tight_layout()
            plt.show()

        # Merge words
        def merge_words(segments, separator="|"):
            words = []
            i1, i2 = 0, 0
            while i1 < len(segments):
                if i2 >= len(segments) or segments[i2].label == separator:
                    if i1 != i2:
                        segs = segments[i1:i2]
                        word = "".join([seg.label for seg in segs])
                        score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                        words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                    i1 = i2 + 1
                    i2 = i1
                else:
                    i2 += 1
            return words

        word_segments = merge_words(segments)

        if verbose_plot:
            for word in word_segments:
                print(word)

        total_number_of_windows = emission.shape[0]
        audio_duration = waveform.shape[1] / sample_rate
        final_word_list = []
        for word in word_segments:
            word_item = {"word": word.label,
                         "initial_percentage": audio_duration * (word.start / total_number_of_windows),
                         "final_percentage": audio_duration * (word.end / total_number_of_windows)}
            final_word_list.append(word_item)

        return final_word_list
