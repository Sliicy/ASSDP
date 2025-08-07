# main.py

# ==============================================================================
# === IMPORT LIBRARIES & APPLY RUNTIME PATCH ===================================
# ==============================================================================
import numpy as np
import scipy
import scipy.signal

# ✅ MONKEY-PATCH for msaf:
# The msaf library is old and tries to import `inf` from scipy, which is deprecated.
# We are manually adding numpy's `inf` to the scipy module's namespace before
# msaf is imported. This makes msaf think it found `scipy.inf` and prevents a crash.

# Patch 1: Fix for "cannot import name 'inf' from 'scipy'"
scipy.inf = np.inf

# ✅ Patch 2: Fix for "module 'scipy.signal' has no attribute 'gaussian'"
# The `gaussian` window function was moved to `scipy.signal.windows`.
scipy.signal.gaussian = scipy.signal.windows.gaussian

# Now we can safely import msaf
import msaf

from scipy.spatial.distance import cdist
import librosa
import librosa.display
import matplotlib.pyplot as plt

from pychorus import find_and_output_chorus, create_chroma
from pychorus.similarity_matrix import TimeTimeSimilarityMatrix, TimeLagSimilarityMatrix, Line
import sys
import os
from pathlib import Path

# Math utilities
from math import isinf
# We still need this for our own code, like the fastdtw function.
from numpy import inf

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# === ALGORITHM PARAMETERS =====================================================
# ==============================================================================
# Denoising size in seconds
SMOOTHING_SIZE_SEC = 2.5

# For line detection
LINE_THRESHOLD = 0.15
MIN_LINES = 10
NUM_ITERATIONS = 20

# We allow an error proportional to the length of the clip
OVERLAP_PERCENT_MARGIN = 0.2

# ==============================================================================
# === HELPER FUNCTIONS =========================================================
# ==============================================================================

def local_maxima_rows(denoised_time_lag):
    """Find the local maxima in the row-wise sum of the time-lag matrix."""
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]

def detect_lines(denoised_time_lag, rows, min_length_samples):
    """Detects horizontal lines in the time-lag matrix, which correspond to repetitions."""
    cur_threshold = LINE_THRESHOLD
    for _ in range(NUM_ITERATIONS):
        line_segments = detect_lines_helper(denoised_time_lag, rows,
                                            cur_threshold, min_length_samples)
        if len(line_segments) >= MIN_LINES:
            return line_segments
        cur_threshold *= 0.95
    return line_segments

def detect_lines_helper(denoised_time_lag, rows, threshold, min_length_samples):
    """Helper function to find line segments for a given threshold."""
    num_samples = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, num_samples):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if (cur_segment_start is not None
                   ) and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments

def count_overlapping_lines(lines, margin, min_length_samples):
    """Scores lines based on how many other lines overlap with them."""
    line_scores = {}
    for line in lines:
        line_scores[line] = 0
    for line_1 in lines:
        for line_2 in lines:
            lines_overlap_vertically = (
                line_2.start < (line_1.start + margin)) and (
                    line_2.end > (line_1.end - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)
            lines_overlap_diagonally = (
                (line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and (
                    (line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)
            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1
    return line_scores

def sorted_segments(line_scores):
    """Sorts the detected lines by score and length to find the best candidates."""
    lines_to_sort = []
    for line in line_scores:
        lines_to_sort.append((line, line_scores[line], line.end - line.start))
    lines_to_sort.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return lines_to_sort

def fastdtw(x, y, dist, warp=1):
    """A fast implementation of Dynamic Time Warping."""
    assert len(x)
    assert len(y)
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = inf # ✅ This line now works correctly because we imported `inf` from numpy
    D0[1:, 0] = inf # ✅ This line also works correctly
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    return D1[-1, -1] / sum(D1.shape)

# ==============================================================================
# === MAIN PROCESSING FUNCTION =================================================
# ==============================================================================

def process_song(audio_path, labels_dir):
    """
    Processes a single audio file to find its structure and saves the labels.
    """
    file_name = audio_path.name
    print(f"\n===== PROCESSING: {file_name} =====")
    
    try:
        # Step 1: Create Chroma and load audio
        print("Step 1: Loading audio file and creating chromagram...")
        # ✅ IMPROVEMENT: Modern pychorus/librosa have sensible defaults.
        # We ensure no outdated parameters like `n_fft` are passed.
        chroma, song_wav_data, sr, song_length_sec = create_chroma(audio_path)
        print(f"   > Audio loaded. Duration: {song_length_sec:.2f}s, Sample Rate: {sr}Hz")

        # Step 2: Use MSAF for initial segmentation
        print("Step 2: Finding initial segment boundaries with MSAF...")
        boundaries, labels = msaf.process(str(audio_path), feature="mfcc", boundaries_id="foote",
                                            labels_id="fmc2d", out_sr=sr)
        print(f"   > MSAF identified {len(boundaries)} initial boundaries.")

        # Step 3: Filter short segments and extract MFCCs
        print("Step 3: Filtering segments (minimum 5 seconds) and calculating MFCCs...")
        new_boundaries = []
        new_labels = []
        mfccs = []
        for x in range(len(boundaries) - 1):
            if boundaries[x + 1] - boundaries[x] >= 5:
                segment_wav_data = song_wav_data[int(boundaries[x]*sr) : int(boundaries[x + 1]*sr)]
                mel_freq = librosa.feature.mfcc(y=segment_wav_data, sr=sr)
                new_boundaries.append(boundaries[x])
                new_labels.append(labels[x])
                mfccs.append(np.average(mel_freq, axis=0))
        print(f"   > Found {len(new_boundaries)} viable segments after filtering.")

        # Step 4: Calculate similarity matrices
        print("Step 4: Calculating similarity matrices (Time-Time and Time-Lag)...")
        num_samples = chroma.shape[1]
        time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
        time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

        # Step 5: Denoise the matrix
        print("Step 5: Denoising the time-lag matrix to highlight repetitions...")
        chroma_sr = num_samples / song_length_sec
        smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
        time_lag_similarity.denoise(time_time_similarity.matrix, smoothing_size_samples)

        # Step 6: Detect repeating lines (potential choruses)
        print("Step 6: Detecting repeating line segments in the matrix...")
        clip_length = 10
        clip_length_samples = clip_length * chroma_sr
        candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
        lines = detect_lines(time_lag_similarity.matrix, candidate_rows, clip_length_samples)

        if not lines:
            print("   > No repeating segments were detected. Cannot determine chorus. Skipping song.")
            return

        print(f"   > Detected {len(lines)} potential repeating segments.")

        # Step 7: Score and sort lines to find chorus candidates
        print("Step 7: Scoring lines based on overlaps to find chorus candidates...")
        line_scores = count_overlapping_lines(
            lines, OVERLAP_PERCENT_MARGIN * clip_length_samples,
            clip_length_samples)
        choruses = sorted_segments(line_scores)
        
        unsorted_chorus_times = [(c[0].start / chroma_sr, c[0].end / chroma_sr) for c in choruses]
        unsorted_chorus_times.sort(key=lambda x: x[0])
        chorus_times = []
        if unsorted_chorus_times:
            chorus_times.append(unsorted_chorus_times[0])
            for i in range(1, len(unsorted_chorus_times)):
                if (unsorted_chorus_times[i][0] - chorus_times[-1][0]) >= clip_length:
                    chorus_times.append(unsorted_chorus_times[i])

        # Step 8: Find the best chorus based on rhythmic intensity (onsets)
        print("Step 8: Identifying the best chorus candidate using onset detection...")
        max_onset = 0
        best_chorus = []
        for time in chorus_times:
            if 10 <= (time[1] - time[0]) <= 30:
                chorus_wave_data = song_wav_data[int(time[0]*sr) : int(time[1]*sr)]
                onset_detect = librosa.onset.onset_detect(y=chorus_wave_data, sr=sr)
                if np.mean(onset_detect) >= max_onset:
                    max_onset = np.mean(onset_detect)
                    best_chorus = chorus_wave_data
        
        if len(best_chorus) == 0:
            print("   > Could not identify a suitable best chorus candidate. Skipping song.")
            return
        
        print("   > Best chorus candidate identified.")
        chorus_mfcc = np.average(librosa.feature.mfcc(y=best_chorus, sr=sr), axis=0)

        # Step 9: Compare all segments to the chorus using DTW
        print("Step 9: Comparing all segments to the identified chorus using Dynamic Time Warping (DTW)...")
        structure_labels = [""] * len(new_labels)
        max_dist = 0
        min_dist = 100
        similarity_measures = []
        euclidean_norm = lambda x, y: np.abs(x - y)
        for x in range(len(new_boundaries)):
            dist = fastdtw(mfccs[x], chorus_mfcc, dist=euclidean_norm)
            similarity_measures.append(dist)
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist

        # Step 10: Normalize scores and label chorus sections
        print("Step 10: Normalizing similarity scores and identifying chorus sections...")
        # Handle case where all distances are the same to avoid division by zero
        max_sim = max(similarity_measures) if similarity_measures else 1.0
        if max_sim == 0: max_sim = 1.0 # Avoid division by zero
        normalized = [float(i) / max_sim for i in similarity_measures]
        sorted_norms = sorted(normalized)
        
        bottom = []
        if max_dist - min_dist <= 2:
            bottom = sorted_norms[:int(len(sorted_norms) * .5)]
        else:
            bottom = sorted_norms[:int(len(sorted_norms) * .40)]

        # Ensure 'bottom' is not empty before accessing the last element
        if bottom:
            for x in range(len(structure_labels)):
                if normalized[x] <= bottom[-1]:
                    structure_labels[x] = "chorus"
        
        chorus_count = structure_labels.count("chorus")
        print(f"   > Identified {chorus_count} chorus sections.")

        # Step 11: Label remaining sections
        print("Step 11: Labeling remaining non-chorus sections (verse, intro, outro, etc.)...")
        for x in range(len(structure_labels)):
            found_match = False
            for y in range(x + 1, len(structure_labels)):
                if (new_labels[x] == new_labels[y]) and structure_labels[y] == ""  and structure_labels[x] == "":
                    found_match = True
                    structure_labels[x] = "verse"
                    structure_labels[y] = "verse"
            if not found_match and structure_labels[x] == "":
                if x == 0:
                    structure_labels[x] = "intro"
                elif x == (len(new_boundaries) - 1):
                    structure_labels[x] = "outro"
                else:
                    structure_labels[x] = "transition"
        print("   > All sections labeled.")

        # Step 12: Write labels to file
        output_filename = audio_path.stem + "_labels.txt"
        output_path = labels_dir / output_filename
        print(f"Step 12: Writing final labels to {output_path}...")
        with open(output_path, "w") as frames:
            for e in range(len(new_boundaries)):
                if e < len(new_boundaries) - 1:
                    outer_bound = e + 1
                    frames.write(f"{round(new_boundaries[e])}\t{round(new_boundaries[outer_bound])}\t{structure_labels[e]}\n")
                else:
                    frames.write(f"{round(new_boundaries[e])}\t{round(song_length_sec)}\t{structure_labels[e]}\n")
        
        print(f"===== FINISHED: {file_name} =====")

    except Exception as e:
        print(f"!!!!!! An error occurred while processing {file_name}: {e} !!!!!!")
        print("!!!!!! Skipping to the next song. !!!!!!")

# ==============================================================================
# === SCRIPT ENTRY POINT =======================================================
# ==============================================================================

def main():
    """
    Main function to find and process all audio files in the 'audio' directory.
    """
    audio_dir = Path("audio")
    labels_dir = Path("labels")

    # Create directories if they don't exist
    audio_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    print("Starting structural analysis process.")
    print(f"Looking for audio files in: {audio_dir.resolve()}")

    # ✅ IMPROVEMENT: Filter for common audio file types to avoid errors
    supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
    all_files = list(audio_dir.glob('*'))
    audio_files = [f for f in all_files if f.suffix.lower() in supported_extensions]

    # ✅ IMPROVEMENT: Add a user-friendly check if no audio files are found.
    if not audio_files:
        print("\n!! No audio files found in the 'audio' directory. !!")
        print("Please add some .mp3, .wav, or other supported audio files and run the script again.")
        return

    print(f"\n--- Found {len(audio_files)} audio file(s) to process. ---")

    for audio_path in audio_files:
        process_song(audio_path, labels_dir)
    
    print("\n--- All songs have been processed. ---")


if __name__ == "__main__":
    main()