import os
import librosa
import numpy as np
import pandas as pd

CLEAN_DIR = "D:\PROJECTS\Emotion Detector\Emotion_Detection\data\processed_clean"
FEATURES_PATH = "D:\PROJECTS\Emotion Detector\Emotion_Detection\data\\features\\features.csv"
os.makedirs("D:\PROJECTS\Emotion Detector\Emotion_Detection\data\\features", exist_ok=True)

def extract_features(file_path, emotion):
    y, sr = librosa.load(file_path, sr=16000)

    # ----- Time Domain -----
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    # ----- Frequency Base -----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ----- Delta MFCC -----
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    # ----- Pitch / F0 -----
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.sum(pitches) > 0 else 0

    # ----- Spectral Shape -----
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    # ----- Chroma -----
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # ===== Combine Features =====
    features = np.hstack([
        zcr, rms, centroid, bandwidth, rolloff,
        contrast, pitch,
        mfcc_mean, mfcc_std, mfcc_delta_mean, chroma_mean
    ])
    
    return features




def generate_dataset():
    rows = []
    for emotion in os.listdir(CLEAN_DIR):
        folder = os.path.join(CLEAN_DIR, emotion)

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)

                features = extract_features(file_path, emotion)
                row = [file, emotion] + list(features)
                rows.append(row)

        print(f"[✔] Features extracted for {emotion}")

    # build column names
    columns = ["file", "label", "zcr", "rms", "centroid", "bandwidth", "rolloff", "contrast", "pitch"] + \
          [f"mfcc_mean_{i}" for i in range(20)] + \
          [f"mfcc_std_{i}" for i in range(20)] + \
          [f"mfcc_delta_{i}" for i in range(20)] + \
          [f"chroma_{i}" for i in range(12)]


    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\n Feature CSV saved → {FEATURES_PATH}")


if __name__ == "__main__":
    generate_dataset()
