import os
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR = "D:\PROJECTS\Emotion Detector\Emotion_Detection\data\preprocessed"
OUTPUT_DIR = "D:\PROJECTS\Emotion Detector\Emotion_Detection\data\processed_clean`"
TARGET_SR = 16000  


def preprocess_audio(file_path, out_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)


    audio = audio / np.max(np.abs(audio) + 1e-9)


    trimmed, _ = librosa.effects.trim(audio, top_db=25)


    sf.write(out_path, trimmed, TARGET_SR)


def process_all():
    for emotion in os.listdir(INPUT_DIR):
        in_folder = os.path.join(INPUT_DIR, emotion)
        out_folder = os.path.join(OUTPUT_DIR, emotion)
        os.makedirs(out_folder, exist_ok=True)

        for file in os.listdir(in_folder):
            if file.endswith(".wav"):
                in_path = os.path.join(in_folder, file)
                out_path = os.path.join(out_folder, file)

                preprocess_audio(in_path, out_path)

        print(f"[✔] {emotion} cleaned successfully!")


if __name__ == "__main__":
    process_all()
    print("\n🔥 All audio cleaned & saved in data/processed_clean!")
