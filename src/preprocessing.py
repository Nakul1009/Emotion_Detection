import os
import shutil


emotion_map = {
    "01": "neutral",
    "02": "positive",
    "03": "positive",
    "04": "negative",
    "05": "negative",
    "06": "negative",
    "07": "negative",
    "08": "positive"
}

RAW_DIR = "D:\\PROJECTS\\Emotion Detector\\Emotion_Detection\\data\\raw"
PROC_DIR = "D:\\PROJECTS\\Emotion Detector\\Emotion_Detection\\data\\preprocessed"

os.makedirs(PROC_DIR, exist_ok=True)

def map_emotion(filename):
    code = filename.split("-")[2]  # 3rd segment = emotion
    return emotion_map.get(code, "unknown")

def organize_files():
    for ff in os.listdir(RAW_DIR):
        ndir = RAW_DIR+f"\\{ff}"
        for f in os.listdir(ndir):
            print(f)
            if f.endswith(".wav"):
                label = map_emotion(f)
                label_folder = os.path.join(PROC_DIR, label)
                os.makedirs(label_folder, exist_ok=True)
                shutil.copy(os.path.join(RAW_DIR+f"\\{ff}", f), label_folder)

if __name__ == "__main__":
    organize_files()
    print("Files organized into Positive/Negative/Neutral folders!")
