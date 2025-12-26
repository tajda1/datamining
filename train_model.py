import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    
    # FFT ratio
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    low_freq = magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10].mean()
    high_freq = magnitude_spectrum.mean()
    fft_ratio = low_freq / high_freq if high_freq != 0 else 0
    
    return [lap_var, fft_ratio]

def load_dataset(dataset_path):
    features = []
    labels = []
    for label, folder in enumerate(['sharp', 'defocused_blurred', 'motion_blurred']):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            continue
        for file in os.listdir(folder_path):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(folder_path, file)
                feat = extract_features(image_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    dataset_path = 'dataset'
    print(f"Loading dataset from {dataset_path}")
    
    # Debug: list contents
    if os.path.exists(dataset_path):
        print("Subfolders:", os.listdir(dataset_path))
        for sub in os.listdir(dataset_path):
            sub_path = os.path.join(dataset_path, sub)
            if os.path.isdir(sub_path):
                files = [f for f in os.listdir(sub_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                print(f"{sub}: {len(files)} files")
    else:
        print("Dataset path does not exist")
        exit(1)
    
    features, labels = load_dataset(dataset_path)
    print(f"Loaded {len(features)} samples")
    
    if len(features) == 0:
        print("No samples loaded. Check dataset structure.")
        exit(1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    joblib.dump(model, 'model.pkl')
    print("Model saved as model.pkl")