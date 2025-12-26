import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import joblib
import os

def detect_blur_laplacian(image_path, threshold=100):
    try:
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return f"Error loading image: {str(e)}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    if variance < threshold:
        return f"Blurry (Laplacian Variance: {variance:.2f})"
    else:
        return f"Sharp (Laplacian Variance: {variance:.2f})"

def detect_blur_fft(image_path, threshold=0.5):
    try:
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return f"Error loading image: {str(e)}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    low_freq = magnitude_spectrum[crow - 10:crow + 10, ccol - 10:ccol + 10].mean()
    high_freq = magnitude_spectrum.mean()
    ratio = low_freq / high_freq if high_freq != 0 else 0
    if ratio > threshold:
        return f"Blurry (FFT Ratio: {ratio:.2f})"
    else:
        return f"Sharp (FFT Ratio: {ratio:.2f})"

def detect_blur_ml(image_path):
    if not os.path.exists('model.pkl'):
        return "Model not found. Train the model first."
    try:
        model = joblib.load('model.pkl')
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return f"Error loading image: {str(e)}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    low_freq = magnitude_spectrum[crow - 10:crow + 10, ccol - 10:ccol + 10].mean()
    high_freq = magnitude_spectrum.mean()
    fft_ratio = low_freq / high_freq if high_freq != 0 else 0
    features = np.array([[lap_var, fft_ratio]])
    prediction = model.predict(features)[0]
    labels = ['Sharp', 'Defocused Blurred', 'Motion Blurred']
    return f"{labels[prediction]} (ML Prediction)"

class BlurDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blur Detection App")
        
        self.label = tk.Label(root, text="Select an image to check for blur")
        self.label.pack(pady=10)
        
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)
        
        self.method_var = tk.StringVar(value="laplacian")
        self.method_frame = ttk.LabelFrame(root, text="Detection Method")
        self.method_frame.pack(pady=10)
        
        ttk.Radiobutton(self.method_frame, text="Laplacian Variance", variable=self.method_var, value="laplacian").pack(anchor=tk.W)
        ttk.Radiobutton(self.method_frame, text="FFT Analysis", variable=self.method_var, value="fft").pack(anchor=tk.W)
        ttk.Radiobutton(self.method_frame, text="Machine Learning", variable=self.method_var, value="ml").pack(anchor=tk.W)
        
        self.threshold_label = tk.Label(root, text="Threshold:")
        self.threshold_label.pack()
        self.threshold_var = tk.DoubleVar(value=100)
        self.threshold_slider = tk.Scale(root, from_=0, to=1000, orient=tk.HORIZONTAL, variable=self.threshold_var)
        self.threshold_slider.pack()
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            method = self.method_var.get()
            threshold = self.threshold_var.get()
            if method == "laplacian":
                result = detect_blur_laplacian(file_path, threshold)
            elif method == "fft":
                result = detect_blur_fft(file_path, threshold / 1000)  # Normalize for FFT
            elif method == "ml":
                result = detect_blur_ml(file_path)
            self.result_label.config(text=result)
            
            # Display the image
            image = Image.open(file_path)
            image = image.resize((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = BlurDetectionApp(root)
    root.mainloop()