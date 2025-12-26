import cv2
import sys

def detect_blur(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    threshold = 100
    if variance < threshold:
        print("Image is blurry (Variance: {:.2f})".format(variance))
    else:
        print("Image is sharp (Variance: {:.2f})".format(variance))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_blur(sys.argv[1])
    else:
        print("Usage: python blur_detect.py <image_path>")