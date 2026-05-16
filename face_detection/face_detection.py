"""
Face detection using OpenCV's pretrained Haar Cascade classifier on a local image file.
Loads image from the path given as a command-line argument, converts to grayscale, applies
haarcascade_frontalface_default.xml via detectMultiScale, draws blue bounding boxes around
each detected face, prints total face count, saves annotated output as <name>_detected.png,
and reruns twice with alternative scaleFactor/minNeighbors to demonstrate sensitivity tuning.
Usage: python face_detection.py <image_path>
"""

import sys
import cv2
import matplotlib.pyplot as plt


def detect_faces(image_path, scale_factor=1.1, min_neighbors=4):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    print(f"scaleFactor={scale_factor}  minNeighbors={min_neighbors}  →  {len(faces)} face(s) detected")

    annotated = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(
        f"Detected {len(faces)} face(s)  |  "
        f"scaleFactor={scale_factor}, minNeighbors={min_neighbors}"
    )
    plt.tight_layout()

    base = image_path.rsplit('.', 1)[0]
    out_path = f"{base}_sf{scale_factor}_mn{min_neighbors}_detected.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_detection.py <image_path>")
        print("Example: python face_detection.py photo.jpg")
        sys.exit(1)

    path = sys.argv[1]

    print("=== Default Parameters ===")
    detect_faces(path, scale_factor=1.1, min_neighbors=4)

    print("=== Stricter (fewer false positives) ===")
    detect_faces(path, scale_factor=1.2, min_neighbors=6)

    print("=== More Sensitive (catches more faces) ===")
    detect_faces(path, scale_factor=1.05, min_neighbors=3)
