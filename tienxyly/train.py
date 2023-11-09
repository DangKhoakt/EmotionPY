import cv2
import os
import numpy as np
from PIL import Image

# Khai báo đường dẫn đến thư mục chứa dữ liệu hình ảnh khuôn mặt và nhãn tương ứng
data_dir = 'data'
recognizer = cv2.face_LBPHFaceRecognizer.create() if cv2.__version__.startswith('3') else cv2.face_LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Hàm để lấy danh sách các hình ảnh khuôn mặt và nhãn từ thư mục dữ liệu
def get_images_and_labels(data_dir):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    face_samples = []
    labels = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Mở hình ảnh và chuyển sang dạng grayscale
        img_numpy = np.array(img, 'uint8')

        label = int(os.path.split(image_path)[-1].split(".")[0])  # Nhãn từ tên tệp
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            labels.append(label)

    return face_samples, labels

print("Đang huấn luyện mô hình nhận diện khuôn mặt...")
faces, labels = get_images_and_labels(data_dir)
recognizer.train(faces, np.array(labels))

# Lưu mô hình đã huấn luyện
recognizer.save('face_recognizer.yml')
print("Hoàn thành việc huấn luyện. Mô hình đã được lưu tại 'face_recognizer.yml'")
