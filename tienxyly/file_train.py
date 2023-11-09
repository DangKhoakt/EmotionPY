import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Đường dẫn đến thư mục chứa dữ liệu hình ảnh và tên thư mục tương ứng với các người
data_directory = '3'
face_cascade_path = 'haarcascade_frontalface_default.xml'  # Đường dẫn đến file XML cho bộ phát hiện khuôn mặt

# Tải bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Tạo danh sách để lưu dữ liệu hình ảnh và nhãn tương ứng
data = []
labels = []

# Lặp qua các thư mục con trong thư mục chứa dữ liệu
for person_name in os.listdir(data_directory):
    person_directory = os.path.join(data_directory, person_name)
    
    # Kiểm tra xem đối tượng có phải là thư mục không
    if os.path.isdir(person_directory):
        # Lặp qua các tệp ảnh trong thư mục con
        for filename in os.listdir(person_directory):
            image_path = os.path.join(person_directory, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # Phát hiện khuôn mặt trong ảnh
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
            # Lặp qua các khuôn mặt được phát hiện trong ảnh
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))  # Thay đổi kích thước khuôn mặt nếu cần
                data.append(face.flatten())  # Biến đổi khuôn mặt thành vector
                labels.append(person_name)  # Nhãn là tên người

# Chuyển danh sách thành mảng NumPy
data = np.array(data)
labels = np.array(labels)

# Chuyển đổi nhãn thành dạng số sử dụng LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Kiểm tra xem có ít nhất một khuôn mặt được phát hiện
if len(data) > 0:
    # Tạo mô hình phân loại (ví dụ: Sử dụng máy vector hỗ trợ - SVM)
    classifier = SVC(kernel='linear')
    classifier.fit(data, labels_encoded)

    # Dự đoán và đánh giá mô hình
    y_pred = classifier.predict(data)
    accuracy = accuracy_score(labels_encoded, y_pred)
    print(f'Accuracy: {accuracy}')

    # Lưu mô hình tiền xử lý khuôn mặt và mô hình phân loại
    cv2.imwrite('preprocessing_model.png', face)
    joblib.dump(classifier, 'classifier_model.pkl')

    print('Mô hình đã được lưu thành công.')
else:
    print('Không có khuôn mặt được phát hiện trong dữ liệu.')
