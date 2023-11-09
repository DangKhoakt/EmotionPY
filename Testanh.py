import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model

# Tạo một mạng nơ-ron đơn giản với một lớp ẩn
model = Sequential()
model.add(Flatten(input_shape=(48, 48)))  # Làm phẳng hình ảnh đầu vào
model.add(Dense(128, activation='relu'))  # Lớp ẩn với 128 nơ-ron và hàm kích hoạt ReLU
model.add(Dense(4, activation='softmax'))  # Lớp đầu ra với 4 nơ-ron và hàm kích hoạt softmax (số lớp cảm xúc)

# Nạp trọng số của mô hình đã được huấn luyện từ tệp 'Emotion_Detection.h5'
model = load_model('F:\Py_emotion\Emotion_Detection.h5')

# Tạo một bộ phân lớp (Cascade Classifier) để phát hiện khuôn mặt trong hình ảnh
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Định nghĩa một từ điển 'labels_dict' để ánh xạ nhãn cảm xúc thành các chuỗi mô tả tương ứng
labels_dict = { 0:'', 1 :'Cuoi', 2: 'binhthong', 3: 'Buon'}

# Đọc hình ảnh từ tệp 'test2.jpg'
frame = cv2.imread("Anh1.jpg")

# Chuyển hình ảnh màu sang ảnh xám để tăng hiệu suất
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Sử dụng bộ phân lớp để phát hiện khuôn mặt trong ảnh
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

# Duyệt qua các khuôn mặt đã phát hiện trong ảnh
for x, y, w, h in faces:
    # Cắt ra ảnh khuôn mặt từ ảnh gốc
    sub_face_img = gray[y:y+h, x:x+w]

    # Thay đổi kích thước ảnh khuôn mặt thành 48x48 pixel
    resized = cv2.resize(sub_face_img, (48, 48))

    # Chuẩn hóa ảnh bằng cách chia cho 255.0 để đưa giá trị về khoảng [0, 1]
    normalize = resized / 255.0

    # Sử dụng mạng nơ-ron để dự đoán cảm xúc
    result = model.predict(normalize.reshape(1, 48, 48))

    # Chọn nhãn có giá trị cao nhất
    label = np.argmax(result)

    # In ra nhãn dự đoán
    print(label)

    # Vẽ hình chữ nhật xung quanh khuôn mặt
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

    # Hiển thị nhãn cảm xúc trên hình ảnh
    cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Hiển thị hình ảnh với các khuôn mặt đã phát hiện và nhãn cảm xúc
cv2.imshow("Frame", frame)

# Đợi cho đến khi một phím được nhấn, sau đó đóng cửa sổ hiển thị
cv2.waitKey(0)
cv2.destroyAllWindows()
