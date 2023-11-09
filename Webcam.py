import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Nạp mô hình đã được huấn luyện từ tệp 'Emotion_Detection.h5'
model = load_model('F:\Py_emotion\Emotion_Detection.h5')

# Tạo một đối tượng VideoCapture để truy cập luồng video từ webcam
video = cv2.VideoCapture(0)

# Tạo một bộ phân lớp (Cascade Classifier) để phát hiện khuôn mặt trong khung hình
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Định nghĩa một từ điển 'labels_dict' để ánh xạ nhãn của cảm xúc thành các chuỗi mô tả tương ứng
labels_dict = { 0:'', 1 :'Cuoi', 2: 'binhthong', 3: 'Buon'}

# Vào vòng lặp chính, thực hiện xử lý video từ webcam
while True:
    # Đọc một khung hình từ video
    ret, frame = video.read()

    # Chuyển ảnh màu sang ảnh xám để tăng hiệu suất
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sử dụng bộ phân lớp để phát hiện khuôn mặt trong ảnh
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    # Duyệt qua các khuôn mặt đã phát hiện
    for x, y, w, h in faces:
        # Cắt ra ảnh khuôn mặt từ khung hình
        sub_face_img = gray[y:y+h, x:x+w]

        # Thay đổi kích thước ảnh khuôn mặt thành 48x48 pixel
        resized = cv2.resize(sub_face_img, (48, 48))

        # Chuẩn hóa ảnh bằng cách chia cho 255.0 để đưa giá trị về khoảng [0, 1]
        normalized = resized / 255.0

        # Chuyển đổi ảnh thành định dạng phù hợp cho mô hình (1, 48, 48, 1)
        reshaped = normalized.reshape(1, 48, 48, 1)

        # Sử dụng mô hình để dự đoán cảm xúc
        result = model.predict(reshaped)
        
        # Chọn nhãn có giá trị cao nhất
        label = np.argmax(result)
        
        # In ra nhãn dự đoán trên khung hình
        print(label)
        
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        # Hiển thị nhãn cảm xúc trên hình ảnh
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Hiển thị khung hình với các phát hiện khuôn mặt và nhãn cảm xúc
    cv2.imshow("Frame", frame)

    # Chờ một phím được nhấn, và kiểm tra nếu phím 'q' được nhấn thì thoát khỏi vòng lặp
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ hiển thị
video.release()
cv2.destroyAllWindows()
