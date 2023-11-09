import cv2
import os
import keyboard

# Tải classifier cho việc phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở video
video_capture = cv2.VideoCapture('5/5.mp4')

# Tạo thư mục đầu ra nếu nó chưa tồn tại
output_directory = '3/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Đọc video frame từng frame
frame_count = 0
i = 0  # Gán giá trị cho biến 'i'
while True:
    ret, frame = video_capture.read()

    # Kiểm tra nếu đã đọc hết video
    if not ret:
        break

    # Phát hiện khuôn mặt trong frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Với mỗi khuôn mặt được phát hiện, cắt ảnh và lưu vào thư mục đầu ra
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y + h, x:x + w]
        # Tạo tên tệp ảnh khuôn mặt với định dạng "anh_khuon_mat_{frame_count}_{i}_{ten_video}.jpg"
        output_path = os.path.join(output_directory, f"anh_khuon_mat_{frame_count}_{i}_{'5.mp4'}.jpg")
        cv2.imwrite(output_path, cropped_face)  # Lưu ảnh khuôn mặt
        frame_count += 1

    # Kiểm tra nút 'q' trên bàn phím để thoát khỏi vòng lặp
    if keyboard.is_pressed('q'):
        break

# Giải phóng tài nguyên
video_capture.release()
