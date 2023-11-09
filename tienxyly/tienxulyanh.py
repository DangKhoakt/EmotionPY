import cv2
import os

# Đường dẫn đến thư mục chứa hình ảnh
input_folder = '\1'

# Đường dẫn đến thư mục để lưu các khuôn mặt đã nhận diện
output_folder = '\2'

# Tạo thư mục đầu ra nếu nó chưa tồn tại
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Khởi tạo bộ phát hiện khuôn mặt của OpenCV (sử dụng haarcascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Duyệt qua tất cả các hình ảnh trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        
        # Đọc hình ảnh từ đường dẫn
        image = cv2.imread(input_path)
        
        # Chuyển hình ảnh sang ảnh grayscale để tăng hiệu suất
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt trong hình ảnh
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Duyệt qua tất cả các khuôn mặt đã phát hiện
        for i, (x, y, w, h) in enumerate(faces):
            # Cắt và lưu khuôn mặt đã nhận diện vào thư mục đầu ra
            face_image = image[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, f"{filename[:-4]}_face_{i}.jpg")
            cv2.imwrite(output_path, face_image)

print("Nhận diện và lưu khuôn mặt hoàn tất.")
