import cv2
import numpy as np
import os

# Đường dẫn đến thư mục chứa các tệp ảnh
image_folder = '3/'

# Lặp qua tất cả các tệp ảnh trong thư mục
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # Tạo đường dẫn đầy đủ đến tệp ảnh
        image_path = os.path.join(image_folder, filename)

        # Đọc ảnh bằng OpenCV
        image = cv2.imread(image_path)

        # Kiểm tra xem ảnh có được đọc thành công không
        if image is not None:
            # Chuyển ảnh thành mảng NumPy và chuyển đổi thành ảnh grayscale (nếu cần)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Chuẩn hóa ảnh bằng hàm sigmoid
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            normalized_image = sigmoid(gray_image)

            # Chuyển đổi ảnh chuẩn hóa thành mảng NumPy
            normalized_image_array = np.array(normalized_image)

            # Bây giờ bạn có thể làm việc với mảng NumPy 'normalized_image_array'
            # Đảm bảo rằng mảng này nằm trong khoảng [0, 1]

            # Ví dụ: In giá trị pixel của mảng chuẩn hóa
            print(f'Giá trị pixel của ảnh chuẩn hóa ({filename}):')
            print(normalized_image_array)

            # Ví dụ: Lưu ảnh chuẩn hóa thành tệp PNG
            save_path = os.path.join('output_normalized_images', filename)
            cv2.imwrite(save_path, (normalized_image_array * 255).astype(np.uint8))  # Đảm bảo ảnh nằm trong khoảng [0, 255]

            print(f'Ảnh chuẩn hóa ({filename}) đã được lưu thành {save_path}')
        else:
            print(f'Không thể đọc ảnh ({filename}). Đảm bảo đường dẫn tới ảnh là chính xác.')
