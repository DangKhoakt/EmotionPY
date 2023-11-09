from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.layers import Dropout
import warnings

# Định nghĩa đường dẫn thư mục chứa dữ liệu huấn luyện và kiểm tra
train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# Tạo một đối tượng ImageDataGenerator để tạo dữ liệu tăng cường cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

# Tạo một đối tượng ImageDataGenerator để chuẩn hóa ảnh kiểm tra
validation_datagen = ImageDataGenerator(rescale=1./255)

class_labels = ['Cuoi', 'Binhthuong', 'Buon']


# Tạo dữ liệu tăng cường cho tập huấn luyện bằng cách đọc dữ liệu từ thư mục huấn luyện
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    classes=class_labels)  # Sử dụng danh sách lớp cảm xúc

# Tạo dữ liệu kiểm tra bằng cách đọc dữ liệu từ thư mục kiểm tra
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    classes=class_labels)  # Sử dụng danh sách lớp cảm xúc

# Tạo một mô hình mạng nơ-ron tuần tự (Sequential)
model = Sequential()

# Làm phẳng các đặc trưng
model.add(Flatten(input_shape=(48, 48, 1)))


# Thêm lớp kết nối đầy đủ (Fully Connected Layer) với 512 đơn vị và hàm kích hoạt ReLU
model.add(Dense(512, activation='relu'))

# Thêm lớp Dropout để tránh overfitting
model.add(Dropout(0.2))

# Thêm lớp đầu ra với 3 đơn vị và hàm kích hoạt softmax cho phân loại đa lớp
model.add(Dense(3, activation='softmax'))

# Biên soạn mô hình với hàm mất mát và trình tối ưu hóa
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# In tổng quan của mô hình
print(model.summary())

# Đếm số ảnh trong thư mục huấn luyện và kiểm tra
train_path = "data/train"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

# Số lượt huấn luyện
epochs = 100

# Huấn luyện mô hình trên dữ liệu tạo bởi ImageDataGenerator
history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs // 32,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs // 32)

# Lưu mô hình đã huấn luyện vào một tệp H5
model.save('model_file.h5')


# plot the evolution of Loss and Acuracy on the train and validation sets

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()




warnings.filterwarnings("ignore", category=UserWarning)
