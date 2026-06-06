import cv2
import numpy as np
import joblib
import sys
import warnings
from ultralytics import YOLO

# 1. Ẩn các cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Tải mô hình KNN đoán màu của bạn lên
model_path = 'color_model.pkl'
try:
    color_model = joblib.load(model_path)
    print("----> Tải mô hình KNN thành công!")
except:
    print(f"LỖI: Không tìm thấy file '{model_path}' trong thư mục hiện tại!")
    sys.exit()

# 3. Tải mô hình YOLOv8 tự động tìm vật thể (Sẽ tự tải file .pt về trong lần đầu chạy)
print("----> Đang khởi tạo mô hình YOLOv8 tự động tìm xe...")
yolo_model = YOLO('yolov8n.pt') 

# 4. Đọc bức ảnh bãi đỗ xe mới bất kỳ của bạn
image_path = 'anh3.jpg' # Thay bằng tên ảnh mới của bạn tại đây
image = cv2.imread(image_path)
if image is None:
    print(f"LỖI: Không tìm thấy file ảnh '{image_path}'!")
    sys.exit()

# Tạo bản sao ảnh HSV để trích xuất màu sắc chuẩn xác nhất
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("\n--- ĐANG TỰ ĐỘNG QUÉT VÀ NHẬN DIỆN MÀU XE BẰNG YOLO + KNN ---")

# 5. Dùng YOLOv8 tự động tìm tất cả các phương tiện giao thông trong ảnh
results = yolo_model(image)[0]

count_car = 0

# Duyệt qua từng vật thể mà YOLO tìm thấy
for box in results.boxes:
    # Lấy ID của lớp vật thể (Class ID)
    class_id = int(box.cls[0])
    
    # YOLOv8 định nghĩa: class 2 là 'car' (ô tô), class 7 là 'truck' (xe tải), class 5 là 'bus' (xe buýt)
    if class_id in [2, 5, 7]:
        count_car += 1
        
        # Tự động lấy tọa độ góc (xmin, ymin, xmax, ymax) từ YOLO
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        
        # Cắt (Crop) vùng ảnh chiếc xe tự động dựa trên tọa độ YOLO trả về
        crop_car = hsv_image[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        if crop_car.size == 0:
            continue
            
        # 6. Trích xuất màu sắc trung vị (Median) giống thuật toán cũ của bạn
        avg_h = np.median(crop_car[:, :, 0])
        avg_s = np.median(crop_car[:, :, 1])
        avg_v = np.median(crop_car[:, :, 2])
        
        # Đưa vào KNN để dự đoán màu sắc xe
        predicted_color = color_model.predict([[avg_h, avg_s, avg_v]])[0]
        
        # In kết quả tự động ra Console
        print(f"Xe thứ {count_car}: Vị trí [{int(xmin)}, {int(ymin)}] -> Màu dự đoán: {predicted_color}")
        
        # 7. Vẽ khung hình và viết chữ kết quả trực quan lên màn hình
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(image, f"Xe {count_car}: {predicted_color}", (int(xmin), int(ymin) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"\n---> HOÀN TẤT: YOLO đã tự động tìm thấy {count_car} chiếc xe!")

# 8. Hiển thị kết quả ra màn hình
cv2.namedWindow('Ket qua Tu dong YOLO + KNN', cv2.WINDOW_NORMAL)
cv2.imshow('Ket qua Tu dong YOLO + KNN', image)
cv2.waitKey(0)
cv2.destroyAllWindows()