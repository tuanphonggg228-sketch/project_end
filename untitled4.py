import cv2
import pandas as pd
import os

# Tên file dữ liệu muốn lưu
csv_file = 'color_dataset.csv'

# Nếu file chưa tồn tại, tạo mới với các cột H, S, V, Color_Name
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['H', 'S', 'V', 'Color_Name'])
    df.to_csv(csv_file, index=False)

# Đọc ảnh mẫu của bạn
image = cv2.imread('anh48.jpg')
# Chuyển sẵn sang không gian màu HSV để lấy giá trị cho chính xác
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lấy giá trị H, S, V tại điểm click chuột
        hsv_pixel = hsv_image[y, x]
        h = hsv_pixel[0]
        s = hsv_pixel[1]  
        v = hsv_pixel[2]
        
        print(f"\nGiá trị vừa click - H:{h}, S:{s}, V:{v}")
        # Yêu cầu bạn gõ tên màu tương ứng ở màn hình console
        color_name = input("Nhập tên màu cho vùng này (ví dụ: White, Black, Red, Silver...): ")
        
        if color_name.strip() != "":
            # Lưu vào file CSV
            df_new = pd.DataFrame([[h, s, v, color_name]], columns=['H', 'S', 'V', 'Color_Name'])
            df_new.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"--> Đã lưu thành công màu '{color_name}' vào file {csv_file}")

# Mở cửa sổ hiển thị ảnh để click
cv2.namedWindow('Chon Mau Data', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Chon Mau Data', click_event)

print("HƯỚNG DẪN: Click chuột trái vào các con xe để lấy màu. Nhập tên màu bên màn hình Console. Ấn phím bất kỳ trên ảnh để THOÁT.")
cv2.imshow('Chon Mau Data', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

