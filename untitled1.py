import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Đọc dữ liệu từ file bạn vừa tạo
data = pd.read_csv('color_dataset.csv')

# 2. Tách các cột đầu vào (H, S, V) và nhãn đầu ra (Color_Name)
X = data[['H', 'S', 'V']]
y = data['Color_Name']

# 3. Chia dữ liệu thành 2 tập: 80% để học và 20% để kiểm tra độ chính xác
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Sử dụng thuật toán KNN với k=3 (tìm 3 điểm lân cận gần nhất)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. Kiểm tra thử xem mô hình đoán chuẩn bao nhiêu %
y_pred = model.predict(X_test)
print(f"Độ chính xác của mô hình màu sắc: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. LƯU MÔ HÌNH NÀY LẠI thành một file đóng go
# Sửa lại dòng 25 thành đường dẫn đầy đủ như sau:
joblib.dump(model, 'color_model.pkl')
print("Đã lưu mô hình thành công thành file 'color_model.pkl'!")