# BTTH_02-DS102_24521820
Bài tập thực hành 02 lớp DS102.R11.CNVN.1

# Thông tin người thực hiện
Họ và Tên: Phan Thị Thùy Trang

MSSV: 24521820

Email: 24521820@gm.uit.edu.vn

# Bố cục tổ chức folder
`data`: Lưu dữ liệu được sử dụng trong bài thực hành (dữ liệu crawl từ BTTH 01)

`docs`: Lưu yêu cầu thực hiện của BTTH 02

`src`: Lưu tất cả file thực hiện yêu cầu của BTTH 02

`visualize`: Lưu thông tin giá trị loss khi thực hiện các yêu cầu

# Nội dung bài thực hành
## 1. Chuẩn bị dữ liệu với dữ liệu từ crawl từ BTTH 01
Dữ liệu được lưu trong folder [data](https://drive.google.com/drive/folders/1ZOIBykFS9hQNXgOSL7cP2ZHCaZjgYEwL?usp=sharing)
Quy tắc lưu dữ liệu: tên folder là nhãn cho tất cả ảnh có trong folder
Dữ liệu bao gồm 3 folder chính, trong đó `train` và `test` được tạo ra khi chạy file `split_data.py`:
1. Dữ liệu thô được cào từ BTTH 01: `raw`
2. Dữ liệu train cho mô hình: `train`
3. Dữ liệu test cho mô hình: `test`

Ngoài ra còn có các file xử lý dữ liệu và rút trích đặc trưng ảnh
1. Đọc dữ liệu ảnh và nhãn tương ứng bằng hàm LoadImagesFromFolder: `data_utils.py`
2. Rút trích đặc trưng của từng ảnh bằng cách kết hợp đặc trưng Histogram HSV và đặc trưng HOG: `feature_extraction.py`

## 2. Assignment 1 - Sử dụng Logistic Regression tự định nghĩa
Bài làm được chia làm 2 file: 
1. Mô hình Logistic Regression tự định nghĩa: `logistic_regression.py`
2. Thực hiện train và đánh giá mô hình Logistic Regression tự định nghĩa: `assignment_1.py`
3. Giá trị loss thay đổi sau mỗi epoch lưu trong `loss_history_logistic.npy`
- Thực hiện train và dự đoán trên 2 nhãn: architecture và art

## 3. Assignment 2 - Sử dụng Softmax Regression tự định nghĩa
Bài làm được chia làm 2 file: 
1. Mô hình Softmax Regression tự định nghĩa: `softmax_regression.py`
2. Thực hiện train và đánh giá mô hình Softmax Regression tự định nghĩa: `assignment_2.py`
3. Giá trị loss thay đổi sau mỗi epoch lưu trong `loss_history_softmax.npy`

## 4. Assignment 3 - Sử dụng Logistic Regression và Softmax Regression trong Machine Learning library
Bài làm được lưu trong file: `assignment_3.py`
- Thực hiện train và đánh giá mô hình Logistic và Softmax Regression trong thư viện Scikit Learn
- Giá trị loss thay đổi sau mỗi epoch của mỗi mô hình được lưu trong `library_loss_history_logistic.npy` và `library_loss_history_softmax.npy`
