# Brain-Tumor-Classification
Tải các file lưu trọng số pretrain ở [đây](https://github.com/dangth2004/Brain-Tumor-Classification/tree/main/pretrain_parameters)

## Mô tả dữ liệu
Dữ liệu là các ảnh chụp MRI não dùng cho bài toán phân loại khối u não được lấy từ [đây](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Dữ liệu bao gồm 5712 ảnh thuộc tập huấn luyện, 1311 ảnh thuộc tập kiểm tra chia làm 4 phân lớp:
- U nguyên bào thần kinh đệm
- U màng não
- Không có khối u
- U tuyến yên

Đối với các ảnh thuộc tập huấn luyện, chúng tôi sẽ thực hiện tăng cường dữ liệu bằng cách lật ngang ngẫu nhiên với tỷ lệ 0.6, xoay ngẫu nhiên 30 độ. Ngoài ra, tất cả ảnh đều được resize về kích thước $224 \times 224$, chuyển về ảnh xám, chuẩn hóa giá trị pixel với kỳ vọng $\mu = 0.5$, độ lệch chuẩn là 0.5.

## Mô hình sử dụng
Chúng tôi huấn luyện và sử dụng 4 mô hình để giải quyết bài toán:
- Mô hình CNN tự xây dựng
- Mô hình encoder - decoder đơn giản với encoder là mô hình CNN ở trên, decoder là mô hình LSTM
- Mô hình ResNet-50
- Mô hình VGG-16

Kịch bản thử nghiệm:
- Kịch bản 1: Huấn luyện tất cả các tầng của các mô hình với trọng số khởi tạo ngẫu nhiên
- Kịch bản 2: Huấn luyện tất cả các tầng của ResNet và VGG với trọng số khởi tạo được lấy từ pretrained = True (full fine-tune)

## Tổng quan chức năng web demo

Ứng dụng web này cho phép người dùng tải lên ảnh MRI não và sử dụng các mô hình học sâu (CNN, ResNet50, VGG16) để phân loại khối u não thành 4 loại:
- U nguyên bào thần kinh đệm
- U màng não
- Không có khối u
- U tuyến yên

Kết quả dự đoán của từng mô hình sẽ được hiển thị cùng với độ tin cậy.

## Hướng dẫn thực hiện demo 

1. **Tạo môi trường ảo (khuyến nghị):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Linux/Mac
   venv\Scripts\activate     # Trên Windows
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

  

## Cách chạy ứng dụng

1. **Tải các file trọng số mô hình (pretrain) tại link ở trên và đặt vào thư mục `pretrain_para` đúng như đường dẫn trong code.**

2. **Chạy ứng dụng Flask:**
   ```bash
   python app.py
   ```

3. **Truy cập ứng dụng tại địa chỉ:**  
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. **Tải ảnh MRI não lên và xem kết quả dự đoán.**

