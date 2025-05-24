# Brain-Tumor-Classification
Pretrain file: [here](https://drive.google.com/drive/folders/13JpLvz-9tb1p2znNg7EM0-Vnwbq71fma?usp=drive_link)

## Tổng quan chức năng

Ứng dụng web này cho phép người dùng tải lên ảnh MRI não và sử dụng ba mô hình học sâu (CNN, ResNet50, VGG16) để phân loại khối u não thành 4 loại:
- U nguyên bào thần kinh đệm
- U màng não
- Không có khối u
- U tuyến yên

Kết quả dự đoán của từng mô hình sẽ được hiển thị cùng với độ tin cậy.

## Hướng dẫn cài đặt thư viện

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

