from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Định nghĩa model CNN (phải giống với model đã train)
class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Input channel = 1 (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 128 * 128, 128),  # Kích thước input phù hợp với 512x512
            nn.ReLU(),
            nn.Linear(128, 4)  # Output = 4 classes
        )

    def forward(self, x):
        return self.network(x)

# Khởi tạo model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Network().to(device)

# Load model đã train
try:
    model.load_state_dict(torch.load('CNN_Brain_Tumor_para.pth', map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Định nghĩa transform cho ảnh (giống với test_transforms khi training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Kích thước giống với training
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Định nghĩa tên các classes
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        try:
            # Lưu file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Đọc và xử lý ảnh
            image = Image.open(filepath)
            
            # In kích thước ảnh gốc để debug
            print(f"Original image size: {image.size}")
            
            # Áp dụng transform
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # In kích thước tensor để debug
            print(f"Input tensor shape: {image_tensor.shape}")
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                probability = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            # Lấy kết quả
            result = {
                'class': classes[predicted.item()],
                'probability': float(probability[predicted].item()),
                'image_path': filepath
            }
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Error details: {str(e)}")  # In chi tiết lỗi
            return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True) 