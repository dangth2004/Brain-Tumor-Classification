from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
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

# Khởi tạo device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo model CNN
cnn_model = CNN_Network().to(device)
try:
    cnn_model.load_state_dict(torch.load(
        'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_para/CNN_Brain_Tumor_para.pth',
        map_location=device))
    cnn_model.eval()
    print("CNN model loaded successfully!")
except Exception as e:
    print(f"Error loading CNN model: {str(e)}")
    raise

# Định nghĩa tên các classes
classes = ['u nguyên bào thần kinh đệm', 'u màng não', 'không có khối u', 'u tuyến yên']

# Khởi tạo model ResNet50
resnet50 = models.resnet50(weights=None)
pretrained_weights = resnet50.conv1.weight.data
resnet50.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet50.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
for param in resnet50.parameters():
    param.requires_grad = True
num_ftrs = resnet50.fc.in_features
resnet50.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 4)
)
try:
    resnet50.load_state_dict(torch.load(
        'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_para/Resnet50_Brain_Tumor_para.pth',
        map_location=device))
    resnet50.eval()
    print("ResNet50 model loaded successfully!")
except Exception as e:
    print(f"Error loading ResNet50 model: {str(e)}")
    raise
resnet50 = resnet50.to(device)

# Khởi tạo model VGG16
vgg16 = models.vgg16(weights=None)
vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
for param in vgg16.features.parameters():
    param.requires_grad = False
vgg16.classifier[6] = nn.Linear(4096, 4)
vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4)
)
try:
    vgg16.load_state_dict(torch.load(
        'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_para/VGG16_Brain_Tumor_para.pth',
        map_location=device))
    vgg16.eval()
    print("VGG16 model loaded successfully!")
except Exception as e:
    print(f"Error loading VGG16 model: {str(e)}")
    raise
vgg16 = vgg16.to(device)

# Định nghĩa transform cho ảnh (giống với test_transforms khi training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

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
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath)
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Dự đoán với CNN
            with torch.no_grad():
                cnn_outputs = cnn_model(image_tensor)
                _, cnn_pred = torch.max(cnn_outputs, 1)
                cnn_prob = torch.nn.functional.softmax(cnn_outputs, dim=1)[0]
            cnn_result = {
                'class': classes[cnn_pred.item()],
                'probability': float(cnn_prob[cnn_pred].item())
            }

            # Dự đoán với ResNet50
            with torch.no_grad():
                resnet_outputs = resnet50(image_tensor)
                _, resnet_pred = torch.max(resnet_outputs, 1)
                resnet_prob = torch.nn.functional.softmax(resnet_outputs, dim=1)[0]
            resnet_result = {
                'class': classes[resnet_pred.item()],
                'probability': float(resnet_prob[resnet_pred].item())
            }

            # Dự đoán với VGG16
            with torch.no_grad():
                vgg_outputs = vgg16(image_tensor)
                _, vgg_pred = torch.max(vgg_outputs, 1)
                vgg_prob = torch.nn.functional.softmax(vgg_outputs, dim=1)[0]
            vgg_result = {
                'class': classes[vgg_pred.item()],
                'probability': float(vgg_prob[vgg_pred].item())
            }

            result = {
                'cnn': cnn_result,
                'resnet50': resnet_result,
                'vgg16': vgg_result,
                'image_path': filepath
            }
            return jsonify(result)
        except Exception as e:
            print(f"Error details: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)