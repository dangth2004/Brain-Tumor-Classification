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

# --- Định nghĩa các model giống notebook ---

# CNN tự xây dựng
class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(in_features=32 * 56 * 56, out_features=56),
            nn.ReLU()
        )
        self.classifier_fc = nn.Linear(in_features=56, out_features=4)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        feature_vector = self.feature_extractor_fc(x)
        output = self.classifier_fc(feature_vector)
        return output

# CNN + LSTM
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Classifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(in_features=32 * 56 * 56, out_features=56),
            nn.ReLU()
        )
        self.input_size_lstm = 56
        self.hidden_size_lstm = 128
        self.num_layers_lstm = 1
        self.num_classes = 4
        self.lstm = nn.LSTM(self.input_size_lstm, self.hidden_size_lstm, self.num_layers_lstm, batch_first=True)
        self.classifier_fc = nn.Linear(self.hidden_size_lstm, self.num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        feature_vector = self.feature_extractor_fc(x)
        x_lstm = feature_vector.unsqueeze(1)
        h0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        c0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm[:, -1, :]
        output = self.classifier_fc(out_lstm)
        return output

# ResNet-50 fine-tune
from torchvision.models import ResNet50_Weights, VGG16_Weights

class ResNet50_full(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet50_full, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            pretrained_weights = models.resnet50(weights=ResNet50_Weights.DEFAULT).conv1.weight
            grayscale_weights = pretrained_weights.mean(dim=1, keepdim=True)
            self.model.conv1.weight.copy_(grayscale_weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# VGG16 fine-tune
class VGG16_full(nn.Module):
    def __init__(self, num_classes=4, in_channels=1):
        super(VGG16_full, self).__init__()
        self.vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)  
        self.vgg16.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vgg16(x)

# --- Khởi tạo device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load trọng số các model ---
cnn_model = CNN_Network().to(device)
cnn_model.load_state_dict(torch.load(
    'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_parameters/CNN_Brain_Tumor_para.pth',
    map_location=device))
cnn_model.eval()

cnn_lstm_model = CNN_LSTM_Classifier().to(device)
cnn_lstm_model.load_state_dict(torch.load(
    'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_parameters/CNN_LSTM_Brain_Tumor_para.pth',
    map_location=device))
cnn_lstm_model.eval()

resnet50_full = ResNet50_full(num_classes=4).to(device)
resnet50_full.load_state_dict(torch.load(
    'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_parameters/fine_tune/ResNet_Full_Brain_Tumor_para.pth',
    map_location=device))
resnet50_full.eval()

vgg16_full = VGG16_full(num_classes=4, in_channels=1).to(device)
vgg16_full.load_state_dict(torch.load(
    'C:/Users/PC/PycharmProjects/pythonProject/hus/computer_vision/project/pretrain_parameters/fine_tune/VGG_Full_Brain_Tumor_para.pth',
    map_location=device))
vgg16_full.eval()

# --- Định nghĩa tên các classes ---
classes = ['u nguyên bào thần kinh đệm', 'u màng não', 'không có khối u', 'u tuyến yên']

# --- Tiền xử lý ảnh  ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

            # Dự đoán với CNN+LSTM
            with torch.no_grad():
                lstm_outputs = cnn_lstm_model(image_tensor)
                _, lstm_pred = torch.max(lstm_outputs, 1)
                lstm_prob = torch.nn.functional.softmax(lstm_outputs, dim=1)[0]
            lstm_result = {
                'class': classes[lstm_pred.item()],
                'probability': float(lstm_prob[lstm_pred].item())
            }

            # Dự đoán với ResNet50 fine-tune
            with torch.no_grad():
                resnet_outputs = resnet50_full(image_tensor)
                _, resnet_pred = torch.max(resnet_outputs, 1)
                resnet_prob = torch.nn.functional.softmax(resnet_outputs, dim=1)[0]
            resnet_result = {
                'class': classes[resnet_pred.item()],
                'probability': float(resnet_prob[resnet_pred].item())
            }

            # Dự đoán với VGG16 fine-tune
            with torch.no_grad():
                vgg_outputs = vgg16_full(image_tensor)
                _, vgg_pred = torch.max(vgg_outputs, 1)
                vgg_prob = torch.nn.functional.softmax(vgg_outputs, dim=1)[0]
            vgg_result = {
                'class': classes[vgg_pred.item()],
                'probability': float(vgg_prob[vgg_pred].item())
            }

            result = {
                'cnn': cnn_result,
                'cnn_lstm': lstm_result,
                'resnet50': resnet_result,
                'vgg16': vgg_result,
                'image_path': filepath
            }
            return jsonify(result)
        except Exception as e:
            print(f"Error details: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'})
    return None


if __name__ == '__main__':
    app.run(debug=True)
