import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Định nghĩa model CNN
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
            nn.Linear(32 * 128 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output = 4 classes
        )

    def forward(self, x):
        return self.network(x)

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Định nghĩa transform cho ảnh grayscale
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Tạo dataset và dataloader
data_dir = 'dataset'  # Thư mục chứa dữ liệu training
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Khởi tạo model
model = CNN_Network().to(device)
print("Model architecture:")
print(model)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hàm train
def train(epochs):
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print statistics
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Lưu model sau mỗi epoch
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f'Saved model checkpoint: model_epoch_{epoch+1}.pth')
    
    # Lưu model cuối cùng
    torch.save(model.state_dict(), 'model_final.pth')
    print('\nTraining completed!')
    print('Final model saved as: model_final.pth')

if __name__ == '__main__':
    # Train model
    train(epochs=10) 