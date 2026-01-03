import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import time

# --- 1. CONFIGURATION ---
DATA_DIR = 'data/images'
MODEL_SAVE_PATH = 'models/resnet18_pcos.pth'
BATCH_SIZE = 32          # Reduce to 16 if you get "Out of Memory" errors
LEARNING_RATE = 0.001
EPOCHS = 10              # How many times to loop through all images

# Check Device (Force GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- 2. DATA PREPARATION ---
# We need to resize all images to 224x224 (Standard for ResNet)
# We also add "Data Augmentation" (flipping/rotating) to make the model smarter.
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet stats
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
print("Loading images...")
try:
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
except Exception as e:
    print(f"❌ Error loading images: {e}")
    print("Check if 'data/images' has subfolders like 'infected' and 'notinfected'")
    exit()

# Split: 80% Train, 20% Validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply the simpler transforms to validation set (overwrite the augmented ones)
val_dataset.dataset.transform = val_transforms 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Images Loaded. Train: {train_size}, Validation: {val_size}")
print(f"Classes: {full_dataset.classes}") # Should be ['infected', 'notinfected']

# --- 3. MODEL SETUP (Transfer Learning) ---
# Load pre-trained ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze early layers (so we don't destroy the pre-trained knowledge)
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (The "Classifier") to output 2 classes (PCOS vs Normal)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # 2 classes

model = model.to(device) # Move model to GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# --- 4. TRAINING LOOP ---
print("\n🔥 Starting Training...")
start_time = time.time()
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training Phase
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / train_size
    epoch_acc = correct / total
    
    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2%} | Val Acc: {val_acc:.2%}")
    
    # Save Best Model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   🎉 New Best Model Saved! ({val_acc:.2%})")

total_time = time.time() - start_time
print(f"\n✅ Training Complete in {total_time/60:.2f} minutes.")
print(f"Best Validation Accuracy: {best_acc:.2%}")