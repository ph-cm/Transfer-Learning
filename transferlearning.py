import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np
import os
import glob
import pytorchcv
import torch.optim as optim


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    criterion = torch.nn.NLLLoss()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def plot_results(train_loss, val_loss, epochs):
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



def check_image_dir(pattern):
    images = glob.glob(pattern)
    if not images:
        print(f"No images found for pattern: {pattern}")
    else:
        print(f"Found {len(images)} images for pattern: {pattern}")

def display_dataset(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.permute(1, 2, 0))  # Transpor canais para exibição
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()


check_image_dir(r'C:\Users\phenr\Downloads\kagglecatsanddogs_5340\PetImages\Dog\*.jpg')
check_image_dir(r'C:\Users\phenr\Downloads\kagglecatsanddogs_5340\PetImages\Cat\*.jpg')

from PIL import Image

image_dir = r'C:\Users\phenr\Downloads\kagglecatsanddogs_5340\PetImages\Dog'
invalid_files = []

for file in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file)
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verifica se o arquivo é uma imagem válida
    except Exception as e:
        print(f"Invalid image file: {file_path} - {e}")
        invalid_files.append(file_path)

print(f"Found {len(invalid_files)} invalid files.")

for file_path in invalid_files:
    os.remove(file_path)
print("Invalid files removed.")


std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        std_normalize])
dataset = torchvision.datasets.ImageFolder(r'C:\Users\phenr\Downloads\kagglecatsanddogs_5340\PetImages',transform=trans)
trainset, testset = torch.utils.data.random_split(dataset,[20000,len(dataset)-20000])

display_dataset(dataset)

#Pre-trained models
vgg = torchvision.models.vgg16(pretrained=True)
sample_image = dataset[0][0].unsqueeze(0)
res = vgg(sample_image)
print(res[0].argmax())

import json, requests
class_map = json.loads(requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json").text)
class_map = { int(k) : v for k,v in class_map.items() }

class_map[res[0].argmax().item()]

summary(vgg,input_size=(1,3,224,224))

#GPU computations

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Doing computations on device = {}'.format(device))

vgg.to(device)
sample_image = sample_image.to(device)

vgg(sample_image).argmax()

#Extracting VGG features
res = vgg.features(sample_image).cpu()
plt.figure(figsize=(15,3))
plt.imshow(res.detach().view(512,-1).T)
print(res.size())
plt.show()

bs = 8
dl = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True)
num = bs*100
feature_tensor = torch.zeros(num,512*7*7).to(device)
label_tensor = torch.zeros(num).to(device)
i = 0
for x,l in dl:
    with torch.no_grad():
        f = vgg.features(x.to(device))
        feature_tensor[i:i+bs] = f.view(bs,-1)
        label_tensor[i:i+bs] = l
        i+=bs
        print('.',end='')
        if i>=num:
            break

vgg_dataset = torch.utils.data.TensorDataset(feature_tensor,label_tensor.to(torch.long))
train_ds, test_ds = torch.utils.data.random_split(vgg_dataset,[700,100])

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=32)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=32)

net = torch.nn.Sequential(torch.nn.Linear(512*7*7,2),torch.nn.LogSoftmax(dim=1)).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

history = train(net, train_loader, test_loader, optimizer, device)

print(vgg)

vgg.classifier = torch.nn.Linear(25088,2).to(device)

for x in vgg.features.parameters():
    x.requires_grad = False

summary(vgg,(1, 3,244,244))

trainset, testset = torch.utils.data.random_split(dataset,[20000,len(dataset)-20000])
train_loader = torch.utils.data.DataLoader(trainset,batch_size=16)
test_loader = torch.utils.data.DataLoader(testset,batch_size=16)

def train_long(model, train_loader, test_loader, loss_fn, epochs=10, print_freq=90):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_freq == 0:  # Log a cada 'print_freq' batches
                print(f"[Epoch {epoch+1}, Batch {i}] Loss: {running_loss / (i+1):.4f}")
        
        print(f"Epoch {epoch+1} completed. Loss: {running_loss / len(train_loader):.4f}")
    
    print("Training completed.")

train_long(vgg, train_loader, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), epochs=1, print_freq=90)
