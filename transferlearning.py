import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision.models import vgg16
import torch.nn as nn
import torch.optim as optim


# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando no dispositivo: {device}")


# Verificar imagens no diretório
def check_image_dir(pattern):
    from glob import glob
    images = glob(pattern)
    if not images:
        print(f"Nenhuma imagem encontrada para o padrão: {pattern}")
    else:
        print(f"Encontradas {len(images)} imagens para o padrão: {pattern}")


# Verificar e limpar imagens inválidas
def clean_invalid_images(directory):
    invalid_files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verifica se o arquivo é uma imagem válida
        except Exception as e:
            print(f"Arquivo inválido: {file_path} - {e}")
            invalid_files.append(file_path)

    print(f"Encontrados {len(invalid_files)} arquivos inválidos.")
    for file_path in invalid_files:
        os.remove(file_path)
    print("Arquivos inválidos removidos.")


# Pré-processamento das imagens
std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    std_normalize
])

# Dataset e divisão de treinamento/teste
dataset_dir = r'C:\Users\phenr\Downloads\kagglecatsanddogs_5340\PetImages'
dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=trans)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Função para exibir imagens do dataset
def display_dataset(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.permute(1, 2, 0))  # Transpor canais para exibição
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

# Modelo pré-treinado VGG16
vgg = vgg16(pretrained=True)

# Congelar os pesos das features
for param in vgg.features.parameters():
    param.requires_grad = False

# Substituir o classificador para a nova tarefa
vgg.classifier[6] = nn.Linear(4096, 2)  # 2 classes: Dog e Cat
vgg.to(device)

# Função de treinamento
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=10, print_freq=90):
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
    
    print("Treinamento concluído.")
    return model

# Configuração para treinamento
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Treinamento do modelo
trained_model = train_model(vgg, train_loader, test_loader, loss_fn, optimizer, epochs=1, print_freq=100)

# Caminho para salvar o modelo
model_path = r'C:\Users\phenr\Desktop\model.pth'

# Criar diretórios, se necessário
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Testar permissão de gravação
try:
    with open(model_path, 'w') as f:
        f.write("Test")
    print("Local acessível para gravação.")
except Exception as e:
    print(f"Erro ao acessar o local para salvar o modelo: {e}")
    exit()

# Salvar o modelo treinado
try:
    torch.save(trained_model.state_dict(), model_path)
    print(f"Modelo salvo com sucesso em: {model_path}")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")

# Inferência em uma única imagem
def predict_image(model, image_path, transform):
    model.eval()
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo de imagem não encontrado em {image_path}")
        return
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # Adicionar dimensão batch
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

# Caminho da imagem de teste
test_image_path = r'c:\Users\phenr\Downloads\depositphotos_191128088-stock-photo-chimera-with-angry-hairless-sphinx.jpg'

predicted_class = predict_image(trained_model, test_image_path, trans)
if predicted_class is not None:
    print(f"Classe prevista: {'Dog' if predicted_class == 1 else 'Cat'}")
