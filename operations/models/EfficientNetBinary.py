import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn, optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy
from PIL import Image

data_dir = 'C:/Users/vinif/OneDrive/Documents/cropped_dataset'

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

labels = ['Positive', 'Negative']


class BinaryDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx = target_class_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Transformar o rótulo para binário: 1 para a classe alvo, 0 para as outras
        binary_label = 1 if label == self.target_class_idx else 0
        return img, binary_label

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f'Model loaded from {path}')
    return model

def classify_image(image_path, model_path):
    transform = data_transforms['val']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_binary = models.efficientnet_b0(pretrained=False)
    num_ftrs = model_binary.classifier[1].in_features
    model_binary.classifier[1] = nn.Linear(num_ftrs, 1)
    model_binary = load_model(model_binary, model_path, device)

    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    image = image.to(device)

    # Set the model to evaluation mode and make the prediction
    model_binary.eval()
    with torch.no_grad():
        outputs = model_binary(image)
        _, preds = torch.max(outputs, 1)

    return labels[preds.item()]


def main():
    # Definindo as transformações para treino e validação
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carregar o dataset completo com as transformações de treino
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    print('Class indices: ', dataset.class_to_idx)

    # Definir a classe alvo (por exemplo, a primeira classe)
    target_class_idx = 4  # Índice da classe que será classificada como 1

    # Transformar o dataset para classificação binária
    binary_dataset = BinaryDataset(dataset, target_class_idx)

    # Definir a proporção de dados para validação
    val_split = 0.2
    val_size = int(len(binary_dataset) * val_split)
    train_size = len(binary_dataset) - val_size

    # Dividir o dataset em treino e validação
    train_dataset, val_dataset = random_split(binary_dataset, [train_size, val_size])

    # Aplicar a transformação de validação no conjunto de validação
    val_dataset.dataset.dataset.transform = val_transforms

    # Criar os DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    }

    dataset_sizes = {'train': train_size, 'val': val_size}

    model = models.efficientnet_b0(pretrained=True)

    # Ajuste o modelo para uma única saída binária
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_model(model, criterion, optimizer, num_epochs=25):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Cada época tem uma fase de treino e outra de validação
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Definir modelo para treinamento
                else:
                    model.eval()  # Definir modelo para avaliação

                running_loss = 0.0
                running_corrects = 0

                # Iterar sobre os dados
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float().unsqueeze(1)  # Ajustar os rótulos para BCEWithLogitsLoss

                    # Zerando os gradientes
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        preds = torch.round(torch.sigmoid(outputs))
                        loss = criterion(outputs, labels)

                        # Backward + otimização apenas se estiver na fase de treino
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Estatísticas
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy do melhor modelo
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Best val Acc: {best_acc:.4f}')

        # Carregar os melhores pesos do modelo
        model.load_state_dict(best_model_wts)
        return model

    def compute_confusion_matrix(model, dataloader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()  # Colocar o modelo em modo de avaliação

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = torch.round(torch.sigmoid(outputs))

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcular a matriz de confusão
        cm = confusion_matrix(all_labels, all_preds)

    model = train_model(model, criterion, optimizer)

    compute_confusion_matrix(model, dataloaders['val'])

    torch.save(model.state_dict(), 'best_model_efficientnet_b0_binnary.pth')


if __name__ == '__main__':
    main()