import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch import nn, optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

data_dir = f'C:/Users/vinif/OneDrive/Documents/cropped_dataset'


def main():
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

    # Definir a proporção de dados para validação
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Dividir o dataset em treino e validação
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Criar os DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    }

    dataset_sizes = {'train': train_size, 'val': val_size}
    class_names = dataset.classes

    model = models.efficientnet_b0(pretrained=True)

    # Ajuste o modelo para o número de classes do seu dataset
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
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
                    labels = labels.to(device)

                    # Zerando os gradientes
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
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

    def compute_confusion_matrix(model, dataloader, class_names):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()  # Colocar o modelo em modo de avaliação

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcular a matriz de confusão
        cm = confusion_matrix(all_labels, all_preds)

        # Exibir a matriz de confusão
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    model = train_model(model, criterion, optimizer)

    compute_confusion_matrix(model, dataloaders['val'], class_names)

    torch.save(model.state_dict(), 'best_model_efficientnet_b0.pth')


if __name__ == '__main__':
    main()
