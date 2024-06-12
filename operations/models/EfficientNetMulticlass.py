import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import os
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

labels = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Negative for intraepithelial lesion', 'SCC']

data_dir = 'C:\\Users\\vinif\\PycharmProjects\\PAI\\cropped_dataset_train_val'


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


class BinaryImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = 1 if self.classes[target] == 'Negative for intraephitelial lesion' else 0
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def classify_image(image_path, model_path):
    transform = data_transforms['val']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_multiclass = models.efficientnet_b0(pretrained=False)
    num_ftrs = model_multiclass.classifier[1].in_features
    model_multiclass.classifier[1] = nn.Linear(num_ftrs, 6)
    model_multiclass = load_model(model_multiclass, model_path, device)

    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    image = image.to(device)

    # Set the model to evaluation mode and make the prediction
    model_multiclass.eval()
    with torch.no_grad():
        output = model_multiclass(image)
        if model_multiclass.classifier[1].out_features == 1:  # Binary classification
            prediction = (torch.sigmoid(output) > 0.5).float().item()
            predicted_class = round(prediction)
        else:  # Multiclass classification
            _, prediction = torch.max(output, 1)
            predicted_class = prediction.item()

    return labels[predicted_class]


def main():
    # Transformações de dados
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_datasets_binary = {x: BinaryImageFolder(os.path.join(data_dir, x), data_transforms[x])
                             for x in ['train', 'val']}
    dataloaders_binary = {x: DataLoader(image_datasets_binary[x], batch_size=32, shuffle=True, num_workers=4)
                          for x in ['train', 'val']}
    dataset_sizes_binary = {x: len(image_datasets_binary[x]) for x in ['train', 'val']}
    class_names_binary = ['Non-Negative', 'Negative for intraephitelial lesion']

    model_binary = models.efficientnet_b0(pretrained=True)
    num_ftrs = model_binary.classifier[1].in_features
    model_binary.classifier[1] = nn.Linear(num_ftrs, 1)
    model_binary = model_binary.to(device)

    model_multiclass = models.efficientnet_b0(pretrained=True)
    num_ftrs = model_multiclass.classifier[1].in_features
    model_multiclass.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model_multiclass = model_multiclass.to(device)

    def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        train_acc_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        if model.classifier[1].out_features == 1:  # Modelo binário
                            loss = criterion(outputs, labels.float().unsqueeze(1))
                            preds = (torch.sigmoid(outputs) > 0.5).float()
                        else:  # Modelo multiclass
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_acc_history.append(epoch_acc)
                else:
                    val_acc_history.append(epoch_acc)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)
        return model, train_acc_history, val_acc_history

    criterion_binary = nn.BCEWithLogitsLoss()
    optimizer_binary = Adam(model_binary.parameters(), lr=0.001)

    model_binary, train_acc_hist_binary, val_acc_hist_binary = train_model(
        model_binary, criterion_binary, optimizer_binary, dataloaders_binary, dataset_sizes_binary, num_epochs=25
    )

    criterion_multiclass = nn.CrossEntropyLoss()
    optimizer_multiclass = Adam(model_multiclass.parameters(), lr=0.001)

    model_multiclass, train_acc_hist_multiclass, val_acc_hist_multiclass = train_model(
        model_multiclass, criterion_multiclass, optimizer_multiclass, dataloaders, dataset_sizes, num_epochs=25
    )

    def compute_confusion_matrix(model, dataloader, class_names):
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                if model.classifier[1].out_features == 1:  # Modelo binário
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:  # Modelo multiclass
                    _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        # disp.plot(cmap=plt.cm.Blues)
        # plt.show()

    # Para o modelo binário
    compute_confusion_matrix(model_binary, dataloaders_binary['val'], class_names_binary)

    # Para o modelo multiclasse
    compute_confusion_matrix(model_multiclass, dataloaders['val'], class_names)


if __name__ == '__main__':
    main()
