import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from icecream import ic

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.backbone(x)['out']

class YourDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def load_model(checkpoint_path, num_classes, device):
    model = SimpleSegmentationModel(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Ensure labels and preds are integers
    all_labels = [int(label) for label in all_labels]
    all_preds = [int(pred) for pred in all_preds]

    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, class_names):
    ic(f"Labels: {labels[:10]}")
    ic(f"Preds: {preds[:10]}")
    ic(f"Labels dtype: {type(labels)}, Preds dtype: {type(preds)}")
    ic(f"Labels shape: {len(labels)}, Preds shape: {len(preds)}")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(labels, preds, class_names):
    report = classification_report(labels, preds, target_names=class_names)
    print(report)

def visualize_predictions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            inputs = inputs.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(min(len(inputs), 5)):  # Visualize up to 5 samples
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(inputs[i].transpose(1, 2, 0))
                plt.title('Original Image')
                plt.subplot(1, 3, 2)
                plt.imshow(labels[i], cmap='gray')
                plt.title('Ground Truth Mask')
                plt.subplot(1, 3, 3)
                plt.imshow(preds[i], cmap='gray')
                plt.title('Predicted Mask')
                plt.show()

def main():
    # Example usage
    checkpoint_path = '/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/checkpoints/checkpoint_epoch_60.pth'
    num_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, num_classes, device)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Define the validation dataset
    val_dataset = YourDataset(
        image_dir='/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/data/images/val',
        mask_dir='/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/data/masks/val',
        transform=transform
    )

    # Create the DataLoader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    labels, preds = evaluate_model(model, val_loader, device)
    class_names = ['Background', 'Foreground']
    plot_confusion_matrix(labels, preds, class_names)
    print_classification_report(labels, preds, class_names)

    # Visualize predictions
    visualize_predictions(model, val_loader, device)

if __name__ == "__main__":
    main()