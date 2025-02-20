import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask.squeeze(0).long()
    
def collate_fn(batch):
    images, masks = zip(*batch)

    # Separate calculations for images and masks
    max_img_h = max(img.shape[1] for img in images)
    max_img_w = max(img.shape[2] for img in images)
    max_mask_h = max(msk.shape[0] for msk in masks)
    max_mask_w = max(msk.shape[1] for msk in masks)

    # Use max of both dimensions
    max_h = max(max_img_h, max_mask_h)
    max_w = max(max_img_w, max_mask_w)

    # Pad both to combined max dimensions
    padded_images = [F.pad(img, (0, max_w-img.shape[2], 0, max_h-img.shape[1])) 
                     for img in images]
    padded_masks = [F.pad(msk.unsqueeze(0), (0, max_w-msk.shape[1], 0, max_h-msk.shape[0]))
                    .squeeze(0) 
                    for msk in masks]

    return torch.stack(padded_images), torch.stack(padded_masks)


# UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out
    
# Training Setup
def main():
    # Dataset paths
    train_dataset = SegmentationDataset(
        "/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/data/train/images",
        "/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/data/train/masks"
    )

    val_dataset = SegmentationDataset(
        "/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/data/val/images",
        "/Users/aja294/Documents/Hemp_local/leaf_morphometrics/semantic_seg_template/data/val/masks"
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()


    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, masks).item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "unet_model.pth")

if __name__ == "__main__":
    main()