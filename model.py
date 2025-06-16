import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np

# -------------------------
# Dataset Class
# -------------------------
class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir=None, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir)) if target_dir else None

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_path).convert("RGB")
        if self.transform:
            input_image = self.transform(input_image)

        if self.target_images:
            target_path = os.path.join(self.target_dir, self.target_images[idx])
            target_image = Image.open(target_path).convert("RGB")
            target_image = self.transform(target_image) if self.transform else target_image
            return input_image, target_image

        return input_image

# -------------------------
# Residual Block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

# -------------------------
# Hybrid Generator (Deep Encoder & Decoder)
# -------------------------
class HybridGenerator(nn.Module):
    def __init__(self):
        super(HybridGenerator, self).__init__()

        # Encoder (8 layers)
        self.e1 = self.encoder_layer(3, 64, normalize=False)
        self.e2 = self.encoder_layer(64, 128)
        self.e3 = self.encoder_layer(128, 256)
        self.e4 = self.encoder_layer(256, 512)
        self.e5 = self.encoder_layer(512, 512)
        self.e6 = self.encoder_layer(512, 512)
        self.e7 = self.encoder_layer(512, 512)
        self.e8 = self.encoder_layer(512, 512)

        # Residual Blocks (From Enhanced GAN)
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(5)])

        # Decoder (8 layers)
        self.d1 = self.decoder_layer(512, 512)
        self.d2 = self.decoder_layer(1024, 512)
        self.d3 = self.decoder_layer(1024, 512)
        self.d4 = self.decoder_layer(1024, 512)
        self.d5 = self.decoder_layer(1024, 256)
        self.d6 = self.decoder_layer(512, 128)
        self.d7 = self.decoder_layer(256, 64)
        self.final_layer = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def encoder_layer(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        res = self.res_blocks(e8)

        d1 = self.d1(res)
        d2 = self.d2(torch.cat((d1, e7), 1))
        d3 = self.d3(torch.cat((d2, e6), 1))
        d4 = self.d4(torch.cat((d3, e5), 1))
        d5 = self.d5(torch.cat((d4, e4), 1))
        d6 = self.d6(torch.cat((d5, e3), 1))
        d7 = self.d7(torch.cat((d6, e2), 1))
        final = self.final_layer(torch.cat((d7, e1), 1))
        return torch.tanh(final)
    

    # -------------------------
# PatchGAN Discriminator (SeaPixGAN)
# -------------------------
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self.discriminator_block(6, 64, normalize=False),
            self.discriminator_block(64, 128),
            self.discriminator_block(128, 256),
            self.discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def discriminator_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        input_tensor = torch.cat((img_A, img_B), 1)
        return self.model(input_tensor)

# -------------------------
# Gradient Difference Loss (GDL)
# -------------------------
class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = torch.tensor(SobelX, dtype=torch.float32).expand(self.chans, 1, 3, 3)
        self.Ky = torch.tensor(SobelY, dtype=torch.float32).expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = nn.functional.conv2d(im, self.Kx.to(im.device), stride=1, padding=1, groups=self.chans)
        gy = nn.functional.conv2d(im, self.Ky.to(im.device), stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        gradX_true, gradY_true = self.get_gradients(true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
        grad_pred = torch.abs(gradX_pred) ** self.alpha + torch.abs(gradY_pred) ** self.alpha
        return 0.5 * torch.mean(grad_true - grad_pred)


# -------------------------
# Perceptual Loss (VGG-Based)
# -------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16]
        self.vgg = nn.Sequential(*[vgg[i] for i in range(len(vgg))]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.criterion(pred_features, target_features)

# -------------------------
# Training Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
input_dir = "./trainA"
target_dir = "./trainB"

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Dataset and DataLoader
train_dataset = UnderwaterDataset(input_dir, target_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model, Loss, and Optimizers
generator = HybridGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

pixel_loss = nn.L1Loss().to(device)
adversarial_loss = nn.BCEWithLogitsLoss().to(device)
perceptual_loss = PerceptualLoss().to(device)
gdl_loss = Gradient_Difference_Loss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

if __name__ == "__main__":
    for epoch in range(5):
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)

            # -----------------------------
            # Train Discriminator
            # -----------------------------
            optimizer_D.zero_grad()
            real_preds = discriminator(imgs, targets)
            fake_imgs = generator(imgs).detach()  # Detach to avoid reusing computation graph
            fake_preds = discriminator(imgs, fake_imgs)

            d_real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds))
            d_fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # -----------------------------
            # Train Generator
            # -----------------------------
            optimizer_G.zero_grad()

            fake_imgs = generator(imgs)  # Generate fresh fake images
            fake_preds = discriminator(imgs, fake_imgs)

            # Compute loss components separately
            g_adv_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))
            g_pixel_loss = pixel_loss(fake_imgs, targets)
            g_perceptual_loss = perceptual_loss(fake_imgs, targets)  # Ensure VGG loss is computed properly
            g_gdl_loss = gdl_loss(fake_imgs, targets)

            # Total generator loss
            g_loss = g_adv_loss + 100 * g_pixel_loss + 10 * g_perceptual_loss + 5 * g_gdl_loss
            g_loss.backward()  # No retain_graph=True needed anymore
            optimizer_G.step()

        print(f"Epoch {epoch + 1}: D Loss={d_loss.item():.4f}, G Loss={g_loss.item():.4f}")
        if (epoch + 1) % 5 == 0:
            model_save_path = f"checkpoints/gan_model_epoch_{epoch+1}.pth"
            torch.save(generator.state_dict(), model_save_path)
            print(f"✅ Generator saved at {model_save_path}")
