import torch
from torchvision import transforms
from PIL import Image
import os
from model import HybridGenerator  # import your model class

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridGenerator().to(device)
model.load_state_dict(torch.load("checkpoints/gan_model_epoch_5.pth", map_location=device))
model.eval()

# Preprocess input
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Postprocess output
def postprocess(tensor):
    img = tensor.squeeze().detach().cpu().numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype("uint8").transpose(1, 2, 0)
    return Image.fromarray(img)

# Enhance Function
def enhance_image(image_file):
    img = Image.open(image_file).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    enhanced_img = postprocess(output[0])
    os.makedirs("output", exist_ok=True)
    save_path = f"output/enhanced_{os.path.basename(image_file.filename)}"
    enhanced_img.save(save_path)
    return save_path
