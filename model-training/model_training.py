import os
import glob
import random
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ------------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# ------------------------------------------------------------------------------
# CONFIG = {
#     "TRAIN_DIR_A": "data/trainA",  # Horse path
#     "TRAIN_DIR_B": "data/trainB",  # Zebra path
#     "IMG_SIZE": 256,
#     "BATCH_SIZE": 1,               # CycleGAN typically uses batch_size=1
#     "LR": 2e-4,
#     "EPOCHS": 50,                  # Adjusted for demonstration
#     "NUM_WORKERS": 4,              # Parallel CPU data loading
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
# }

print("CUDA Available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

# When running in Docker via Airflow, we mount the volume to /outputs
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", "data") 
OUTPUT_MODEL_DIR = os.getenv("OUTPUT_MODEL_DIR", "saved_models")

CONFIG = {
    "TRAIN_DIR_A": os.path.join(BASE_DATA_DIR, "trainA"),
    "TRAIN_DIR_B": os.path.join(BASE_DATA_DIR, "trainB"),
    "IMG_SIZE": 256,
    "BATCH_SIZE": 1,
    "LR": 2e-4,
    "EPOCHS": 5,
    "NUM_WORKERS": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MLFLOW_URI": os.getenv("MLFLOW_URI", "http://localhost:5000"),
    "EXPERIMENT_NAME": "Image-to-Image_CycleGAN"
}

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Enable CUDNN Benchmark for optimized performance on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------------------------
# 2. Dataset & DataLoader (Optimized Memory Usage)
# ------------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        if not os.path.exists(root_dir):
            self.files = []
        else:
            self.files = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img

def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds_A = ImageDataset(CONFIG["TRAIN_DIR_A"], transform=transform)
    ds_B = ImageDataset(CONFIG["TRAIN_DIR_B"], transform=transform)
    loader_A = DataLoader(ds_A, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    loader_B = DataLoader(ds_B, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    return loader_A, loader_B

# ------------------------------------------------------------------------------
# 3. Model Architecture (ResNet Generator & PatchGAN Discriminator)
# ------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, n_resnet=9):
        super().__init__()
        
        # Initial Convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual Blocks
        for _ in range(n_resnet):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, padding=0),
            nn.Tanh() # Output range [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# ------------------------------------------------------------------------------
# 4. Utilities
# ------------------------------------------------------------------------------
class ImagePool:
    """Keeps a history of generated images to stabilize discriminator training."""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)

# ------------------------------------------------------------------------------
# 5. Training Loop (MLflow)
# ------------------------------------------------------------------------------
def train():
    device = CONFIG["DEVICE"]
    print(f"Training on: {device} | Data: {CONFIG['TRAIN_DIR_A']}")

    # --- MLflow Setup ---
    mlflow.set_tracking_uri(CONFIG["MLFLOW_URI"])
    mlflow.set_experiment(CONFIG["EXPERIMENT_NAME"])

    # Initialize Models
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Initialize Weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=CONFIG["LR"], betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=CONFIG["LR"], betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=CONFIG["LR"], betas=(0.5, 0.999))

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # DataLoaders
    loader_A, loader_B = get_loaders()
    if len(loader_A) == 0 or len(loader_B) == 0:
        print("Error: Datasets are empty.")
        with mlflow.start_run(run_name="no_data_run"):
            mlflow.log_param("status", "no_data")
        return
    
    # Image Pools
    fake_A_pool = ImagePool()
    fake_B_pool = ImagePool()

    # Mixed Precision Scaler (Updated for PyTorch 2.x)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Training")
    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        for epoch in range(CONFIG["EPOCHS"]):
            progress_bar = tqdm(zip(loader_A, loader_B), desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}", total=min(len(loader_A), len(loader_B)))
            
            for i, (real_A, real_B) in enumerate(progress_bar):
                # Move to GPU
                real_A = real_A.to(device)
                real_B = real_B.to(device)
                
                # Ground Truths
                valid = torch.ones(real_A.size(0), 1, 16, 16, requires_grad=False).to(device) # PatchGAN output size depends on input
                fake = torch.zeros(real_A.size(0), 1, 16, 16, requires_grad=False).to(device)

                # ------------------
                #  Train Generators
                # ------------------
                optimizer_G.zero_grad()
                
                # Updated autocast for PyTorch 2.x
                with torch.amp.autocast('cuda'):
                    # Identity loss
                    loss_id_A = criterion_identity(G_BA(real_A), real_A)
                    loss_id_B = criterion_identity(G_AB(real_B), real_B)
                    loss_identity = (loss_id_A + loss_id_B) / 2

                    # GAN loss
                    fake_B = G_AB(real_A)
                    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                    fake_A = G_BA(real_B)
                    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                    # Cycle loss
                    rec_A = G_BA(fake_B)
                    loss_cycle_A = criterion_cycle(rec_A, real_A)
                    rec_B = G_AB(fake_A)
                    loss_cycle_B = criterion_cycle(rec_B, real_B)
                    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                    # Total loss
                    loss_G = loss_GAN + (10.0 * loss_cycle) + (5.0 * loss_identity)

                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)

                # -----------------------
                #  Train Discriminator A
                # -----------------------
                optimizer_D_A.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # Real loss
                    loss_real = criterion_GAN(D_A(real_A), valid)
                    # Fake loss (sampled from pool)
                    fake_A_ = fake_A_pool.query(fake_A.detach())
                    loss_fake = criterion_GAN(D_A(fake_A_), fake)
                    loss_D_A = (loss_real + loss_fake) / 2

                scaler.scale(loss_D_A).backward()
                scaler.step(optimizer_D_A)

                # -----------------------
                #  Train Discriminator B
                # -----------------------
                optimizer_D_B.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # Real loss
                    loss_real = criterion_GAN(D_B(real_B), valid)
                    # Fake loss (sampled from pool)
                    fake_B_ = fake_B_pool.query(fake_B.detach())
                    loss_fake = criterion_GAN(D_B(fake_B_), fake)
                    loss_D_B = (loss_real + loss_fake) / 2

                scaler.scale(loss_D_B).backward()
                scaler.step(optimizer_D_B)

                # Updates scaler once per iteration
                scaler.update()

                # Update progress bar
                progress_bar.set_postfix({"Loss G": loss_G.item(), "Loss D": (loss_D_A.item() + loss_D_B.item())})

        # Save Checkpoints
        os.makedirs("saved_models", exist_ok=True)
        torch.save(G_AB.state_dict(), "saved_models/G_AB.pth")
        torch.save(G_BA.state_dict(), "saved_models/G_BA.pth")
        print("Training Complete. Models saved.")

        # Log Model to MLflow
        mlflow.pytorch.log_model(G_AB, "cyclegan_model_AB")
        mlflow.pytorch.log_model(G_BA, "cyclegan_model_BA")

        # --------------------------------------------------------------------------
        # 6. Export to ONNX (For Go Inference)
        # --------------------------------------------------------------------------
        print("Exporting to ONNX...")
        G_AB.eval()
        
        # Create dummy input (Batch size 1, 3 channels, 256x256)
        dummy_input = torch.randn(1, 3, 256, 256, device=device)
        
        onnx_filename = "cyclegan_candidate.onnx"
        onnx_path = os.path.join(OUTPUT_MODEL_DIR, onnx_filename)

        torch.onnx.export(
            G_AB,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Export complete: {onnx_path}")

        mlflow.log_artifact(onnx_path)
        
# def export_onnx():
#     # Configuration
#     model_path = "saved_models/G_AB.pth"  # Path to your saved weights
#     output_onnx = "cyclegan_horse2zebra.onnx"
#     device = torch.device("cpu") # Exporting on CPU is standard and safer

#     # Check if model file exists
#     if not os.path.exists(model_path):
#         print(f"Error: Could not find model file at {model_path}")
#         print("Make sure you have run the training script at least once to save the weights.")
#         return

#     print(f"Loading model architecture...")
#     model = Generator(n_resnet=9).to(device)

#     print(f"Loading weights from {model_path}...")
#     try:
#         # Load state dictionary
#         state_dict = torch.load(model_path, map_location=device)
#         model.load_state_dict(state_dict)
#         model.eval() # Set to evaluation mode
#     except Exception as e:
#         print(f"Failed to load weights: {e}")
#         return

#     print("Creating dummy input for ONNX export...")
#     # Batch size 1, 3 channels, 256x256 image
#     dummy_input = torch.randn(1, 3, 256, 256, device=device)

#     print(f"Exporting to {output_onnx}...")
#     try:
#         torch.onnx.export(
#             model,
#             dummy_input,
#             output_onnx,
#             export_params=True,        # Store the trained parameter weights inside the model file
#             opset_version=11,          # Standard ONNX version
#             do_constant_folding=True,  # Optimization
#             input_names=['input'],
#             output_names=['output'],
#             dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
#         )
#         print(f"Success! Model exported to: {os.path.abspath(output_onnx)}")
#     except ImportError:
#         print("Error: 'onnx' module is missing.")
#         print("Please run: pip install onnx")
#     except Exception as e:
#         print(f"Export failed: {e}")

# if __name__ == "__main__":
#     # Ensure data directories exist before running
#     if not os.path.exists(CONFIG["TRAIN_DIR_A"]) or not os.path.exists(CONFIG["TRAIN_DIR_B"]):
#         print(f"Error: Dataset directories not found at {CONFIG['TRAIN_DIR_A']} or {CONFIG['TRAIN_DIR_B']}")
#     else:
#         train()


# Only Export model (ALready trained)   
# if __name__ == "__main__":
#     export_onnx()

if __name__ == "__main__":
    train()