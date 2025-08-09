import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.unet import UNet
from src.diffusion import DiffusionModel
from src.transforms import get_transforms, reverse_transform

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def get_dataset(config):
    transform = get_transforms(config['data']['image_size']) 
    dataset = ImageFolder(root=config['data']['dataset_path'], transform=transform)
    
    return dataset

def train_step(batch, t, model, diffusion, optimizer, device):
    batch = batch.to(device)
    t = t.to(device)
    
    batch_noisy, noise = diffusion.forward(batch, t, device)
    predicted_noise = model(batch_noisy, t)
    
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(noise, predicted_noise)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():  
    config = load_config('config.yaml')
    
    device = torch.device(config['training']['device'])
    
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(save_dir)
    
    #init models
    unet = UNet(config).to(device)
    diffusion = DiffusionModel(config)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=config['training']['learning_rate'])
    
    # dataset
    dataset = get_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    # Training
    for epoch in range(config['training']['epochs']):
        mean_epoch_loss = []
        
        for batch, _ in dataloader:
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],)).long()
            loss = train_step(batch, t, unet, diffusion, optimizer, device)
            mean_epoch_loss.append(loss)
        
        if epoch % config['training']['print_frequency'] == 0:
            avg_loss = sum(mean_epoch_loss) / len(mean_epoch_loss)
            logging.info(f"Epoch: {epoch} | Train Loss: {avg_loss:.6f}")
            
            # Save ckpt
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
            
            with torch.no_grad():
                img = torch.randn((1, 3) + tuple(config['data']['image_size'])).to(device)
                for i in reversed(range(diffusion.timesteps)):
                    t = torch.full((1,), i, dtype=torch.long, device=device)
                    img = diffusion.backward(img, t, unet.eval())
                
                plt.figure(figsize=(8, 8))
                plt.imshow(reverse_transform()(img[0]))
                plt.savefig(save_dir / f'sample_epoch_{epoch}.png')
                plt.close()

if __name__ == '__main__':
    main()

