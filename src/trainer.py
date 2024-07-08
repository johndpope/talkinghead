from src.dataloader import VideoDataset
import yaml

from torch.utils.data import DataLoader
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from src.model.loss import PerceptualLoss, GANLoss, CycleConsistencyLoss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
import tqdm
import math
import os
from PIL import Image
from torchvision import transforms
from src.model.portrait import Portrait
import torch.nn.functional as F
from CelebADataset import CelebADataset,ProgressiveDataset,AffectNetDataset
from torchvision.utils import save_image


def collate_frames(batch):
    """
    Custom collate function that processes a batch of tensors, each containing sampled frames from videos.

    Args:
    batch (list of Tensors): The batch containing video frame tensors.

    Returns:
    tuple: A tuple containing processed tensors ready for model input.
    """
    # Randomly select two frames from each half of the batch for processing
    Xs_stack = []
    Xd_stack = []
    Xs_prime_stack = []
    Xd_prime_stack = []

    half_point = len(batch) // 2
    for item in batch[:half_point]:
        if item.shape[0] > 1:  # Ensure there is more than one frame
            indices = random.sample(range(item.shape[0]), 2)
            Xs_stack.append(item[indices[0]])
            Xd_stack.append(item[indices[1]])

    for item in batch[half_point:]:
        if item.shape[0] > 1:
            indices = random.sample(range(item.shape[0]), 2)
            Xs_prime_stack.append(item[indices[0]])
            Xd_prime_stack.append(item[indices[1]])

    # Stack all selected frames to create batches
    if Xs_stack and Xd_stack and Xs_prime_stack and Xd_prime_stack:  # Check if lists are not empty
        Xs = torch.stack(Xs_stack)
        Xd = torch.stack(Xd_stack)
        Xs_prime = torch.stack(Xs_prime_stack)
        Xd_prime = torch.stack(Xd_prime_stack)
        
        # Concatenate Xs with Xs_prime and Xd with Xd_prime
        Xs_combined = torch.cat((Xs, Xs_prime), dim=0)
        Xd_combined = torch.cat((Xd, Xd_prime), dim=0)
        Xs_prime_combined = torch.cat((Xs_prime, Xs), dim=0)
        Xd_prime_combined = torch.cat((Xd_prime, Xd), dim=0)

        return Xs_combined, Xd_combined, Xs_prime_combined, Xd_prime_combined
    else:
        # Return zero tensors if not enough frames were available
        zero_tensor = torch.zeros((1, 3, 224, 224))  # Adjust dimensions as per your model's requirement
        return zero_tensor, zero_tensor, zero_tensor, zero_tensor

def load_data(root_dir, batch_size=8, transform=None):
    """
    Function to load data using VideoDataset and DataLoader with a custom collate function.
    
    Args:
    root_dir (str): Path to the directory containing videos.
    batch_size (int): Number of samples per batch.
    transform (callable, optional): Transformations applied to video frames.

    Returns:
    DataLoader: A DataLoader object ready for iteration.
    """
    dataset = VideoDataset(root_dir=root_dir, transform=transform, frames_per_clip=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_frames)
    return dataloader

def save_debug_images(x_s, x_t, x_s_recon, x_t_recon, step, resolution, output_dir):
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    x_s, x_t = denorm(x_s), denorm(x_t)
    x_s_recon, x_t_recon = denorm(x_s_recon), denorm(x_t_recon)
    
    combined = torch.cat([x_s, x_s_recon, x_t, x_t_recon], dim=0)
    
    num_sets = min(16, x_s.size(0))
    save_image(combined[:num_sets*4], os.path.join(output_dir, f"debug_step_{step}_resolution_{resolution}.png"), nrow=4)


def train_model(config, p, train_loader):
    initial_resolution = config["training"]["initial_resolution"]
    optimizer = torch.optim.Adam(p.parameters(), lr=p.config["training"]["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p.to(device)
    if config["training"]["use_wandb"]:
        wandb.init(project='portrait_project', resume="allow", config=config)

    checkpoint_path = f"./models/portrait/{p.config['training']['name']}/"
    num_epochs = config["training"]["num_epochs"]

    start_epoch = 0

    epochs_per_stage = config["training"]["epochs_per_stage"]
    transition_epochs = config["training"]["transition_epochs"]
    final_resolution = config["training"]["final_resolution"]
    initial_resolution = config["training"]["initial_resolution"]
    epochs_per_full_stage = epochs_per_stage + transition_epochs
    current_resolution = initial_resolution

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) == 0:
            latest_epoch = max([int(epoch_dir.split("epoch")[1]) for epoch_dir in os.listdir(checkpoint_path)])
            checkpoint_path = os.path.join(checkpoint_path, f"epoch{latest_epoch}/checkpoint.pth")

            checkpoint = torch.load(checkpoint_path)
            p.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            current_resolution = checkpoint['current_resolution']

    p.train()
    perceptual_loss = PerceptualLoss(config)
    gan_loss = GANLoss(config, model=p)

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0

        # Wrap the training loader with tqdm for a progress bar
        train_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
        log_interval = 1000# len(train_loader) // 3

        current_resolution = min(initial_resolution*2**(epoch//epochs_per_full_stage), final_resolution)
        step = int(math.log2(current_resolution))-2

        in_transition = (epoch % epochs_per_full_stage) >= epochs_per_stage
        if not in_transition:
            passed_transitions = 0 # current alpha is the amount of transition phases that has passed
        else:
            passed_transitions = epoch % epochs_per_full_stage - epochs_per_stage

        for step, batch in enumerate(train_loader):
            x_s, x_t = batch["source_image"], batch["target_image"]
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"], batch["emotion_labels_t"]
            
            optimizer.zero_grad()

            if in_transition:
                alpha = (passed_transitions * len(train_loader) + step) / (transition_epochs * len(train_loader))
                alpha = max(0, min(alpha, 1))
            else:
                alpha = 0

            # IRFD forward pass
            x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t = p.irfd_forward(x_s, x_t,alpha,step)

            # Calculate IRFD losses
            L_identity = F.mse_loss(fi_s, p.Ei(x_s_recon)) + F.mse_loss(fi_t, p.Ei(x_t_recon))
            L_emotion = F.mse_loss(fe_s, p.Ee(x_s_recon)) + F.mse_loss(fe_t, p.Ee(x_t_recon))
            L_pose = F.mse_loss(fp_s, p.Ep(x_s_recon)) + F.mse_loss(fp_t, p.Ep(x_t_recon))
            # L_self = F.mse_loss(x_s_recon, Xs) + F.mse_loss(x_t_recon, Xd)

            # Calculate classification loss if needed (you'll need to implement or use a pre-trained classifier)
            # L_cls = classification_loss(fe_s, emotion_labels_s) + classification_loss(fe_t, emotion_labels_t)

            # Combine IRFD losses
            L_disentanglement = L_identity + L_emotion + L_pose # + L_self  # + L_cls if using classification loss

            # # Original talking head generation forward pass
            fi_d, fe_d, fp_d = p.encode(x_t)
            print(f"fi_d:{fi_d.shape} fe_d:{fe_d.shape} fp_d:{fp_d.shape}")
            gd = p.decode(fi_d, fe_d, fp_d, alpha, step)

            # Calculate original losses
            Lper = perceptual_loss(x_t, gd)
            Lgan = gan_loss(x_t, gd, alpha, step)

            # Combine all losses
            total_loss = L_disentanglement + Lper[0] + Lgan[0]

            total_loss.backward()
            optimizer.step()

            if step % config['training']['save_image_steps'] == 0:
                save_debug_images(x_s, x_t, x_s_recon, x_t_recon, epoch, step, config['training']['output_dir'])
            if step % log_interval == 0 and config["training"]["use_wandb"]:
                wandb.log({
                    'Example Source': wandb.Image(x_s[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Driver': wandb.Image(x_t[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Output': wandb.Image(gd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                })

            wandb_log = {
                'Epoch': epoch + 1,
                'Total Loss': total_loss.item()
            }

            # if self.config['weights']['perceptual']['gaze'] != 0:
            #     wandb_log['Gaze Loss'] = Lper[1]['Lgaze'].item()
            if p.config['weights']['perceptual']['lpips'] != 0:
                wandb_log['lpips Loss'] = Lper[1]['lpips'].item()
            if p.config['weights']['gan']['real'] + p.config['weights']['gan']['fake'] + p.config['weights']['gan'][
                'feature_matching'] != 0:
                wandb_log['GAN Loss'] = Lgan[0].item()
            if p.config['weights']['gan']['real'] != 0:
                wandb_log['GAN real Loss'] = Lgan[1]['real_loss'].item()
            if p.config['weights']['gan']['fake'] != 0:
                wandb_log['GAN fake Loss'] = Lgan[1]['fake_loss'].item()
            if p.config['weights']['gan']['adversarial'] != 0:
                wandb_log['GAN adversarial Loss'] = Lgan[1]['adversarial_loss'].item()
            # if p.config['weights']['gan']['feature_matching'] != 0:
            #     wandb_log['Gan feature Loss'] = Lgan[1]['feature_matching_loss'].item()

            if config["training"]["use_wandb"]:
                wandb.log(wandb_log)


            train_iterator.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.item():.4f}"
            )

        # p.save_model(path="./models/portrait/epoch{}/".format(epoch))
        p.save_model(path=f"./models/portrait/{p.config['training']['name']}/epoch{epoch}/", epoch=epoch,
                     optimizer=optimizer, current_resolution=current_resolution)
        print(f'Epoch {epoch + 1}, Average Loss {running_loss / len(train_loader):.4f}')

def create_progressive_dataloader(config, base_dataset, resolution, is_validation=False):
    print("config:")
    progressive_dataset = ProgressiveDataset(base_dataset, resolution)

    # return torch.utils.data.DataLoader(
    #     OverfitDataset('S.png', 'T.png'),
    #     batch_size=1,
    #     num_workers=config.training.num_workers,
    #     pin_memory=True
    # )
    
    # Split the dataset into training and validation
    train_size = int(0.8 * len(progressive_dataset))  # 80% for training
    val_size = len(progressive_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(progressive_dataset, [train_size, val_size])
    
    if is_validation:
        dataset = val_dataset
        batch_size = config.training.eval_batch_size
        shuffle = False
    else:
        dataset = train_dataset
        batch_size = config['training']['batch_size']
        shuffle = True

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers= config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script for handling emopath and eapp_path arguments.")
    
    parser.add_argument('--config_path', type=str, default='./config/local_train.yaml', help='Path to the config')
    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print("Config path is None")
        assert False


    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Start with the highest resolution
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


    def get_base_dataset(preprocess):
        test =  AffectNetDataset(
            root_dir="/media/oem/12TB/AffectNet/train",
            preprocess=preprocess,
            remove_background=False,
            use_greenscreen=False,
            cache_dir='/media/oem/12TB/AffectNet/train/cache'
        )
        return test
        # return CelebADataset(config.dataset.name, config.dataset.split, preprocess)

      # Load the dataset
    base_dataset = get_base_dataset(preprocess)

    train_dataloader = create_progressive_dataloader(config, base_dataset, 64, is_validation=False)
    # val_dataloader = create_progressive_dataloader(config, base_dataset, 64, is_validation=True)



    p = Portrait(config)

    train_model(config, p, train_dataloader)


if __name__ == '__main__':
    main()
