import random
import os
import wandb

import tqdm

from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.checkpoint import checkpoint

import torchvision.models as models
from src.model.generator import Generator, Discriminator
# from src.model.discriminator import MultiScalePatchDiscriminator

class Portrait(nn.Module):
    def __init__(self, config):
        super(Portrait, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.center_size = 224

        self.discriminator = Discriminator(512)

          # IRFD encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder

        self.irfd_generator = Generator(128*3, 512)  # 1536 = 512*3 for identity, emotion, and pose

  

        self.to(self.device)

    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        # return nn.Sequential(*list(encoder.children())[:-1])
        encoder.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Tanh()
        )
        return encoder



    def decode(self, fi_s, fe_s, fp_s, alpha, step, zero_noise=False):
        Y = self.irfd_generator(torch.cat([fi_s, fe_s, fp_s], dim=1), alpha, step, zero_noise)
        return Y

    def encode(self, X):
        fi_s = self.Ei(X)
        fe_s = self.Ep(X)
        fp_s = self.Ee(X)
        return fi_s, fe_s, fp_s
    
    def discriminator_forward(self, X, alpha, step):
        return self.discriminator(X, alpha, step)

    def irfd_forward(self, x_s, x_t, alpha, step, zero_noise=False):
        # print(f"Input shapes: x_s: {x_s.shape}, x_t: {x_t.shape}")

        # Encode source and target images
        fi_s = checkpoint(self.Ei, x_s).squeeze()
        fe_s = checkpoint(self.Ee, x_s).squeeze()
        fp_s = checkpoint(self.Ep, x_s).squeeze()
        
        fi_t = checkpoint(self.Ei, x_t).squeeze()
        fe_t = checkpoint(self.Ee, x_t).squeeze()
        fp_t = checkpoint(self.Ep, x_t).squeeze()

        # print(f"Encoded shapes: fi_s: {fi_s.shape}, fe_s: {fe_s.shape}, fp_s: {fp_s.shape}")
        # print(f"Encoded shapes: fi_t: {fi_t.shape}, fe_t: {fe_t.shape}, fp_t: {fp_t.shape}")

        # Randomly swap one type of feature
        swap_type = torch.randint(0, 3, (1,)).item()
        if swap_type == 0:
            fi_s, fi_t = fi_t, fi_s
        elif swap_type == 1:
            fe_s, fe_t = fe_t, fe_s
        else:
            fp_s, fp_t = fp_t, fp_s

        # Concatenate features and generate reconstructed images
        features_s = torch.cat([fi_s, fe_s, fp_s], dim=1)
        features_t = torch.cat([fi_t, fe_t, fp_t], dim=1)

        # print(f"Concatenated feature shapes: features_s: {features_s.shape}, features_t: {features_t.shape}")

        reconstructed_s = self.irfd_generator(features_s, alpha, step, zero_noise)
        reconstructed_t = self.irfd_generator(features_t, alpha, step, zero_noise)

        # print(f"Reconstructed shapes: reconstructed_s: {reconstructed_s.shape}, reconstructed_t: {reconstructed_t.shape}")

        return reconstructed_s, reconstructed_t, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t


    def save_model(self, path, epoch, optimizer, current_resolution):
        if not os.path.exists(path):
            os.makedirs(path)
        model_state = {
            'epoch': epoch,
            'current_resolution':current_resolution ,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(self.state_dict(), path + "portrait.pth")
        torch.save(model_state, os.path.join(path, "checkpoint.pth"))

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    # Use the loaded config to initialize the model
    model = Portrait(None)
    model.train()  # Ensure model is in training mode

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create some dummy input data
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    input_data = video_dataset[0][0:2].to(model.device)
    input_data_backup = input_data.clone()  # Backup to check for modifications
    input_data_clone = input_data.clone()  # Clone to prevent modification
    input_data_clone.requires_grad = False

    # Forward pass to get initial outputs
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    output = model(input_data_clone, input_data_clone, 0.5, 6, zero_noise=True)
    print(max(output.flatten()), min(output.flatten()))
    discrim_out = model.discriminator_forward(output, 0.5, 6)
    loss = output.mean() + discrim_out[0][0].mean() # random losss doesnt matter
    loss.backward()
    optimizer.step()  # Update weights with backpropagation

    # Get encoder outputs after training
    trained_pose, trained_iden, trained_emot = model.pose_encoder(input_data_clone), model.iden_encoder(input_data_clone), model.emot_encoder(input_data_clone)
    trained_output = model(input_data_clone, input_data_clone, 0.5, 6,  zero_noise=True)
    trained_discrim_out = model.discriminator_forward(trained_output, 0.5, 6)

    # Save model
    saved_state_dict = model.state_dict()
    torch.save(model.state_dict(), 'portrait_model.pth')

    # Load model
    model_loaded = Portrait(None)

    # test unloaded  
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    loaded_output = model_loaded(input_data_clone, input_data_clone, 0.5,6,  zero_noise=True)
    discrim_out_loaded = model_loaded.discriminator_forward(output, 0.5, 6)
    loaded_pose, loaded_iden, loaded_emot = model_loaded.pose_encoder(input_data_clone), model_loaded.iden_encoder(input_data_clone), model_loaded.emot_encoder(input_data_clone)
    assert not torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(loaded_output, trained_output, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(trained_pose, loaded_pose, atol=1e-6), "Pose encoder outputs same before load."
    assert not torch.allclose(trained_iden, loaded_iden, atol=1e-6), "Identity encoder outputs same before load."
    assert not torch.allclose(trained_emot, loaded_emot, atol=1e-6), "Emotion encoder outputs same before load."
    del loaded_output, loaded_pose, loaded_iden, loaded_emot, discrim_out_loaded


    loaded_state_dict = torch.load('portrait_model.pth')
    model_loaded.load_state_dict(loaded_state_dict)

    saved_keys = set(saved_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())
    assert saved_keys == loaded_keys, "Mismatch in model state dict keys after loading."

    # Perform forward pass again with the loaded model
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    loaded_output = model_loaded(input_data_clone, input_data_clone, 0.5,6, zero_noise=True)
    loaded_pose, loaded_iden, loaded_emot = model_loaded.pose_encoder(input_data_clone), model_loaded.iden_encoder(input_data_clone), model_loaded.emot_encoder(input_data_clone)
    discrim_out_loaded = model_loaded.discriminator_forward(loaded_output, 0.5, 6)

    # Compare encoder outputs
    assert torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(loaded_output, trained_output, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(trained_pose, loaded_pose, atol=1e-6), "Pose encoder outputs differ after load."
    assert torch.allclose(trained_iden, loaded_iden, atol=1e-6), "Identity encoder outputs differ after load."
    assert torch.allclose(trained_emot, loaded_emot, atol=1e-6), "Emotion encoder outputs differ after load."
    
    del trained_output, loaded_pose, loaded_iden, loaded_emot, discrim_out_loaded, loaded_output



    print("All checks passed successfully. Encoder outputs are consistent after training and loading.")
