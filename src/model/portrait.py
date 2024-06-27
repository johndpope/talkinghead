import random
import os
import wandb

import tqdm

from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

from PIL import Image
import numpy as np

import torchvision.models as models
from src.model.generator import StyleBased_Generator

class Portrait(nn.Module):
    def __init__(self, config):
        super(Portrait, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.pose_encoder = models.resnet50(pretrained=False)
        self.iden_encoder = models.resnet50(pretrained=False)
        self.emot_encoder = models.resnet50(pretrained=False)

        self.generator = StyleBased_Generator()


    def forward(self, Xs, Xd):
        batch_size = Xs.shape[0]
        Ep = self.pose_encoder(Xs)
        Ei = self.iden_encoder(Xs)
        Ee = self.emot_encoder(Xs)

        print(Ep.shape, Ei.shape, Ee.shape)
        
        print(Ep.shape, Ei.shape, Ee.shape)

        generaator_input = torch.cat([Ep, Ei, Ee])
        return None

if __name__ == '__main__':

    model = Portrait(None, None)

    # Create an instance of the model

    # Create some dummy input data
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    input_data = video_dataset[0][0:3]
    input_data2 = video_dataset[0][1:4]
    print(input_data.shape)

    # Pass the input data through the model
    output = model(input_data, input_data2)

    # Print the shape of the output
    print(output.shape)
