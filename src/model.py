import random

import tqdm

from src.decoder.facedecoder import FaceDecoder
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader
from src.encoder.emocoder import get_trainable_emonet
from src.encoder.deep3dfacerecon import get_face_recon_model
from src.encoder.hopenet import get_model_hopenet
from src.encoder.arcface import get_model_arcface
from src.encoder.eapp import get_eapp_model
from src.train.loss import PortraitLoss
import torch
import torch.nn as nn
import dlib

class Portrait(nn.Module):
    def __init__(self, eapp_path="./models/eapp_path", emo_path="./models/emo_path"):
        super(Portrait, self).__init__()


        arcface_model_path = "./models/arcface2/model_ir_se50.pth"
        face3d_model_path = "./models/face3drecon.pth"
        hope_model_path = "./models/hopenet_robust_alpha1.pkl"

        self.detector = dlib.get_frontal_face_detector()

        self.face3d = get_face_recon_model(face3d_model_path)

        if eapp_path is not None:
            self.eapp = get_eapp_model(None, "cuda")
        else:
            self.eapp = get_eapp_model(None, "cuda")

        self.arcface = get_model_arcface(arcface_model_path)

        if emo_path is not None:
            self.emodel = get_trainable_emonet(emo_path)
        else:
            self.emodel = get_trainable_emonet(None)

        self.decoder = FaceDecoder()


        self.loss = PortraitLoss()


    def select_frames(self, video, device):
        # Randomly select two different frames
        frame_indices = random.sample(range(len(video)), 2)
        return video[frame_indices[0]].unsqueeze(0).to(device), video[frame_indices[1]].unsqueeze(0).to(device)


    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.L1Loss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(num_epochs):
            epoch_loss = 0
            # Wrap the training loader with tqdm for a progress bar
            train_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
            for Xs, Xd, Xs_prime, Xd_prime in train_iterator:
                Xs, Xd, Xs_prime, Xd_prime = Xs.to(device), Xd.to(device), Xs_prime.to(device), Xd_prime.to(device)
                optimizer.zero_grad()
                # Y_pred = self.forward(Xs, Xd)
                # Y_prime_pred = self.forward(Xs_prime, Xd_prime)
                pass
                # Define your actual Y_true and Y_prime_true here
                # Y_true = ...  # Define how you get Y_true
                # Y_prime_true = ...  # Similarly for the prime sets

                # loss = loss_function(Y_pred, Y_true) + loss_function(Y_prime_pred, Y_prime_true)
                # loss.backward()
                # optimizer.step()
                # print(f'Epoch {epoch+1}, Loss {loss.item()}')


    def forward(self, Xs, Xd):
        # input are images
        with torch.no_grad():

            coeffs_s = self.face3d(Xs, compute_render=False)
            coef_dict_s = self.face3d.facemodel.split_coeff(coeffs_s)
            r_s = coef_dict_s['angle']
            t_s = coef_dict_s['trans']
            e_s = self.arcface(Xs)

            coeffs_d = self.face3d(Xd, compute_render=False)
            coef_dict_d = self.face3d.facemodel.split_coeff(coeffs_d)
            r_d = coef_dict_d['angle']
            t_d = coef_dict_d['trans']
            e_d = self.arcface(Xd)
        
        v_s = self.eapp(Xs)
        z_s = self.emodel(Xs) # expression
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, r_s, t_s, z_s), (None, e_d, r_d, t_d, z_d))

        return Y



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
