import torch
import torch.nn as nn
import numpy as np

class resnet(nn.Module):
    def __init__(self, height):
        super(resnet, self).__init__()
        self.height = height
        self.conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 5, 3, 1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(5, 10, 3, 1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(10, 5, 3, 1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(5, 2, 3, 1), nn.ReLU()),
        ])
        self.res_conv = nn.ModuleList([
            nn.Conv2d(3, 5, 3, 1),
            nn.Conv2d(5, 10, 3, 1),
            nn.Conv2d(10, 5, 3, 1),
            nn.Conv2d(5, 2, 3, 1),
        ])

    def model(self, image):
        B = image.size(0)
        for layer, res_layer in zip(self.conv, self.res_conv):
            old_image = res_layer(image)
            image = layer(image)
            image += old_image
        
        image = image.view(B, 2, -1)
        roll_angle = image[:, 0].max()
        pitch_angle = image[:, 1].max()
        
        return roll_angle, pitch_angle
    
    def forward(self, image, keypoint_coords):
        # roll, pitch = self.model(image)
        left_eye = np.array(keypoint_coords[1][:2])
        right_eye = np.array(keypoint_coords[2][:2])
        left_shoulder = np.array(keypoint_coords[5][:2])
        right_shoulder = np.array(keypoint_coords[6][:2])

        eye_vector = right_eye - left_eye

        roll = np.arctan2(eye_vector[1], eye_vector[0]) * (180 / np.pi)

        if roll > 180:
            roll -= 360
        roll += 90

        shoulder_mid = (left_shoulder + right_shoulder) / 2
        eye_mid = (left_eye + right_eye) / 2
        mid_vector = eye_mid - shoulder_mid
        pitch = np.arctan2(mid_vector[1], mid_vector[0]) * (180 / np.pi) - roll

        pitch = -pitch
        pitch += 180
        if pitch > 180:
            pitch -= 360
        pitch += self.height / 30

        return roll, pitch

def get_resnet(height):
    model = resnet(height)
    # model.load_state_dict(torch.load('_models/resnet.pt'))
    return model