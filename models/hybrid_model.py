import torch
import torch.nn as nn

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(768+128,256),
            nn.ReLU(),

            nn.Linear(256,64),
            nn.ReLU(),

            nn.Linear(64,2)
        )

    def forward(self,wav_feat,cnn_feat):

        combined = torch.cat((wav_feat,cnn_feat),dim=1)

        return self.fc(combined)