import torch.nn as nn

class DeepfakeClassifier(nn.Module):

    def __init__(self,input_size=768):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,64),
            nn.ReLU(),

            nn.Linear(64,2)
        )

    def forward(self,x):
        return self.model(x)