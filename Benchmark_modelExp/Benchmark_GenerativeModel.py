from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from torchvision import transforms
from torch import nn
from torch.nn import functional as F



class ModelLWF(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Flatten(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),            
            nn.Linear(256,output_dim)
        )

    def forward(self,x):
        x = x.reshape(-1, x.shape[1],1)
        return self.model(x)

class ModelEWC(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Flatten(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),            
            nn.Linear(256,output_dim)
        )

    def forward(self,x):
        x = x.reshape(-1, x.shape[1],1)
        return self.model(x)

class ModelSI(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Flatten(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),            
            nn.Linear(256,output_dim)
        )

    def forward(self,x):
        x = x.reshape(-1, x.shape[1],1)
        return self.model(x)

class ModelJoint(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Flatten(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),            
            nn.Linear(256,output_dim)
        )

    def forward(self,x):
        x = x.reshape(-1, x.shape[1],1)
        return self.model(x)
    
class ModelNaive(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.01,inplace=False),

            nn.Flatten(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),            
            nn.Linear(256,output_dim)
        )

    def forward(self,x):
        x = x.reshape(-1, x.shape[1],1)
        return self.model(x)