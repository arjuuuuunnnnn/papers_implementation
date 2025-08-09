import torch
import torch.nn as nn
import math



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, num_filters=3, downsample=True):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        
        if labels:
            self.label_mlp = nn.Linear(1, channels_out)
        
        self.downsample = downsample
        
        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out, num_filters, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)
            
        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)
        
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time[(..., ) + (None, ) * 2]
        
        if self.labels:
            label = kwargs.get('labels')
            o_label = self.relu(self.label_mlp(label))
            o = o + o_label[(..., ) + (None, ) * 2]
            
        o = self.bnorm2(self.relu(self.conv2(o)))
        return self.final(o)

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_channels = config["model"]["image_channels"]
        self.time_embedding_dims = config["model"]["time_embedding_dims"]
        self.labels = config["model"]["labels"]
        self.sequence_channels = config["model"]["sequence_channels"]
        
        self.downsampling = nn.ModuleList([
            Block(channels_in, channels_out, self.time_embedding_dims, self.labels)
            for channels_in, channels_out in zip(self.sequence_channels, self.sequence_channels[1:])
        ])
        
        self.upsampling = nn.ModuleList([
            Block(channels_in, channels_out, self.time_embedding_dims, self.labels, downsample=False)
            for channels_in, channels_out in zip(self.sequence_channels[::-1], self.sequence_channels[::-1][1:])
        ])
        
        self.conv1 = nn.Conv2d(self.img_channels, self.sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(self.sequence_channels[0], self.img_channels, 1)

    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
            
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)
            
        return self.conv2(o)

