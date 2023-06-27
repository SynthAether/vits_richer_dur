import torch.nn as nn

from .layers import LayerNorm


class Layer(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm = LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x * mask
    

class DurationPredictor(nn.Module):
    def __init__(self, channels, kernel_size, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(channels, kernel_size)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Conv1d(channels, 1, 1)
        
    def forward(self, x, mask):
        x = x.detach()
        for layer in self.layers:
            x = layer(x, mask)
            x = self.dropout(x)
        x = self.out_layer(x) * mask
        return x
