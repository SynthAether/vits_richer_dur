import torch
import config as cfg
from model import VITSModule


m = VITSModule.load_from_checkpoint('../out/ckpt/last.ckpt', params=cfg)
m = m.net_g.vocoder
print(m)
