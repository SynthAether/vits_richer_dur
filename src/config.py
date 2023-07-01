from text import num_vocab


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


train = DotDict(
    seed=42,
    num_epoch=10000,
    batch_size=16,
    save_ckpt_interval=500,
)

loss_coef = DotDict(
    mel=45,
    fm=2
)

mel_dim = 80
spec_dim = 513
sample_rate = 24000
hop_length = 256
segment_size = 64
sample_segment_size = segment_size * hop_length

audio = DotDict(
    sample_rate=sample_rate,
    n_fft=1024,
    win_length=1024,
    hop_length=hop_length,
    power=1,
    f_min=0.0,
    f_max=sample_rate//2,
    n_mels=mel_dim,
    mel_scale='slaney',
    norm='slaney',
    center=False
)

channels = 192

vits = DotDict(
    segment_size=segment_size,
    hop_length=hop_length
)
vits.phoneme_encoder = DotDict(
    num_vocab=num_vocab(),
    channels=channels,
    num_head=2,
    num_layers=6,
    kernel_size=3,
    dropout=0.1,
    window_size=4
)
vits.duration_predictor = DotDict(
    channels=channels,
    kernel_size=3,
    dropout=0.3,
    num_layers=3
)
vits.flow = DotDict(
    channels=channels, 
    kernel_size=5,
    dilation_rate=1, 
    n_layers=4, 
    n_flows=4
)
vits.posterior_encoder = DotDict(
    in_channels=spec_dim, 
    hidden_channels=channels, 
    out_channels=channels, 
    kernel_size=5, 
    dilation_rate=1, 
    num_layers=16
)
vits.vocoder = DotDict(
    in_channel=channels, 
    upsample_initial_channel=512, 
    upsample_rates=[8, 8, 2, 2],
    upsample_kernel_sizes=[16, 16, 4, 4],
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]
)

optimizer_g = DotDict(
    lr=2e-4,
    betas=(0.8, 0.99),
    eps=1e-9
)
optimizer_d = DotDict(
    lr=2e-4,
    betas=(0.8, 0.99),
    eps=1e-9
)

scheduler_g = DotDict(
    gamma=0.999875
)
scheduler_d = DotDict(
    gamma=0.999875
)
