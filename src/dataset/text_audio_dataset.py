import torch
import torchaudio

from pathlib import Path

from audio import SpectrogramTransform
from text import text_to_sequence


class TextAudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, wav_dir, lab_dir, audio_config, split='|'):
        with open(file_path) as f:
            lines = f.readlines()
            data = list()
            for line in lines:
                bname, label = line.strip().split(split)
                data.append((bname, label))
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.lab_dir = Path(lab_dir)
        
        self.audio_config = audio_config
        self.hop_length = audio_config.hop_length
        self.sample_rate = audio_config.sample_rate

        self.spec_tfm = SpectrogramTransform(**audio_config)

    def preprocess(self, bname):
        with open(self.lab_dir / f'{bname}.lab', 'r') as f:
            fullcontext = f.readlines()
        s, e, _ = fullcontext[0].split()
        s, e = int(s), int(e)
        wav_s = int(e * 1e-7 * self.sample_rate)
        s, e, _ = fullcontext[-1].split()
        s, e = int(s), int(e)
        wav_e = int(s * 1e-7 * self.sample_rate)
        wav, _ = torchaudio.load(self.wav_dir / f'{bname}.wav')
        wav = wav[:, wav_s:wav_e]
        spec = self.spec_tfm.to_spec(wav).squeeze(0)
        return wav, spec

    def __getitem__(self, idx):
        bname, label = self.data[idx]

        phonemes = torch.LongTensor(text_to_sequence(label.split()))
        phonemes = phonemes[1:-1]

        wav, spec = self.preprocess(bname)

        return (
            bname,
            phonemes,
            wav,
            spec
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    (
        bnames,
        phonemes,
        wavs,
        specs
    ) = tuple(zip(*batch))

    B = len(bnames)
    x_lengths = [len(x) for x in phonemes]
    frame_lengths = [x.size(-1) for x in specs]
    sample_lengths = [x.size(-1) for x in wavs]

    x_max_length = max(x_lengths)
    frame_max_length = max(frame_lengths)
    sample_max_length = max(sample_lengths)
    spec_dim = specs[0].size(0)

    x_pad = torch.zeros(size=(B, x_max_length), dtype=torch.long)
    spec_pad = torch.zeros(size=(B, spec_dim, frame_max_length), dtype=torch.float)
    wav_pad = torch.zeros(size=(B, 1, sample_max_length), dtype=torch.float)
    for i in range(B):
        x_l, f_l, s_l = x_lengths[i], frame_lengths[i], sample_lengths[i]
        x_pad[i, :x_l] = phonemes[i]
        spec_pad[i, :, :f_l] = specs[i]
        wav_pad[i, :, :s_l] = wavs[i]

    x_lengths = torch.LongTensor(x_lengths)
    frame_lengths = torch.LongTensor(frame_lengths)
    sample_lengths = torch.LongTensor(sample_lengths)

    return (
        bnames,
        x_pad,
        x_lengths,
        wav_pad,
        sample_lengths,
        spec_pad,
        frame_lengths,
    )
