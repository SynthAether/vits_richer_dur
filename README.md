# VITS

My implementation of VITS([paper](https://arxiv.org/abs/2106.06103)) for JSUT([link](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)).

Difference between originals and this implementation is 
- not using MonotonicAlignmentSearch because I use GT durations.

# Usage
Running run.sh will automatically download the data and begin training.

```sh
cd scripts
./run.sh
```

synthesize.sh uses last.ckpt by default, so if you want to use a specific weight, change it.

```sh
cd scripts
./synthesis.sh
```

# Requirements
```sh
pip install torch torchaudio lightning soundfile tqdm pyworld
```

# Result
WIP
