#### Forked from:
[OpenAI/Baselines](https://github.com/openai/baselines): "a set of high-quality implementations of reinforcement learning algorithms"


# Mean Actor-Critic


This repository has been customized to run Atari experiments comparing Mean Actor-Critic (MAC) with Advantage Actor-Critic (AAC). The details of these customizations can be found in our Arxiv paper: https://arxiv.org/abs/1709.00503.

### Installation

If you're on MacOS, you may need to install some dependencies first:
```
brew install cmake openmpi
```

Then grab the code:
```bash
git clone https://github.com/camall3n/atari-MAC.git
cd atari-MAC
```

Optionally, create a python3 virtualenv and activate it here:
```
virtualenv env --python=python3
. env/bin/activate
```

And finally install OpenAI's baselines package:
```
pip install -e .
```

### Running

First choose the appropriate git branch, either `mac` or `benchmark-aac`:
```bash
git checkout mac
# git checkout benchmark-aac
```

To run an experiment using the hyperparameters from the paper use:
```
./run_atari
```

This will call `python -m baselines/a2c/run_atari.py` and forward the arguments to the python script. For information on how to specify the arguments, use:
```
./run_atari --help
```

The script will train for the specified number of frames, and it will periodically log training progress, evaluate the network, and save model weights.

A new directory is created for each experiment, and you can use the `--note` argument to tag the log files with the details of each training run.
