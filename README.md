# NIPS-2018-AI-for-Prosthetics

Code for the 7th-placed solution to the 2018 NIPS AI For Prosthetics competition: http://osim-rl.stanford.edu/docs/nips2018/. OpenAI's Baselines implementation of PPO serves as the basis for the learning algorithm. To speed up training, state trajectories at different walking speeds are included in the osim-rl/osim/data folder.

## Installation
1. Clone repo

```
git clone --recursive https://github.com/lancerane/NIPS-2018-AI-for-Prosthetics.git
```

2. Opensim requirements

```
conda create -n opensim-rl -c kidzik opensim python=3.6.1
source activate opensim-rl
conda install -c conda-forge lapack git
```

3. Prerequisites for baselines (Ubuntu)

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

```
And on Mac OS:

```
brew install cmake openmpi
```
Tensorflow:

```
pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
```
or:
```
pip install tensorflow
```
For more info, see:

https://github.com/openai/baselines


4. Install mpi and modified baselines

```
cd NIPS-2018-AI-for-Prosthetics
cd baselines
conda install mpi4py
pip install -e .
```
5. Install modified Opensim modules

```
cd ..
cd osim-rl
pip install -e .
```

## Usage
main.py defines the entry-point. Training can be distributed across 4 workers with:

```
mpirun -np 4 python main.py
```

Training for approximately 10 hours with 4 workers produces a model with lifelike walking capable of scoring around 9700 in the test environment. 


![Alt Text](https://github.com/lancerane/NIPS-2018-AI-for-Prosthetics/blob/master/out.gif)


Further training with removal of imitation from the reward function will improve scores further; a score of 9853 was reached in the competition.

Please note:
- Changing the number of workers will necessitate alteration of the batch size and total timestep parameters in main.py.
- For speed, the integrator was changed to SemiExplicitEuler2 and the accuracy decreased to 1e-3. To fully replicate the test evaluation environment requires resetting these changes back to their defaults; in practice doing so was found to give a slight boost to the achieved score.
