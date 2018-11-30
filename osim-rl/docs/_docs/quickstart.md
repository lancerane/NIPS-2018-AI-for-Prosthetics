---
title: Quickstart
---

You need **Anaconda** to run `osim-rl` simulations. Anaconda will create a virtual environment with all the necessary libraries, to avoid conflicts with libraries in your operating system. You can get anaconda from here https://www.continuum.io/downloads. In the following instructions we assume that Anaconda is successfully installed.

We support Windows, Linux, and Mac OSX (all in 64-bit). To install our simulator, you first need to create a conda environment with the OpenSim package.

On **Windows**, open a command prompt and type:

    conda create -n opensim-rl -c kidzik opensim git python=3.6
    activate opensim-rl

On **Linux/OSX**, run:

    conda create -n opensim-rl -c kidzik opensim git python=3.6
    source activate opensim-rl

These commands will create a virtual environment on your computer with the necessary simulation libraries installed. Next, you need to install our python reinforcement learning environment. Type (on all platforms):

    conda install -c conda-forge lapack git
    pip install git+https://github.com/stanfordnmbl/osim-rl.git

If the command `python -c "import opensim"` runs smoothly, you are done! Otherwise, please refer to our [FAQ](#frequently-asked-questions) section.

Note that `source activate opensim-rl` activates the anaconda virtual environment. You need to type it every time you open a new terminal.

## Basic usage

To execute 200 iterations of the simulation enter the `python` interpreter and run the following:
```python
from osim.env import L2RunEnv

env = L2RunEnv(visualize=True)
observation = env.reset(difficulty = 0)
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
```
![Random walk](https://raw.githubusercontent.com/stanfordnmbl/osim-rl/1679344e509e29bdcc2ee368ddf83e868d93bf61/demo/random.gif)

The function `env.action_space.sample()` returns a random vector for muscle activations, so, in this example, muscles are activated randomly (red indicates an active muscle and blue an inactive muscle).  Clearly with this technique we won't go too far.

Your goal is to construct a controller, i.e. a function from the state space (current positions, velocities and accelerations of joints) to action space (muscle excitations), that will enable to model to travel as far as possible in a fixed amount of time. Suppose you trained a neural network mapping observations (the current state of the model) to actions (muscle excitations), i.e. you have a function `action = my_controller(observation)`, then
```python
# ...
total_reward = 0.0
for i in range(200):
    # make a step given by the controller and record the state and the reward
    observation, reward, done, info = env.step(my_controller(observation))
    total_reward += reward
    if done:
        break

# Your reward is
print("Total reward %f" % total_reward)
```
There are many ways to construct the function `my_controller(observation)`. We will show how to do it with a DDPG (Deep Deterministic Policy Gradients) algorithm, using `keras-rl`. If you already have experience with training reinforcement learning models, you can skip the next section and go to [evaluation](#evaluation).
