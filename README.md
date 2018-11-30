# NIPS-2018-AI-for-Prosthetics

Code for the 7th-placed solution to the 2018 NIPS AI For Prosthetics competition. OpenAI's BAselines implementation of PPO serves as the basis for the learning algorithm. To speed up training, state trajectories at different walking speeds are included in the osim-rl/osim/data folder.

main.py defines the entry-point. Training can be distributed across 4 workers with:

```
mpirun -np 4 python main.py
```

Training for approximately 10 hours with 4 workers produces a model with lifelike walking capable of scoring around 9700 in the test environment. Further training with removal of imitation from the reward function will improve scores further; a score of 9853 was reached in the competition.

Please note:
- Changing the number of workers will necessitate alteration of the batch size and total timestep parameters in main.py.
- For speed, the integrator was changed to SemiExplicitEuler2 and the accuracy decreased to 1e-3. To fully replicate the test evaluation environment requires resetting these changes back to their defaults; in practice doing so was found to give a slight boost to the achieved score.
