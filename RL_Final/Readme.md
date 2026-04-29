pyb_env.py
This script handles the simulation - it starts the PyBullet physics environment, sets physics 
values such as friction between the wheels and gravity, and collects observation values 
for the RL agent to receive

gym_env.py
This contains class for the Stable-Baselines agent to interact with the physics 
environment, this class contains the functions reset(), and step(), the agent.py script
calls step() until termination/truncation conditions are met, then reset() is called

agent.py
This contains the training loop. This loads the neural network for training and calls
from the gym_env class to interact with the pyb_env.

balro4.urdf
This is the universal robot description format file which contains an accurate model of
our robot in dimenstion and weight for simulation

balro_sac_model_38.zip
This file contains the neural network, it contains all of the model weights which are tuned
by training

balro_norm_sstats_38.pkl
This file contains the running mean and variance for all observation values which is used 
for normalization during training, SAC works best with observation values normalized to
+-1
