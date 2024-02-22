# Introduction: Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame

This project provides a game environment and an RL agent playing the well-known game Snake.

# Setup

Create a virtual environment:
- `python3 -m virtualenv venv`
- `source venv/bin/activate`

Install Python dependencies:

- `python3 -m pip install torch`
- `python3 -m pip install numpy`
- `python3 -m pip install pygame`
- `python3 -m pip install matplotlib`

Note: Under Ubuntu 22.02, for some reason, you have to install these packages in order to visualize with matplotlib:
- `python3 -m pip install PyQt5`
- `python3 -m pip install "cython<3"`

Execute the application (PPO):
- python3 06-Snake-PPO.py

# Notes

- Danger of numeric instability if the rewards barely change (division by 0) is removed.

# References
This is a combination of two tutorials by Patrick Loeber (ref. [YouTube Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV), [GitHub Repository](https://github.com/patrickloeber/snake-ai-pytorch/tree/main)) introducing the Pygame environment and training with Deep Q Learning,
and Ben Trevett providing a guide through various reinforcement learning algorithms (ref. [GitHub Repository](https://github.com/bentrevett/pytorch-rl)).
