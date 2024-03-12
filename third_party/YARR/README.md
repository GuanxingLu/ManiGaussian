![Logo Missing](logo.png)

**Note**: Pirate qualification not needed to use this library.

YARR is **Y**et **A**nother **R**obotics and **R**einforcement learning framework for PyTorch.

The framework allows for asynchronous training (i.e. agent and learner running in separate processes), which makes it suitable for robot learning.
For an example of how to use this framework, see my [Attention-driven Robot Manipulation (ARM) repo](https://github.com/stepjam/ARM).

This project is mostly intended for my personal use (Stephen James) and facilitate my research.

## Modifcations

This is my (Mohit Shridhar) fork of YARR. Honestly, I don't understand what exactly is happening in a lot of places, so there a lot of hacks to make it work for my purposes. If you are doing simple behavior cloning, you can probably write simpler training and evaluation routines, but YARR might be useful if you also want to do RL. Here is a quick summary of my modifcations:

- Switched from randomly sampling evaluation episodes to deterministic reloading of val/test dataset episodes for one-to-one comparisons across models.
- Separated training and evaluation routines. 
- Task-uniform replay buffer for multi-task training. Each batch has a uniform distribution of tasks. 
- Added cinematic recorder for rollouts.
- Some other weird hacks to prevent memory leaks.

## Install

Ensure you have [PyTorch installed](https://pytorch.org/get-started/locally/).
Then simply run:
```bash
python setup.py develop
```
