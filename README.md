PyTorch-CNS
==============
A work-in-progress implementation of [Compressed Network Search](http://people.idsia.ch/~juergen/compressednetworksearch.html)
for PyTorch models.

This creates a genome per layer, rather than a single one for the entire model as is described in the paper.

Install with `pip install pytorch-cns`

To do distributed agent evaluation (as in all of the examples) you'll need
to install `redis`. All of the AI Gym examples require `gym`.

Examples
--------
So far, I've used the aigym example to solve CartPole and LunarLander.
The Pong-ram atari example also begins converging but I haven't waited
long enough to see if the stock hyperparameters actually allow it to
solve the game.
The image generation examples haven't yielded anything good for me, yet.

`aigym.py`: Evolve a group of agents to solve OpenAI Gym environments.

`atari.py`: Solve atari ram-base games in the OpenAI Gym.

`atari_pix.py`: Solve atari pixel-based games in the OpenAI Gym

`cnsdcgan.py`: DCGAN adapted from the PyTorch DCGAN example. Attempts to train
both the discriminator and generator with compressed network search.

`vggmse.py`: An autoencoder which uses VGG16 to calculate the loss.
