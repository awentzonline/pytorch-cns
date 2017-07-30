PyTorch-CNS
==============
A work-in-progress implementation of [Compressed Network Search](http://people.idsia.ch/~juergen/compressednetworksearch.html)
for PyTorch models.

This creates a genome per layer, rather than a single one for the entire model as is described in the paper.

Install with `pip install pytorch-cns`

To do distributed agent evaluation (as in all of the examples) you'll need
to `pip install redis` and start a redis server where the gene pool will be
stored in an ordered set.

AI Gym Examples
---------------
 * `aigym.py`: CartPole!
 * `atari.py`: Atari ram-base games
 * `atari_pix.py`: Atari pixel-based games
 * `atari_pixrnn.py`: Atari pixel-based games with a recurrent neural network

Install additional requirements: `pip install gym atari_py box2d`

To run a pool of workers with default settings simply run the python file
(e.g. `python atari_pix.py`). If you make any changes to the hyperparameters
you'll to use the `--clear-store` flag which deletes the old gene pool upon start.
Use `--num-agents` to customize the number of child processes spawned.

Invoke the example with `python atari_pix.py --render --best` to run the simulation
with the fittest genome. This can be done at the same time as the workers are
running to monitor progress.

Image generation
----------------
These do not converge on anything at the moment. Maybe, if you are a real GANimal,
you can find the right configuration. The code here is hacked together from
the pytorch example repo.

`cnsdcgan.py`: DCGAN adapted from the PyTorch DCGAN example. Attempts to train
both the discriminator and generator with compressed network search.

`vggmse.py`: An autoencoder which uses VGG16 to calculate the loss.
