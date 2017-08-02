PyTorch-CNS
==============
An implementation of [Generalized Compressed Network Search](http://people.idsia.ch/~juergen/compressednetworksearch.html)
for PyTorch models.

This creates a genome per layer, rather than a single one for the entire model as is described in the paper.

There are two optimizers:
 * Asynchronous Gene Pool: a master list of genomes, sorted by fitness, is worked
 on by a pool of agents which draw fit genomes for mutation and evaluation.
 * Synchronous Score Swap: each worker maintains a full copy of all genomes and
 only fitness scores are exchanged.

Installation
------------
Install python package with `pip install pytorch-cns`

Install [redis](https://redis.io/) which is used as the datastore. The examples
expect a redis instance listening on localhost at the default port. You can change
this by passing a JSON dict of `StrictRedis` kwargs via the `--redis-params` command
line option.

AI Gym Examples
---------------
 * `aigym.py`: CartPole!
 * `atari.py`: Atari ram-based games
 * `atari_pix.py`: Atari pixel-based games
 * `atari_pixrnn_gpa.py`: Atari pixel-based games with a recurrent neural network,
 using the asynchronous gene pool optimizer.
 * `atari_pixrnn_ss.py`: Atari pixel-based games with a recurrent neural network,
 using the synchronous

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
