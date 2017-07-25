PyTorch-CNS
==============
A work-in-progress implementation of [Compressed Network Search](http://people.idsia.ch/~juergen/compressednetworksearch.html)
for PyTorch models.

This creates a genome per layer, rather than a single one for the entire model as is described in the paper.

Usage
-----
```
from cnslib.population import Population
from yourcool.lib import Model
...
population = Population(lambda: Model(), yourconfig.num_models, yourconfig.cuda)
...
criterion = nn.BCELoss()
...training loop...
population.generation(batch_input, batch_output, criterion)  # update the population
best_model = population.best_model()  # current best model
```
Examples
--------

 * Download this repo
 * `pip install .` inside repo

`cnsdcgan.py`: DCGAN adapted from the PyTorch DCGAN example. Trains both
the discriminator and generator with compressed network search.
