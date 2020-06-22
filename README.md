# Description

This repository provides an implementation of the CILP++ system from [1]. It
contains a copy of Aleph (obtained from the [The Aleph
Manual](https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html)),
that is used for the bottom clause propositionalization in the training pipeline.

It also includes an implementation of TREPAN [2] originally developed by [Kester
Jarvis](kester.jarvis@city.ac.uk) and [Artur d'Avila
Garcez](a.garcez@city.ac.uk) for rule extraction from the trained neural
network.

The included datasets are:

1. Mutagenesis, Alzheimers from [here](https://www.doc.ic.ac.uk/~shm/Datasets/)
2. Trains, IMDb from [here](https://relational.fit.cvut.cz/)

# Instructions

Requirements:

1. Ubuntu, Debian or similar
2. [Anaconda](https://docs.anaconda.com/anaconda/install/) (or
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
3. [SWI Prolog](https://www.swi-prolog.org/)
4. The required GPU drivers (optional)

To get the code and setup the environment, run:

```sh
git clone https://github.com/vakker/CILP.git
cd CILP
conda env create -f environment.yml
conda activate cilp
```

To run the training:

```sh
python run.py ...
```

The following arguments are available:

```sh
--log-dir <log-dir>   # e.g. logs
--data-dir <data-dir> # e.g. datasets/muta/muta188
--max-epochs <max-epochs>
--n-splits <n-splits>
--no-cache            # don't get data from cache instead run BCP again
--use-gpu             # use GPU for MLP
--trepan              # run a single train/val split and then TREPAN instead of cross-val
--dedup               # keep only unique data samples
```

To plot the training curves:

```sh
python plot.py ...
```

With arguments:

```sh
--log-file <log-file>     # e.g. logs/7992926c.npz (generated during training)
--param-file <param-file> # e.g. logs/params.json (also generated during training)
--max-epochs <max-epochs> # limit the number of epochs for plotting
```

# References

[1] França, Manoel VM, Gerson Zaverucha, and Artur S. d’Avila Garcez. "Fast
relational learning using bottom clause propositionalization with artificial
neural networks." Machine learning 94.1 (2014): 81-104.

[2] Craven, Mark, and Jude W. Shavlik. "Extracting tree-structured
representations of trained networks." Advances in neural information processing
systems. 1996.
