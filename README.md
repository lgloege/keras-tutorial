# Keras-tutorial

This repository contains material for a tensorflor-keras tutorial, part of the "Deep learning for climate modeling seminar".

# Setup TensorFlow Keras environment
We first need a tensorflow/keras development environment. These steps walk you through setting up an environment using Python 3.6.

1. First we need to create a new environment  `conda create --name keras`
2. Now we need to activate our environment `source activate keras`
3. Now let’s import all the modules we will use:
	1. This is so we can work with NetCDF files `conda install -c conda-forge xarray`
	2. This is so we can make plots `conda install -c conda-forge matplotlib`
	3. This is so we can use TensorFlow and Keras  `conda install -c conda-forge tensorflow`
	4. This is a  nice machine learning library. `conda install -c conda-forge scikit-learn`
	5. This  is so our environment can be a Jupyter kernel `conda install ipykernel`
4. Now make your new environment a Jupyter kernel: `python -m ipykernel install --user --name keras --display-name "keras"`
5. Now open a Jupyter notebook and make sure you can import everything. You may need to change to your new keras kernel:  

``` python
import xarray as xr
import numpy as np
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
```

# notebooks
This is a collection of examples using the keras API.
Notebooks can be accessed and modified via binder [here](https://mybinder.org/v2/gh/lgloege/keras-tutorial/master?filepath=notebooks)

# doc
This contains documentation files:
- [`setting_up_a_keras_environment.md`](https://github.com/lgloege/keras-tutorial/blob/master/doc/setup-environment.md)
- [`neural-network-theory.md`](https://github.com/lgloege/keras-tutorial/blob/master/doc/neural-network-theory.md)
