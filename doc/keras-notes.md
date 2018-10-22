# Import modules
Open a notebook via `jupyter notebook` and make sure you switch to your `keras` kernel. Now import everything to make sure it works
```python
### Base imports
import xarray as xr
import tensorflow as tf

### Keras imports
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers

### sklearn imports
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
```

# 0. Create a data pipeline
Before we start building a feed-forward neural network we need training data. These are reference values used to update the model weights. We then need to create a data pipeline. This involved extracting the data (maybe from a remove server), transforming the data (maybe this involves removing bad data points), and loading the data into the network. This process is termed an ETL pipeline or extract, transform, and load pipeline. If your data is in netCDF format, then `xarray` is an excellent tool for loading and transforming data. To load data into the network we will use the  `tf.data` API.  

Here is an example of of how to use the `tf.data` API:

```python
import tensorflow as tf
# setup your data pipline here...
```
[Building a data pipeline](https://cs230-stanford.github.io/tensorflow-input-data.html)

# 1. Define architecture
This defines our network architecture. We assume 5 features, 1 dense hidden layer with 10 ReLU neurons, and a single output linear output neuron. We can use `model.summary()` for an overview of our network
```python
### Number of input features
n_features = 5

### Define sequential class
model = Sequential()

### Hidden Layer 1
model.add(Dense(10, input_shape=(n_features,),
                activation = 'relu',
                kernel_initializer = initializers.he_uniform(),
                bias_initializer = initializers.he_uniform(),
                name = 'hidden_layer1'))
### Output layer
model.add(Dense(1, activation='linear', name='output_layer'))

### prints a summary representation of your model.
model.summary()
```

# 2. Compile our model
Now we compile our model with a specified optimizer and loss function.  various metrics can specified to see how the model is performing during training. You can specify multiple metrics as once. Everybody uses MSE, but it penalized extreme differences harshly, MAE is little more conservative. It’s nice to show both.
```python
### Compile the model
### Optimizer = Adam optimizer with learning rate of 0.02
### loss = logcosh
### metrics = MSE and MAE
model.compile(optimizer = keras.optimizers.Adam(lr = 0.02),
              loss = 'logcosh',
              metrics = ['mean_squared_error', 'mean_absolute_error'])
```

# 3. Train our model
Trains the model for a fixed number of epochs (iterations on a dataset).
```python
### This trains our model with `X_train` as our inputs and `y_train` as our target
### validation_split = hold X% for validation.
### epochs = how many times the model sees our dataset
### batch_size = show the model this many training samples before calc loss
### verbose = output to screen
history_callback = model.fit(X_train, y_train,
                             validation_split = 0.1,
                             epochs = 2000,
                             batch_size = 512,
                             verbose = 2)
```

To see how our model is doing we can plot the loss metrics on the training and validation sets. The first few epochs have a crazy loss, because the model quite shitty to begin with.

```python
### Plot training loss in black
### Plot validation loss in red
plt.plot(history_callback.history['mean_absolute_error'][5:], color='k')
plt.plot(history_callback.history['val_mean_absolute_error'][5:], color='r')
```

# 4. Evaluate our model
This will return the loss value and metrics values for the model in test mode.
```python
### Evaluate our model on testing data
result = model.evaluate(x = X_test, y = y_test)
```

Now let’s compare our results to the testing data
```python
### This displays our metrics. showing how well the model performs
### on data it has never seen before
for name, value in zip(model.metrics_names, result):
    print(name, value)
    if name=='mean_squared_error':
        print('mean_error ',np.sqrt(value))
```

# 5. Make predictions
Generates output predictions for the input samples.

```python
pred = model.predict(X_test, verbose=1)
```

# 6. Save our model
Let’s save our model for later use or so someone else can run it
```python
model.save('FFN_1layer_v0.keras')
```


Here is how to load the saved model and make predictions with it
```python
### Make sure load_model is imported from keras
from tensorflow.keras.models import load_model

### Give the path to the .keras file
model = load_model('FFN_1layer_v0.keras')

### Make some predictions
pred = model.predict(X_test, verbose=1)
```

Return a dictionary containing the configuration of the model.
```python
model.get_config()
```
