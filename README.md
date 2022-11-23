# ANN vs CNN - Comparison of 2 main networks


Results obtained on Mac M2 with tensorFlow usage on Visual Studio Code with Jupyter extension


## First install TensorFlow on Apple M2

To compile TensorFlow on Apple M2 (November 2022), there is no other choice than Miniconda (bazel does not work at this time)

https://developer.apple.com/metal/tensorflow-plugin/

1. First miniconda

bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate
conda install -c apple tensorflow-deps

2. Virtual Environment

Create venv-metal with Visual Studio Code
conda activate venv-metal
python -m pip install -U pip 

=> assume this is the conda's python now and below

1.  TensorFlow

python -m pip install tensorflow-macos

4.  Metal plugin

python -m pip install tensorflow-metal


5. Test

Open activity monitor 

Launch python in terminal and copy / paste 

import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)

=> see the GPU usage on M2


## Install Visual Keras

This help to visualize the different layers of different Networks

Installing Visual Keras is a little it touchy in the venv-metal

After activating venv-metal, install graphviz

conda install graphviz  (here 2.50.0 has been installed)

then 

pip install pydot
pip install pydotplus

=> relaunch Visual Studio Code


## TensorBoard

TensorBoard is really an interesting interactive tool. There are a lot of possibilities inside to analyse NN and the different behaviors

pip install tensorboard


## Compare the 2 networks CNN and ANN

as expected CNN is far more accurate than ANN even with overfitting 

4 models have been tested : 2 ANN and 2 CNN

=> the overall success rate is for CNN with more than 90% of success instead of 77% for ANN

