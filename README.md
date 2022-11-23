# ann-vs-cnn
ANN vs CNN - Comparison of 2 main networks

Results obtained on Mac M2 with tensorFlow usage

To compile TensorFlow on Mac M2 (November 2022), there is no other choice than Miniconda (bazel does not work at this time)

https://developer.apple.com/metal/tensorflow-plugin/

1. First miniconda

bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate
conda install -c apple tensorflow-deps

2. Virtual Environment

python3 -m venv ~/venv-metal
source ~/venv-metal/bin/activate
python -m pip install -U pip

3.  TensorFlow

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


** as expected CNN is far more accurate than ANN even with overfitting 

