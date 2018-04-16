## Install dependencies

    pip --user install pillow

## Running

    rm -rf /tmp/mnist_model
    python mnist.py --train_epochs=20

Use `--train_epochs=40` for greater accuracy, or `--train_epochs=1` for speed.

Try `--model_dir=/tmp/some_unused_path` if there's an unexpected exception,
or delete the current `model_dir` (defaults to `/tmp/mnist_model`).

## Predicting

After running successfully, the folder `trained_model` will exist and contain the trained model.
You can use the trained model to make predictions:

* Create a file image.png (28x28, grayscale, white on black)
* Run the model

      python predict.py
