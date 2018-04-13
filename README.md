## Install dependencies

    pip --user install pillow

## Running

    python mnist.py --train_epochs=20

Use train_epochs=40 for greater accuracy.

## Predicting

After running successfully, the folder `trained_model` will exist and contain the trained model.
You can use the trained model to make predictions:

* Create a file image.png (28x28, grayscale, white on black)
* Run the model

      python predict.py
