import argparse
import sys
import glob
import os

import numpy as np
from PIL import Image
from tensorflow.contrib import predictor


def load(input_file):
  img = Image.open(input_file).load()
  m = np.zeros((28, 28), np.float32)
  for i in range(0, 28):
    for j in range(0, 28):
      m[i, j] = 0.00390625 * img[j, i]
  return [m]


class MyParser(argparse.ArgumentParser):
  def __init__(self):
    super(MyParser, self).__init__()
    self.add_argument("--export_dir")
    self.add_argument("--input_file")
    self.set_defaults(
      export_dir='trained_model',
      input_file='input.png')


if __name__ == '__main__':
  parser = MyParser()
  flags = parser.parse_args(args=sys.argv[1:])
  list_of_files = glob.glob(flags.export_dir + '/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  predict_fn = predictor.from_saved_model(latest_file)
  predictions = predict_fn({'image': load(flags.input_file)})
  print('Prediction: %d' % (np.argmax(predictions['probabilities'][0])))
