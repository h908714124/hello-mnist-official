from official.utils.arg_parsers import parsers

import argparse

class MNISTArgParser(argparse.ArgumentParser):
  """Argument parser for running MNIST model."""
  
  def __init__(self):
    super(MNISTArgParser, self).__init__(parents=[
      parsers.BaseParser(),
      parsers.ImageModelParser(),
      parsers.ExportParser(),
    ])
    
    self.set_defaults(
      export_dir='trained_model',
      data_dir='mnist_data',
      model_dir='/tmp/mnist_model',
      batch_size=100,
      train_epochs=10)
