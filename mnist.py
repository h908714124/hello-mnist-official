from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

from mnist_arg_parser import MNISTArgParser
from official.mnist import dataset


def cnn_model_fn(features, labels, mode):
  """
  https://www.tensorflow.org/tutorials/layers
  """
  image = features['image'] if isinstance(features, dict) else features
  
  # Input Layer
  input_layer = tf.reshape(image, [-1, 28, 28, 1])
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
  
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  
  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  
  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      export_outputs={
        'classify': tf.estimator.export.PredictOutput(predictions)
      })
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)
  
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
      labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(flags):
  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=flags.model_dir)
  
  # Set up training and evaluation input functions.
  def train_input_fn():
    ds = dataset.train(flags.data_dir)
    ds = ds.cache().shuffle(buffer_size=50000).batch(flags.batch_size)
    return ds
  
  def eval_input_fn():
    return dataset.test(flags.data_dir).batch(
      flags.batch_size).make_one_shot_iterator().get_next()
  
  # Train and evaluate model.
  for _ in range(flags.train_epochs):
    mnist_classifier
    mnist_classifier.train(input_fn=train_input_fn)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print('\nEvaluation results:\n\t%s\n' % eval_results)
  
  # Export the model
  if flags.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
      'image': image,
    })
    export_dir = mnist_classifier.export_savedmodel(flags.export_dir, input_fn)
    print('\nExported to:\n\t%s\n' % export_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  flags = MNISTArgParser().parse_args(args=sys.argv[1:])
  main(flags)
