import os 
import sys 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from facenet import FaceNet
from loss import ArcFaceSoftmaxLinear
from utils import check_folder
from progressbar import Percentage, Bar, Timer, ETA, ProgressBar


class Trainer():
  def __init__(self, config):
    self.config = config
    self.model = FaceNet(config).model
    self.train_ds, self.nrof_train = create_tfrecord_dataset(
      tfrecord_dir=config.datasets,
      batch_size=config.batch_size,
      mode='train'
    )

    self.val_ds, self.nrof_val = create_tfrecord_dataset(
      tfrecord_dir=config.datasets,
      batch_size=config.batch_size,
      mode='validation'
    )

    self.lr_scheduler = schedules.ExponentialDecay(
      config.learning_rate,
      decay_steps=1000,
      decay_rate=.96,
      staircase=True
    )

    self.optimizer = Adam(learning_rate=self.lr_scheduler, epsilon=1e-3)
    self.checkpoint = tf.train.Checkpoint(
      epoch=tf.Variable(0, dtype=tf.int64),
      n_iter=tf.Variable(0, dtype=tf.int64),
      best_pred=tf.Variable(.0, dtype=tf.float32),
      optimizer=self.optimizer,
      model=self.model
    )

    self.manager = tf.train.CheckpointManager(self.checkpoint, config.checkpoint_dir, max_to_keep=3)
    check_folder(config.log_dir)
    self.train_summary_writer = tf.summary.create_file_writer(config.log_dir)

  def train_one_step(self, train_acc_metrics, loss_layer, batch_examples, trainable_variables):
    with tf.GradientTape() as tape:
      batch_images, batch_labels = batch_examples
      features = self.model(batch_images, training=True)
      embedding = tf.math.l2_normalize(features, axis=1, epsilon=1e-10)
      logits = loss_layer(embedding, batch_labels)
      loss = SparseCategoricalCrossentropy(from_logits=True)(batch_labels, logits)
      train_acc_metrics(batch_labels, logits)
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,trainable_variables))
    return loss

  def training(self, epoch):
    loss_layer = ArcFaceSoftmaxLinear(
      self.config.num_classes,
      self.config.embedding_size,
      self.config.margin,
      self.config.feature_scale
    )
    trainable_variables = []
    trainable_variables.extend(loss_layer.trainable_variables)
    trainable_variables.extend()
    train_acc_metric = SparseCategoricalAccuracy()
    widgets = ['train : ', Percentage, ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=int(self.nrof_train//self.config.batch_size)+1).start()
    for _, batch_examples in pbar(enumerate(self.train_ds)):
      loss = self.train_one_step(train_acc_metric, loss_layer, batch_examples, trainable_variables)
      with self.train_summary_writer.as_default():
        tf.summary.scalar('Total loss: ', loss, self.checkpoint.n_iter)
      self.checkpoint.n_iter.assign_add(1)
    pbar.finish() 
    train_acc = train_acc_metric.result()
    print('\nTraining acc over epoch {}: {:.4f}'.format(epoch, train_acc))
    with self.train_summary_writer.as_default():
      tf.summary.scalar('train/acc', train_acc_metric.result(), self.checkpoint.epoch)
    train_acc_metric.reset_states()
    save_path = self.manager.save()
    print('save checkpoint to {}'.format(save_path))

  def validate(self, epoch):
    widgets = ['validate :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, max_value=int(self.nrof_val//self.config.batch_size)+1).start()
    val_acc_metric = SparseCategoricalAccuracy()
    for _, (batch_images_validate, batch_labels_validate) in pbar(enumerate(self.val_ds)):
      prediction = self.model(batch_images_validate)
      val_acc_metric(batch_labels_validate, prediction)
    pbar.finish() 
    val_acc = val_acc_metric.result()
    print('\nvalidate acc over epoch {}: {:.4f}'.format(epoch, val_acc))
    with self.train_summary_writer.as_default():
      tf.summary.scalar('val/acc', val_acc_metric.result(),self.checkpoint.epoch)
    self.checkpoint.epoch.assign_add(1)    
    val_acc_metric.reset_states()

    if(val_acc > self.checkpoint.best_pred):
      self.checkpoint.best_pred = val_acc
      with open(os.path.join(self.checkpoint_dir, 'best_pred.txt'), 'w') as f:
        f.write(str(self.best_pred))
      self.model.save(os.path.join(self.checkpoint_dir, 'best_model.h5'))