# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:37:51 2022

@author: alber
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

df0 = pd.read_csv('data.csv')
df05 = df0
sorted(df0)   # Just to see col names

drop_list = ['IG_OAS_6M_Up',
 'HY_OAS_1M_Up',
 'HY_OAS_3M_Up',
 'HY_OAS_6M_Up',
 'IG_OAS_1M_Up',
 'IG_OAS_3M_Up',
 'Date',
 'Day',
 'DayWeek',
 'HY_ER',
 'HY_OAS_1M',
 'HY_OAS_3M',
 'HY_OAS_6M',
 'IG_ER',
 'IG_OAS',
 'IG_OAS_1M',
 'IG_OAS_3M',
 'IG_OAS_6M',
 'Quarter',
 'Month']

df = df0.drop(drop_list, axis = 1)
column_indices = {name: i for i, name in enumerate(df.columns)}     # Dictionary with column indices


# Time of month Signal

date_time = pd.to_datetime(df05.pop('Date'), format='%Y-%m-%d')
timestamp_s = np.array(range(1,len(date_time)+1))

yr4 = 253*4                                         # Trading Days


df['4_yr_sin'] = np.sin(timestamp_s * (2 * np.pi / yr4))
df['4_yr_cos'] = np.cos(timestamp_s * (2 * np.pi / yr4))

# Train/Test

train_df = df[:4795]        # Year 2000 to 2019
val_df = df[4795:5056]          # Year 2019 to 2020
test_df = df[5056:5356]         # Year 2020 to 2021

# Scaling

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std



# Window Maker

# Window Maker

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

#### How you should use it ####
#  WindowGenerator(input_width=6, label_width=1, shift=1,
#                      label_columns=['HY_OAS'])


# Splitting the window

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# Creating a tf.data.Dataset

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

# Add properties

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.test)) 
    return result

@property
def example0(self):
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.val)) 
    return result

WindowGenerator.example0 = example0
WindowGenerator.example = example
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test

# Compiler

MAX_EPOCHS = 130
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


# Define the Plot Method

def plot(self, model=None, plot_col='HY_OAS', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

# Plot Val

def plot0(self, model=None, plot_col='HY_OAS', max_subplots=3):
  inputs, labels = self.example0
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot0 = plot0

