import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import json
import os
import time
import datetime
import itertools

import nltk
nltk.download('wordnet')
#from nltk.stem import WordNetLemmatizer
import re



def preprocessing(data):
  # 불필요한 문자(열) 제거
  stopword = ['(adjective)','(noun)','(adverb)','(verb)',
          '/colour','/harbour',"'s","’s",";",":","[","]",
          '(',')','-',"'","’",'/',".",'“','”','?']
  for s in stopword:
    data = data.replace(s, ' ')
  
  data = data.replace('tritones', 'tritone')
  data = ' '.join(data.split()) # 중복 공백 제거
  data = data.lower() # 모두 소문자로
  #print(data)

  # Lemmatization 원형 (lemma) 찾기
  lem = nltk.WordNetLemmatizer()
  lemmatized_words = ''
  for word in data.split(' '):
      new_word = lem.lemmatize(word)
      #print('lem:: '+new_word)
      lemmatized_words += ' ' + new_word
  
  lemmatized_words = ' '.join(lemmatized_words.split()) #중복 공백 제거
  return lemmatized_words
  #return data

# Referrence:
# https://stackoverflow.com/questions/53801998/python-json-request-shows-a-keyerror-even-though-key-exists 
url = 'http://192.9.24.248:19002/query/service'
#url = 'http://sclab.gachon.ac.kr:19002/query/service'
def get_vec300(word):
  statement = 'use conceptnet; '
  statement += 'select vec300 from numberbatch_en where word="'+word+'";'

  req = requests.post(url, data = { 'statement': statement })
  try: 
    if req.status_code == 400: return []
    result = req.json()['results']
    if len(result) >= 1:
      vec300 = result[0]['vec300'].split(' ')
      return vec300
    else:
      print('word: '+word)
      return []
  except KeyError:
    print('keyerror')
    print(req)

    print(req.status_code)
    for i in range(0,10):
      req = requests.post(url, data = { 'statement': statement })
      if 'success' in req and req['success'] == 0:
        req = requests.post(url, data = { 'statement': statement })
      else:
        result = req.json()['results']
        print('\tsolved')
        if len(result) >= 1:
          vec300 = result[0]['vec300'].split(' ')
          return vec300
        else:
            print('word: '+word)
            return []
    print('\tkey error again!')

def make_temp_input(max_row, max_col=300):
  row = [0] * max_col
  temp_input = []
  for i in range(0,max_row):
      temp_input.append(row)
  return temp_input

def get_x_input(X, max_col=300,  max_token_length=33):
  empty_row = [0.0] * max_col
  x_input = []

  for x1 in X:
    temp = []
    #print(x1)
    # fill vectors in the temp
    for i, token in enumerate(x1):
      vec300 = get_vec300(token)
      if vec300 == []:
        # delete that token in token list(x1)
        x1.remove(token)
        continue
      else:
        temp.append(vec300)
    # check whether the temp's length is max_token_length or not
    if len(temp) < max_token_length:
        for i in range(len(temp), max_token_length):
            temp.append(empty_row)
    #print(len(temp))
    x_input.append(np.array(temp))
  return np.array(x_input)

def get_y_input(Y):
  y_input = []

  for y1 in Y:
    vec300 = get_vec300(y1)
    if vec300 == []:
      # delete that data in Y list
      Y.remove(y1)
      continue
    else:
      y_input.append(vec300)
  return np.array(y_input)




# cnn model
class CNN(object):
  """
  A CNN for word embedding recommendation
  Uses an embedding layer, followed by a convolutional, max-pooling layer
  pre-embedding vector: ConceptNet NumberBatch (English)

  <Parameters>
    - embedding_size: 각 단어에 해당되는 embedded vectors의 차원 (= 300)
    - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
    - num_filters: 각 filter size 별 filter 수
    - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
  """

  def __init__(self, train_size, embedding_size, channel_size,
          filter_sizes, num_filters, threshold, max_token_length, batch_size):
    # Placeholders for input(x) and output(y)
    # x 차원수: [batch_size * max_token_length(None: 가변길이) * embedding_size]
    self.input_x = tf.placeholder(tf.float32, [None, None, embedding_size], name='input_x')
    # x 차원수: [ batch_size(None) * None (token_length) * embedding_size * 1 (channel_size) ]
    self.input_y = tf.placeholder(tf.float32, [None, embedding_size], name='input_y')
    self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    self.threshold = threshold
    self.max_token_length = max_token_length
    self.batch_size = batch_size

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    with tf.name_scope('embedding'):
      self.expand_input_x = tf.expand_dims(self.input_x, -1) # channel 차원 추가
    
    """
    Create a convolution + max-pool layer for each filter size
    """
    pool_output = []
    for filter_size in filter_sizes:
      with tf.name_scope('conv-maxpool-%s'%filter_size):  
        # filter shape: bigram_filter, trigram_filter
        filter_shape = [filter_size, embedding_size, channel_size, num_filters]
      
        # weight and bias for conv layer1
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

        # convolutional layer
        conv = tf.nn.conv2d(
            #self.input_x, # conv input
            self.expand_input_x, # conv input
            W, 
            strides=[1,1,1,1], 
            padding='VALID', 
            name='conv')
        print('conv shape: ', end='')
        print(conv.shape)
        # add bias and apply activation function (ReLU) in conv layer
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu') # nonlinearity
        print('h shape: ', end='')
        print(h.shape)
        # max-pooling
        # ksize = [1,2,2,1] : 각 batch data & channel 별로 가로, 세로 2칸씩 움직이면서 max-pooling하라는 의미
        k_height = self.max_token_length - filter_size + 1
      
        #k_height = len(self.input_x) - filter_size + 1
        pool = tf.nn.max_pool(
            h, # pooling input
            ksize=[1, k_height, 1, 1], #ksize=[batch_size, height, width, channel]
            strides=[1,1,1,1], 
            padding='VALID', 
            name='pool')
        print('pool shape: ', end='')
        print(pool.shape)

        pool_output.append(pool)
      """
      # combine each vectors in same position of each frame (frame: num_filters)
      pool3d = tf.nn.max_pool3d(
          pool, # [batch_size, frame, width, height, in_channels]
          ksize=[num_filters, 1, 300, 1, 1], # [frame, width, height, in_channels, out_channels]
          strides=[1,1,1,1,1],
          padding='VALID',
          name='pool3d')
          
    pool_output.append(pool)
    """
    # //End of for

    # Combine all the polled features
    num_filters_total = num_filters * len(filter_sizes)
    self.h_pool = tf.concat(axis=3,values=pool_output)
    #self.h_pool = tf.concat(3,[pool_output[0], pool_output[1], pool_output[2]])
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope('dropout'):
      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    # Final (unnormalized) scores and predictions
    with tf.name_scope('output'):
      W = tf.get_variable(
              'W', 
              shape=[num_filters_total, embedding_size],
              initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[embedding_size]), name='b')
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)
      self.vectors = tf.nn.xw_plus_b(self.h_drop, W, b, name='vectors')
      print('output(vectors): ', end='')
      print(self.vectors.shape)
      self.vectors = tf.math.tanh(self.vectors)
    """
    with tf.name_scope('output'):
      # combine each vectors in same position of each frame (frame: len(filter_size) )
      pool3d_output = tf.nn.max_pool3d(
          pool_output,
          ksize=[len(filter_sizes), 1, 300, 1, 1],
          strides=[1,1,1,1,1],
          padding='VALID',
          name='pool3d_output'
          )
      # add dropout
      self.predictions = tf.nn.dropout(pool3d_output, dropout_prob, name='predictions') # 차원수: [1 * 300]
    """ 
    # MSE loss
    with tf.name_scope('loss'):
     # self.loss = tf.reduce_mean(tf.square(self.input_y - self.vectors)/embedding_size) / self.batch_size
      #self.loss = tf.reduce_mean(tf.square(self.input_y - self.vectors)) / embedding_size
      self.loss = tf.reduce_mean(tf.square(self.vectors - self.input_y))


    # calculate accuracy through cosine similarity
    # If cos sim > threshold; correct, Otherwise, incorrect
    with tf.name_scope('accuracy'):
      cos_similarity = tf.reduce_sum(
              tf.multiply(
                  self.vectors/tf.norm(self.vectors, axis=1, keep_dims=True),
                  self.input_y/tf.norm(self.input_y, axis=1, keep_dims=True)
                  ), axis=1)
      correct_predictions = tf.math.greater_equal(cos_similarity, self.threshold)
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
      




# Load and Read data from a csv file
# When using 'basic_english_word280.csv' file
"""
total_word = 280 # 280 단어
df = pd.read_csv('basic_english_word280.csv', delimiter=',', 
        encoding='utf-8', usecols=['word','definition'], index_col=None)
"""
# When using 'english_words_1009.csv' file
total_word = 1009 # 1009 단어
df = pd.read_csv('english_words_1009.csv', delimiter=',', 
        encoding='utf-8', usecols=['word','definition'], index_col=None)

print(len(df))

# Shuffle all data
df = df.sample(frac=1).reset_index(drop=True)

train_prob = 0.8
X, Y = df['definition'], df['word']

train_size = round(len(X))*train_prob
test_size = len(X) - train_size

# Split data into training and testing
x_train, x_test, y_train, y_test = [], [], [], []

max_token_length = 0
for i, x1 in enumerate(X):
  x1 = preprocessing(x1)
  tokens = x1.split(' ')
  if max_token_length < len(tokens): max_token_length = len(tokens)
  if i < train_size:
    x_train.append(tokens)
  else:
    x_test.append(tokens)

print('\nmax token length: '+str(max_token_length)+'\n')

#y_train, y_test = Y[:train_size], Y[train_size:]
for i, y1 in enumerate(Y):
    y1 = preprocessing(y1)
    if i < train_size:
        y_train.append(y1)
    else:
        y_test.append(y1)

# Make model input data
# x_train_input 차원수: [train_size * max_token_length(None: 가변길이) * embedding_size]
# x_test_input 차원수: [test_size * max_token_length(None: 가변길이) * embedding_size]
# y_train_input 차원수: [train_size * embedding_size]
# y_test_input 차원수: [test_size * embedding_size]
x_train_input, x_test_input = get_x_input(x_train, 300, max_token_length), get_x_input(x_test, 300, max_token_length)
y_train_input, y_test_input = get_y_input(y_train), get_y_input(y_test)

print("\nx_train_input dims:", end='')
print(x_train_input.shape)

print("x_test_input dims:", end='')
print(x_test_input.shape)

print("y_train_input dims:", end='')
print(y_train_input.shape)
print("y_test_input dims:", end='')
print(y_test_input.shape)



batch_size = 30
dropout_keep_prob = 1


# train the model and test
with tf.Graph().as_default():
  sess = tf.Session()
  with sess.as_default():
    model = CNN(train_size=train_size,
                embedding_size=300, # embedding vector dims: 300 (vec300)
                channel_size=1,
                #dropout_prob=0.8,
                filter_sizes=[2,3,4], # filter size=1: unigram, 2: bigram, 3: trigram, ...
                num_filters=128,
                threshold=0.60, # cosine similarity threshold (accuracy)
                max_token_length=max_token_length,
                batch_size=batch_size)
    
    # Define training procedure
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
      if g is not None:
        grad_hist_summary = tf.summary.histogram("{}".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev Summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
      """
      A single training step
      """
      feed_dict = {
          model.input_x: x_batch,
          model.input_y: y_batch,
          model.dropout_keep_prob: dropout_keep_prob 
      }
      _, step, summaries, loss, accuracy = sess.run(
          [train_op, global_step, train_summary_op, model.loss, model.accuracy],
          feed_dict
      )
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)
    

    def dev_step(x_batch, y_batch, writer=None):
      """
      Evaluates model on a dev set
      """
      feed_dict = {
          model.input_x: x_batch,
          model.input_y: y_batch,
          model.dropout_keep_prob: 1.0
      }
      step, summaries, loss, accuracy = sess.run(
          [global_step, dev_summary_op, model.loss, model.accuracy],
          feed_dict
      )
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      if writer:
        writer.add_summary(summaries, step)
    

    def batch_iter(x_data, y_data, batch_size, num_epochs):
      """
      Generates a batch iterator for a dataset.
      """
      x_data = np.array(x_data)
      y_data = np.array(y_data)
      data_size = len(x_data)
      num_batches_per_epoch = int((len(x_data) - 1) / batch_size) + 1
      for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
          start_index = batch_num * batch_size
          end_index = min((batch_num + 1) * batch_size, data_size)
          yield (x_data[start_index:end_index], y_data[start_index:end_index])
    
    # Generate batches
    """
    (x_batches, y_batches) = batch_iter(
        x_data = x_train_input,
        y_data = y_train_input, 
        batch_size = 30, 
        num_epochs=10
    )
    """

    testpoint = 0
    aPoint = 25
    evaluate_every, checkpoint_every = 25, 25
    # Training loop. For each batch...
    #for x_batch, y_batch in zip(x_batches, y_batches):
    for (x_batch, y_batch) in batch_iter(x_train_input, y_train_input, batch_size, 20):
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            if testpoint + aPoint < len(x_test_input):
                testpoint += aPoint
            else:
                testpoint = 0
            print("\nEvaluation:")
            dev_step(x_test_input[testpoint:testpoint + aPoint], y_test_input[testpoint:testpoint + aPoint],
                     writer=dev_summary_writer)
            print("")
        if current_step % checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))