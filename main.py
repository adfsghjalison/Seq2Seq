import tensorflow as tf
import numpy as np
import json
import re, os, sys, csv, math
sys.path.append('sentiment_analysis/')
from termcolor import colored
from flags import FLAGS, SEED, buckets, replace_words, reset_prob 
from utils import qulify_sentence

import data_utils
import seq2seq_model
from seq2seq import bernoulli_sampling
from sentiment_analysis import main
from sentiment_analysis import utils

# mode variable has three different mode:
# 1. MLE
# 2. RL
# 3. TEST
def create_seq2seq(session, mode):

  if FLAGS.mode == 'TEST' or 'val' in FLAGS.mode:
    FLAGS.schedule_sampling = False 
  else:
    FLAGS.beam_search = False
  print('FLAGS.beam_search: ',FLAGS.beam_search)
  if FLAGS.beam_search:
    print('FLAGS.beam_size: ',FLAGS.beam_size)
    print('FLAGS.debug: ',bool(FLAGS.debug))
      
  model = seq2seq_model.Seq2seq(mode)

  if FLAGS.mode == 'val_rl':
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_rl_dir)
  else:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_pre_dir)
  
  if FLAGS.load != '':
    print("Reading model from %s, mode: %s" % (FLAGS.load, FLAGS.mode))
    model.saver.restore(session, FLAGS.load)
  elif ckpt:
    print("Reading model from %s, mode: %s" % (ckpt.model_checkpoint_path, FLAGS.mode))
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters, mode: %s" % FLAGS.mode)
    session.run(tf.global_variables_initializer())
  
  return model

def _output(output, trg_vocab_list):
  outputs = [int(np.argmax(logit)) for logit in output]
  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
  if outputs == []:
    outputs = ['.']
  sys_reply = "".join([tf.compat.as_str(trg_vocab_list[output]) for output in outputs])
  return sys_reply

def train_MLE(): 

  data_utils.prepare_whole_data(FLAGS.data, FLAGS.data_test, FLAGS.source_data, FLAGS.target_data, FLAGS.src_vocab_size, FLAGS.trg_vocab_size)
  _ , trg_vocab_list = data_utils.read_map(FLAGS.target_data + '.' + str(FLAGS.trg_vocab_size) + '.mapping')

  d_train = data_utils.read_data(FLAGS.source_data + '_train.token',FLAGS.target_data + '_train.token',buckets)
  d_valid = data_utils.read_data(FLAGS.source_data + '_val.token',FLAGS.target_data + '_val.token',buckets)
  
  print('Total document size of training data: %s' % sum(len(l) for l in d_train))
  print('Total document size of validation data: %s' % sum(len(l) for l in d_valid))

  train_bucket_sizes = [len(d_train[b]) for b in range(len(d_train))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]
  print('train_bucket_sizes: ',train_bucket_sizes)
  print('train_total_size: ',train_total_size)
  print('train_buckets_scale: ',train_buckets_scale)
  valid_bucket_sizes = [len(d_valid[b]) for b in range(len(d_valid))]
  valid_total_size = float(sum(valid_bucket_sizes))
  valid_buckets_scale = [sum(valid_bucket_sizes[:i + 1]) / valid_total_size
                         for i in range(len(valid_bucket_sizes))]
  print('valid_bucket_sizes: ',valid_bucket_sizes)
  print('valid_total_size: ',valid_total_size)
  print('valid_buckets_scale: ',valid_buckets_scale)

  with tf.Session() as sess:

    model = create_seq2seq(sess, 'MLE')
    if FLAGS.reset_sampling_prob: 
      with tf.variable_scope('sampling_prob',reuse=tf.AUTO_REUSE):
        sess.run(tf.assign(model.sampling_probability,reset_prob))
    if FLAGS.schedule_sampling:
      print('model.sampling_probability: ',model.sampling_probability_clip)
    #sess.run(tf.assign(model.sampling_probability,1.0))
    step = 0
    loss = 0
    loss_list = []
 
    if FLAGS.schedule_sampling:
      print('sampling_decay_steps: ',FLAGS.sampling_decay_steps)
      print('sampling_probability: ',sess.run(model.sampling_probability_clip))
      print('-----')

    while step < FLAGS.max_step:
      step += 1

      random_number = np.random.random_sample()
      # buckets_scale accumulated percentage
      bucket_id = min([i for i in range(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number])
      encoder_input, decoder_input, weight, en_s, de_s = model.get_batch(d_train, bucket_id, sen=True)
      #print('batch_size: ',model.batch_size)      ==> 64
      #print('batch_size: ',len(encoder_input[0])) ==> 64
      #print('batch_size: ',len(encoder_input))    ==> 15,50,...
      #print('batch_size: ',len(decoder_input))    ==> 15,50,... 
      #print('batch_size: ',len(weight))           ==> 15,50,...
      output, loss_train, _ = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
      loss += loss_train / FLAGS.check_step

      #if step!=0 and step % FLAGS.sampling_decay_steps == 0:
      #  sess.run(model.sampling_probability_decay)
      #  print('sampling_probability: ',sess.run(model.sampling_probability))
        
      if step % FLAGS.print_step == 0:
        print('Input :')
        print(en_s[0].strip())
        print('Output:')
        print(_output(output[0], trg_vocab_list))
        print('\n{} steps trained ...\n\n'.format(step))

      if step % FLAGS.check_step == 0:
        print('\nStep %s, Training perplexity: %s, Learning rate: %s' % (step, math.exp(loss),
                                  sess.run(model.learning_rate))) 
        for i in range(len(d_train)):
          encoder_input, decoder_input, weight = model.get_batch(d_valid, i)
          _, loss_valid = model.run(sess, encoder_input, decoder_input, weight, i, forward_only = True)
          print('  Validation perplexity in bucket %s: %s' % (i, math.exp(loss_valid)))
        if len(loss_list) > 2 and loss > max(loss_list[-3:]):
          sess.run(model.learning_rate_decay)
        else:
          if step!=0:
            if FLAGS.schedule_sampling:
              sess.run(model.sampling_probability_decay)
              print('sampling_probability: ',sess.run(model.sampling_probability_clip))
        loss_list.append(loss)  
        loss = 0

        checkpoint_path = os.path.join(FLAGS.model_pre_dir, "MLE.ckpt")
        model.saver.save(sess, checkpoint_path, global_step = step)
        print('Saving model at step %s\n' % step)
      if step == FLAGS.sampling_global_step: break

def val(mo):

  d_valid = data_utils.read_val_data(FLAGS.source_data + '_val.token',FLAGS.target_data + '_val.token',buckets)

  _ , trg_vocab_list = data_utils.read_map(FLAGS.target_data + '.' + str(FLAGS.trg_vocab_size) + '.mapping')
  
  print('Total document size of validation data: %s' % sum(len(l) for l in d_valid))

  with tf.Session() as sess:
    
    model = create_seq2seq(sess, 'TEST')
    loss_list = []
    
    cf = csv.writer(open(FLAGS.output, 'w'), delimiter = '|')
    cf.writerow(['context', 'utterance'])    

    for i in range(len(d_valid)):
      encoder_input, decoder_input, weight, en_s, de_s = model.get_one(d_valid, i, sen=True)
      output = model.run(sess, encoder_input, decoder_input, weight, d_valid[i][0])
      cf.writerow([''.join(en_s.strip().split()), _output(output[0], trg_vocab_list)])
      if i % 1000 == 0:
        print('Generate {} ...'.format(i))

def train_RL():

  data_utils.prepare_whole_data(FLAGS.data, FLAGS.data_test, FLAGS.source_data, FLAGS.target_data, FLAGS.src_vocab_size, FLAGS.trg_vocab_size)
  d_train = data_utils.read_data(FLAGS.source_data + '_train.token',FLAGS.target_data + '_train.token',buckets)
  #print(d_train[0][0])

  g1 = tf.Graph()
  g2 = tf.Graph()
  g3 = tf.Graph()
  sess1 = tf.Session(graph = g1)
  sess2 = tf.Session(graph = g2)
  sess3 = tf.Session(graph = g3)
  # model is for training seq2seq with Reinforcement Learning
  with g1.as_default():
    model = create_seq2seq(sess1, 'RL')
    # we set sample size = ?
    model.batch_size = 5
  # model_LM is for a reward function (language model)
  with g2.as_default():
    model_LM = create_seq2seq(sess2, 'MLE')
    model_LM.beam_search = False
    # calculate probibility of only one sentence
    model_LM.batch_size = 1

  def LM(encoder_input, decoder_input, weight, bucket_id):
    return model_LM.run(sess2, encoder_input, decoder_input, weight, bucket_id, forward_only = True)[0]
  # new reward function: sentiment score
  with g3.as_default():
    model_SA = main.create_model(sess3) 
    model_SA.batch_size = 1
 
  def SA(sentence, encoder_length):
    sentence = ' '.join(sentence)
    token_ids = utils.convert_to_token(sentence, model_SA.vocab_map)
    encoder_input, encoder_length, _, _ = model_SA.get_batch([(0, token_ids, sentence)])
    return model_SA.step(sess3, encoder_input, encoder_length)[0][0]


  train_bucket_sizes = [len(d_train[b]) for b in range(len(d_train))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

  # make RL object read vocab mapping dict, list  
  model.RL_readmap(FLAGS.source_data + '.' + str(FLAGS.src_vocab_size) + '.mapping', FLAGS.target_data + '.' + str(FLAGS.trg_vocab_size) + '.mapping')
  step = 0
  

  while step < FLAGS.max_step:
    step += 1

    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number])
    
    # the same encoder_input for sampling batch_size times
    #encoder_input, decoder_input, weight = model.get_batch(d, bucket_id, rand = False)    

    encoder_input, decoder_input, weight, en_s, de_s = model.get_batch(d_train, bucket_id, sen=True)
    output, loss, _ = model.run(sess1, encoder_input, decoder_input, weight, bucket_id, X = LM, Y = SA)
   
    # debug 
    #encoder_input = np.reshape(np.transpose(encoder_input, (1, 0, 2)), (-1, FLAGS.vocab_size))
    #encoder_input = np.split(encoder_input, FLAGS.max_length)

    #print(model.token2word(encoder_input)[0])
    #print(model.token2word(sen)[0])
    
    if step % FLAGS.print_step == 0:
      print('Input :')
      print(en_s[0].strip())
      print('Output:')
      print(_output(output[0], model.trg_vocab_list))
      print('\n{} steps trained ...'.format(step))

    if step % FLAGS.check_step == 0:
      print('Loss at step %s: %s' % (step, loss))
      checkpoint_path = os.path.join(FLAGS.model_rl_dir, "RL.ckpt")
      model.saver.save(sess1, checkpoint_path, global_step = step)
      print('Saving model at step %s' % step)


def test():
  if FLAGS.src_word_seg == 'word':
    import jieba
    jieba.initialize()
  sess = tf.Session()
  src_vocab_dict, _ = data_utils.read_map(FLAGS.source_data + '.' + str(FLAGS.src_vocab_size) + '.mapping')
  _ , trg_vocab_list = data_utils.read_map(FLAGS.target_data + '.' + str(FLAGS.trg_vocab_size) + '.mapping')
  model = create_seq2seq(sess, 'TEST')
  model.batch_size = 1
  
  sys.stdout.write("Input sentence: ")
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  if FLAGS.src_word_seg == 'word':
    sentence = (' ').join(jieba.lcut(sentence))
    print('sentence: ',sentence)
  elif FLAGS.src_word_seg == 'char':
    sentence = (' ').join([s for s in sentence])
  while(sentence):
    token_ids = data_utils.convert_to_token(tf.compat.as_bytes(sentence), src_vocab_dict, False)
    bucket_id = len(buckets) - 1
    for i, bucket in enumerate(buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    # Get a 1-element batch to feed the sentence to the model.
    encoder_input, decoder_input, weight = model.get_batch({bucket_id: [(token_ids, [], "", "")]}, bucket_id)
    # Get output logits for the sentence.
    output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    
    # beam search all
    if bool(model.beam_search) is True:
        if bool(FLAGS.debug):
            outs = []
            for _ in range(model.beam_size):
                outs.append([])
   
            for out in output:
                for i,o in enumerate(out):
                    outs[i].append(o)
            outs = np.array(outs)
            #print('outs: ',outs.shape)
            outputss = []
            for out in outs:
                #print('out: ',out.shape)
                outputs = [int(np.argmax(logit)) for logit in out]
                outputss.append(outputs)
    
            for i,outputs in enumerate(outputss):
                sys_reply = "".join([tf.compat.as_str(trg_vocab_list[output]) for output in outputs])
                sys_reply = data_utils.sub_words(sys_reply)
                sys_reply = qulify_sentence(sys_reply)
                if i == 0:
                    print(colored("Syetem reply(bs best): " + sys_reply,"red"))
                else:
                    print("Syetem reply(bs all): " + sys_reply)
        else:
            output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output]
            if data_utils.EOS_ID in outputs:
              outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            sys_reply = "".join([tf.compat.as_str(trg_vocab_list[output]) for output in outputs])
            sys_reply = data_utils.sub_words(sys_reply)
            sys_reply = qulify_sentence(sys_reply)
            print("Syetem reply(bs best): " + sys_reply)
            

    # MLE
    else:
        output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
        print(output)
        print('output: ', len(output), output.shape, output[0].shape)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        sys_reply = "".join([tf.compat.as_str(trg_vocab_list[output]) for output in outputs])
        sys_reply = data_utils.sub_words(sys_reply)
        sys_reply = qulify_sentence(sys_reply)
        print("Syetem reply(MLE): " + sys_reply)


    # Print out French sentence corresponding to outputs.
    #print("Syetem reply: " + "".join([tf.compat.as_str(trg_vocab_list[output]) for output in outputs]))
    print ("User input  : ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    if FLAGS.src_word_seg == 'word':
      sentence = (' ').join(jieba.lcut(sentence))
      print ('sentence: ', sentence)
    elif FLAGS.src_word_seg == 'char':
      sentence = (' ').join([s for s in sentence])

if __name__ == '__main__':
  if FLAGS.mode == 'MLE':
    train_MLE()
  elif FLAGS.mode == 'RL':
    train_RL()
  elif 'val' in FLAGS.mode:
    val(FLAGS.mode)
  else:
    test()

