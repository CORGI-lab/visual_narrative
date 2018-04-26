import time
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import json
import WordVocabBuilder

StoredImageFeaturesFile = 'img_feats_train_0_1_2_3_10_12_fc1'
ImageDescriptionMetaDataFile = 'train.description-in-isolation.json'

class SequentialImageDescription:
    def __init__(self, feature_dimension, sentRNN_lstm_dim, sentRNN_FC_dim, wordRNN_lstm_dim, 
                 word_embed_dim, project_dim, N_max_word, n_words, batch_size):
        self.feature_dimension = feature_dimension
        self.wordRNN_lstm_dim = wordRNN_lstm_dim
        self.word_embed_dim = word_embed_dim
        self.project_dim = project_dim
        self.sentRNN_lstm_dim = sentRNN_lstm_dim
        self.sentRNN_FC_dim = sentRNN_FC_dim
        self.N_max_word = N_max_word
        self.n_words = n_words
        self.batch_size = batch_size
        self.regionPooling_W = tf.Variable(tf.random_uniform([feature_dimension, project_dim], -0.1, 0.1))
        self.regionPooling_b = tf.Variable(tf.zeros([project_dim]))
        
        self.fc1_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, sentRNN_FC_dim], -0.1, 0.1))
        self.fc1_b = tf.Variable(tf.zeros(sentRNN_FC_dim))
        self.fc2_W = tf.Variable(tf.random_uniform([sentRNN_FC_dim, 1024], -0.1, 0.1))
        self.fc2_b = tf.Variable(tf.zeros(1024))
        
        self.embed_word_W = tf.Variable(tf.random_uniform([wordRNN_lstm_dim, n_words], -0.1,0.1))
        self.embed_word_b = tf.Variable(tf.zeros([n_words]))
        
        self.W_embeddings = tf.Variable(tf.random_uniform([n_words, word_embed_dim], -0.1, 0.1))
        
        self.sent_LSTM = tf.nn.rnn_cell.BasicLSTMCell(sentRNN_lstm_dim, state_is_tuple=True)
        self.word_LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(wordRNN_lstm_dim, state_is_tuple=True)
        self.word_LSTM_cell = tf.nn.rnn_cell.MultiRNNCell([self.word_LSTM_cell] * 2, state_is_tuple=True)
    
    # Follow from https://github.com/chenxinpeng/im2p
    def build_model(self):
        loss_word = 0.0
        feats = tf.placeholder(tf.float32, [self.batch_size, self.feature_dimension])
        caption = tf.placeholder(tf.int32, [self.batch_size, self.N_max_word + 1])
        # topic RNN
        project_vec_all = tf.matmul(feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [self.batch_size, 1, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)
        
        sent_state = self.sent_LSTM.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        
        sent_output, sent_state = self.sent_LSTM(project_vec, sent_state)
        hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b)
        sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b)
        
        state = tf.nn.rnn_cell.LSTMStateTuple(sent_topic_vec[:, 0:512], sent_topic_vec[:, 512:])
        word_state = (state, state)
        
        for j in range(0, self.N_max_word):
            if j > 0:
                tf.get_variable_scope().reuse_variables()
                    
            current_embed = tf.nn.embedding_lookup(self.W_embeddings, caption[:, j])
            word_output, word_state = self.word_LSTM_cell(current_embed, word_state)
    
            indices = caption[:,j]
            onehot_labels = tf.one_hot(indices, depth = self.n_words)
            
            logit_words = tf.nn.xw_plus_b(word_output[:], self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels)
            
            prediction = tf.nn.softmax(logit_words)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(onehot_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            loss_wordRNN = tf.reduce_sum(cross_entropy) / self.batch_size
            loss_word += loss_wordRNN
    
        return feats, caption, loss_word, accuracy

def loadImageFeatures():
    image_features_list = pickle.load(open(StoredImageFeaturesFile, 'rb'))
    image_features = dict()
    for j in range(len(image_features_list)):
        image_id = image_features_list[j]['id']
        features = image_features_list[j]['features']
        image_features[image_id] = features
    
    return image_features
    
def loadAnnotations(image_features):
    description_metadata = json.load(open(ImageDescriptionMetaDataFile))
    annotations = description_metadata['annotations']
    description = dict()
    annotations_len = len(annotations)
    
    for i in range(annotations_len):
        for j in range(len(annotations[i])):
            photo_id = annotations[i][j]['photo_flickr_id']
            if photo_id in image_features.keys():
                image_des = annotations[i][j]['original_text']
                image_des = image_des.lower()
                image_des = image_des.strip()
                image_des = image_des.replace(',', ' ,')
                image_des = image_des.replace('.', '')
                image_des = image_des.replace('"', ' " ')
                description[photo_id] = image_des
    
    return description
                    
def buildVocabDictionary(annotationsList):
    all_sentences = []
    for key, value in annotationsList.items():
        sent = value
        sent = sent.lower()
        sent = sent.strip()
        sent = sent.replace(',', ' ,')
        sent = sent.replace('.', '')
        sent = sent.replace('"', ' " ')
        all_sentences.append(sent)
            
    word2idx, idx2word = WordVocabBuilder.preProBuildWordVocab(all_sentences, 2)
    return word2idx, idx2word
    
def main():
    
    image_features = loadImageFeatures()
    annotations = loadAnnotations(image_features)
    word2idx, idx2word = buildVocabDictionary(annotations)
if __name__ == "__main__":
    main()
        
