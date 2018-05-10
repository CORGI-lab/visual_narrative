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
N_max_word = 30
n_epochs = 10
batch_size = 2
model_path = 'models'

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

def get_caption_matrix(img_sent):
    img_captions_matrix = np.ones([ N_max_word+1], dtype=np.int32) * 2
    for idx, word in enumerate(img_sent.lower().split(' ')):
        # because the biggest number of words in a sentence is N_max, here is 50
        if idx == N_max_word:
            break
            
        if word in word2idx:
            img_captions_matrix[ idx] = word2idx[word]
        else:
            img_captions_matrix[ idx] = word2idx['<unk>']
    
    return img_captions_matrix


image_features = loadImageFeatures()
annotations = loadAnnotations(image_features)
word2idx, idx2word = buildVocabDictionary(annotations)   

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
        self.beam_width = 5
    
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

    def generate_model(self):
    
        feats = tf.placeholder(tf.float32, [1, self.feature_dimension])
            
        project_vec_all = tf.matmul(feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [1, 1, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)
        
        sent_state = self.sent_LSTM.zero_state(batch_size=1, dtype=tf.float32)
        
        sent_output, sent_state = self.sent_LSTM(project_vec, sent_state)
        hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b)
        sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b)
        
    
        generated_sent = []
        k_top_generated_sent = []
        prob_top_generated_sent = []
    
        state = tf.nn.rnn_cell.LSTMStateTuple(sent_topic_vec[:, 0:512], sent_topic_vec[:, 512:])
        word_state = (state, state)
        
        # beam search initial step
        with tf.device('/cpu:0'):
            current_embed = tf.nn.embedding_lookup(self.W_embeddings, tf.zeros([1], dtype=tf.int64))
            
        word_output, init_word_state = self.word_LSTM_cell(current_embed, word_state)
        logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)
        
        
        prediction = tf.nn.softmax(logit_words)
        flatten_prediction = tf.reshape(prediction, [-1])
        
        top_k_index = tf.nn.top_k(flatten_prediction, self.beam_width)
        
        k_sequences = []
        for i in range(self.beam_width):
            candidate = [[top_k_index[1][i]], tf.log(top_k_index[0][i])]
            k_sequences.append(candidate)
        
        probability_dist = []
        softmax_probability_dist = []
        # beam search next steps
        for j in range(1, N_max_word):
            tf.get_variable_scope().reuse_variables()
            
            temp_sequences = []
            candidate_words_list = []
            candidate_prob_list = []
            for sequence in k_sequences:
                prior_words = sequence[0]
                prior_probability = sequence[1]
                k = 0
    
                for word_index in prior_words:
                    
                    current_embed = tf.nn.embedding_lookup(self.W_embeddings, word_index)
                    current_embed = tf.expand_dims(current_embed, 0)
                    if k == 0:
                        word_output, word_state = self.word_LSTM_cell(current_embed, init_word_state)
                        k = 1
                    else:
                        word_output, word_state = self.word_LSTM_cell(current_embed, word_state)
                
                logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)
                
                prediction = tf.nn.softmax(logit_words)
                flatten_prediction = tf.reshape(prediction, [-1])
                
                top_k_index = tf.nn.top_k(flatten_prediction, self.beam_width)
                
                
                probability_dist.append(logit_words)
                softmax_probability_dist.append(prediction)
                for i in range(self.beam_width):
                    candidate_words = []
                    for word_index in prior_words:
                        candidate_words.append(word_index)
                    
                    candidate_words.append(top_k_index[1][i])
                    candidate_prob = tf.add(prior_probability, tf.log(top_k_index[0][i]))
                    
                    candidate_words_list.append(candidate_words)
                    candidate_prob_list.append(candidate_prob)
                    
            
            
            top_k_prob_index = tf.nn.top_k(candidate_prob_list, self.beam_width)
            sorted_sequences = []
            
            for i in range(self.beam_width):
                word_list = tf.gather(candidate_words_list, top_k_prob_index[1][i])
                prob = tf.gather(candidate_prob_list, top_k_prob_index[1][i])
                word_list = tf.unstack(word_list)
                sorted_sequences.append([word_list, prob])
            
            k_sequences = sorted_sequences
    
        return feats, k_sequences, probability_dist, softmax_probability_dist

def train_model():
    learning_rate = 0.001
    # format the training data set
    image_features_len = len(image_features)
    x_train = []
    y_train = []
    image_ids = []
    for j in range(image_features_len):
        photo_id = image_features[j]['id']
        features = image_features[j]['features']
        
        if photo_id in annotations:
            caption = annotations[photo_id]
            caption_matrix = get_caption_matrix(caption)
            x_train.append(features)
            y_train.append(caption_matrix)
            image_ids.append(photo_id)
            
    seqImageDescModel = SequentialImageDescription()
    tf_feats, tf_caption, tf_loss, tf_acc = seqImageDescModel.build_model()
    sess = tf.InteractiveSession()
    
    #pretrained_model_path = './models_batch/model_partial-125'
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    #saver = tf.train.import_meta_graph('./models_batch/model_partial-125.meta')

    #saver.restore(sess, pretrained_model_path)
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()
    training_data_length = len(x_train)
    for epoch in range(0, n_epochs):
            loss = 0
            total_pass = 0
            start_time = time.time()
            for i in range(0, training_data_length, batch_size):
                start = i
                end = i + batch_size
                if training_data_length > end:
                    x = x_train[start:end]
                    y = y_train[start:end]
                    
                    features = np.asarray(np.reshape(x, (batch_size, -1)))
                    caption_matrix = np.asarray(np.reshape(y, (batch_size, -1)))
                    

                    _, loss_word, acc_word = sess.run(
                                    [train_op, tf_loss, tf_acc],
                                    feed_dict={
                                               tf_feats: features,
                                               tf_caption: caption_matrix
                                    })
                    loss = loss + loss_word
                    total_pass = total_pass + 1
            
            loss = loss/total_pass
            print ('Epoch: ', epoch, ' loss: ', loss, ' Time cost: ', str((time.time() - start_time)))
            if np.mod(epoch, 50) == 0:
                print ("Epoch ", epoch, " is done. Saving the model ...")
                saver.save(sess, os.path.join(model_path, 'model_ckpt'), global_step=epoch)


 
def main():
    train_model()
    
if __name__ == "__main__":
    main()
        
