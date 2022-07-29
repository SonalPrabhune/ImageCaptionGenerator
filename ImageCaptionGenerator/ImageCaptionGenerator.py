import numpy as np
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

import string
import os
import glob
from PIL import Image
from time import time
import csv

from keras import Input, layers
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

#Step 2:- Data loading and Preprocessing
token_path = ".\Data\Text\captions.txt"
#train_images_path = ".\Data\Text\Flickr8k_train.token.txt"
#test_images_path = ".\Data\Text\Flickr8k_test.token.txt"
images_path = ".\Data\Images\\"
glove_path = ".\Glove\\"
model_path = ".\Model\\"
checkpoint_path = ".\Model\Checkpoint\\"
param_path = ".\Model\Parameters\\"

class ImageCaptionGenerator:
    model = InceptionV3(weights='imagenet')
    #Removing the softmax layer from the model since the purpose is to extract a vector and not image classification
    model_new = Model(model.input, model.layers[-2].output)

    trained_model = None
    vocab_size=0
    max_length = 0
    wordtoix = {}
    ixtoword = {}
    checkpoint_dir = os.path.dirname(checkpoint_path)

    def generateCaption(self, pic = '2398605966_1d0c9e6a20.jpg'):
        try:
            trained_model = load_model(model_path+"trained_model")                 
                        
            #Reading Vocab size
            vocab = []
            with open(param_path+'vocab_size.csv', newline='') as f:
                reader = csv.reader(f)
                vocab = list(reader)         
            str1 = ""
            self.vocab_size = int(str1.join(vocab[0]))

            #Reading Max length
            length = []
            with open(param_path+'max_length.csv', newline='') as f:
                reader = csv.reader(f)
                length = list(reader)
            str1 = ""
            self.max_length = int(str1.join(length[0]))

            word_to_index = pd.read_csv(param_path+"word_to_index.csv")
            self.wordtoix = word_to_index.to_dict('r')
            
            index_to_word = pd.read_csv(param_path+"index_to_word.csv")
            self.ixtoword = index_to_word.to_dict('r')
                        
            return self.test(self, pic, trained_model)
        except:
            print("No trained model saved! Training a new model, this will take a few hours...")
            doc = open(token_path,'r').read()
            print(doc[:410])

            descriptions = dict()
            for line in doc.split('\n'):
                tokens = line.split()
                if len(line) > 2:
                    image_id = tokens[0].split('.')[0]
                    image_desc = ' '.join(tokens[1:])
                    if image_id not in descriptions:
                        descriptions[image_id] = list()
                    descriptions[image_id].append(image_desc)


            table = str.maketrans('', '', string.punctuation)
            for key, desc_list in descriptions.items():
                for i in range(len(desc_list)):
                    desc = desc_list[i]
                    desc = desc.split()
                    desc = [word.lower() for word in desc]
                    desc = [w.translate(table) for w in desc]
                    desc_list[i] =  ' '.join(desc)

            #pic = '1000268201_693b08cb0e.jpg'
            #x=plt.imread(images_path+pic)
            #plt.imshow(x)
            #plt.show()
            #descriptions['1000268201_693b08cb0e']

            vocabulary = set()
            for key in descriptions.keys():
                    [vocabulary.update(d.split()) for d in descriptions[key]]
            print('Original Vocabulary Size: %d' % len(vocabulary))

            lines = list()
            for key, desc_list in descriptions.items():
                for desc in desc_list:
                    lines.append(key + ' ' + desc)
            new_descriptions = '\n'.join(lines)

    
            doc = open(token_path,'r').read()
            dataset = list()
            for line in doc.split('\n'):
                if len(line) > 1:
                  identifier = line.split('.')[0]
                  dataset.append(identifier)

            totalDataset = dataset[1:]
            totalImageCt = len(totalDataset)
            trainDataCt = int(totalImageCt*0.8)

            #doc = open(train_images_path,'r').read()
            #dataset = list()
            #for line in doc.split('\n'):
            #    if len(line) > 1:
            #      identifier = line.split('.')[0]
            #      dataset.append(identifier)

            train = set(dataset[1:trainDataCt])
            test = set(dataset[(trainDataCt+1):])

            img = glob.glob(images_path + '*.jpg')
            #train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
            train_images = [im + '.jpg' for im in train]
            train_img = []
            for i in img: 
                if i[len(images_path):] in train_images:
                    train_img.append(i)

            #test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
            test_images = [im + '.jpg' for im in test]
            test_img = []
            for i in img: 
                if i[len(images_path):] in test_images: 
                    test_img.append(i)

            train_descriptions = dict()
            for line in new_descriptions.split('\n'):
                tokens = line.split()
                image_id, image_desc = tokens[0], tokens[1:]
                if image_id in train:
                    if image_id not in train_descriptions:
                        train_descriptions[image_id] = list()
                    desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                    train_descriptions[image_id].append(desc)

            all_train_captions = []
            for key, val in train_descriptions.items():
                for cap in val:
                    all_train_captions.append(cap)

            word_count_threshold = 10
            word_counts = {}
            nsents = 0
            for sent in all_train_captions:
                nsents += 1
                for w in sent.split(' '):
                    word_counts[w] = word_counts.get(w, 0) + 1
            vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

            print('Vocabulary = %d' % (len(vocab)))
                    
            ix = 1
            for w in vocab:
                self.wordtoix[w] = ix
                self.ixtoword[ix] = w
                ix += 1

            word_to_index = pd.DataFrame(self.wordtoix, index=[0])
            word_to_index.to_csv(param_path+"word_to_index.csv")
            index_to_word = pd.DataFrame(self.ixtoword, index=[0])
            index_to_word.to_csv(param_path+"index_to_word.csv")

            self.vocab_size = len(self.ixtoword) + 1

            with open(param_path+'vocab_size.csv', 'w') as f:      
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerow(str(self.vocab_size))
                write.writerow(vocab)

            all_desc = list()
            for key in train_descriptions.keys():
                [all_desc.append(d) for d in train_descriptions[key]]
            lines = all_desc
            self.max_length = max(len(d.split()) for d in lines)

            print('Description Length: %d' % self.max_length)
            with open(param_path+'max_length.csv', 'w') as f:      
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerow(str(self.max_length))

            #Step 3:- Glove Embeddings

            embeddings_index = {} 
            f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

            embedding_dim = 200
            embedding_matrix = np.zeros((self.vocab_size, embedding_dim))
            for word, i in self.wordtoix.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector


            #Step 4:- Model Building and Training
            #model = InceptionV3(weights='imagenet')

            ##Removing the softmax layer from the model since the purpose is to extract a vector and not image classification
            #model_new = Model(model.input, model.layers[-2].output)

            encoding_train = {}
            for img in train_img:
                encoding_train[img[len(images_path):]] = self.encode(self, img)
            train_features = encoding_train

            encoding_test = {}
            for img in test_img:
                encoding_test[img[len(images_path):]] = self.encode(self, img)
            test_features = encoding_test

            inputs1 = Input(shape=(2048,))
            fe1 = Dropout(0.5)(inputs1)
            fe2 = Dense(256, activation='relu')(fe1)

            inputs2 = Input(shape=(self.max_length,))
            se1 = Embedding(self.vocab_size, embedding_dim, mask_zero=True)(inputs2)
            se2 = Dropout(0.5)(se1)
            se3 = LSTM(256)(se2)

            decoder1 = add([fe2, se3])
            decoder2 = Dense(256, activation='relu')(decoder1)
            outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

            trained_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
            trained_model.summary()

            try:
                latest = tf.train.latest_checkpoint(self.checkpoint_dir)
                print(latest)
                trained_model.load_weights(latest)
            except:           
                #Step 5:- Model Training
    
                trained_model.layers[2].set_weights([embedding_matrix])
                trained_model.layers[2].trainable = False

                trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                epochs = 30
                batch_size = 3
                steps = len(train_descriptions)//batch_size

                # Create a callback that saves the model's weights
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_freq=5*batch_size,
                                                     verbose=1)


                generator = self.data_generator(self, train_descriptions, train_features, self.wordtoix, self.max_length, batch_size)
                print(next(generator))
                trained_model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
                                 callbacks=[cp_callback])

                #Saving trained model to disk
                trained_model.save(model_path+"trained_model")

            #Step 7:- Evaluation

            return self.test(self, pic, trained_model)

            #pic = '2398605966_1d0c9e6a20.jpg'
            #image = encoding_test[pic].reshape((1,2048))
            #x=plt.imread(images_path+pic)
            #plt.imshow(x)
            #plt.show()

            #Example 1
            #print("Greedy Search:",self.greedySearch(self, encoded_img, trained_model))
            #print("Beam Search, K = 3:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 3))
            #print("Beam Search, K = 5:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 5))
            #print("Beam Search, K = 7:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 7))
            #print("Beam Search, K = 10:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 10))
        
            #pic = list(encoding_test.keys())[1]
            #image = encoding_test[pic].reshape((1,2048))
            ##x=plt.imread(images_path+pic)
            ##plt.imshow(x)
            ##plt.show()

            ##Example 2
            #print("Greedy Search:",self.greedySearch(self, encoded_img, trained_model))
            #print("Beam Search, K = 3:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 3))
            #print("Beam Search, K = 5:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 5))
            #print("Beam Search, K = 7:",self.beam_search_predictions(self, encoded_img, trained_model, beam_index = 7))


    def preprocess(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def encode(self, image):
        image = self.preprocess(self, image) 
        fea_vec = self.model_new.predict(image) 
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec

    def data_generator(self, descriptions, photos, wordtoix, max_length, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n=0
        # loop for ever over images
        while 1:
            for key, desc_list in descriptions.items():
                n+=1
                # retrieve the photo feature
                photo = photos[key+'.jpg']
                for desc in desc_list:
                    # encode the sequence
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    # split one sequence into multiple X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                        # store
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)

                if n==num_photos_per_batch:
                    yield ([array(X1), array(X2)], array(y))
                    X1, X2, y = list(), list(), list()
                    n=0

    #Step 6:- Greedy and Beam Search

    def test(self, pic, trained_model):
        encoded_img = {}
        encoded_img[pic] = self.encode(self, images_path+pic)
        img = encoded_img[pic].reshape((1,2048))
        greedy_search_caption = "Greedy Search: " + self.greedySearch(self, img, trained_model)
        print(greedy_search_caption)
        beam_search_caption = "Beam Search, K = 3: " + self.beam_search_predictions(self, img, trained_model, beam_index = 5)
        print(beam_search_caption)
        final_caption = greedy_search_caption + "\n" + beam_search_caption
        return final_caption
        #print("Beam Search, K = 5:",self.beam_search_predictions(self, img, trained_model, beam_index = 5))
        #print("Beam Search, K = 7:",self.beam_search_predictions(self, img, trained_model, beam_index = 7))
        #print("Beam Search, K = 10:",self.beam_search_predictions(self, img, trained_model, beam_index = 10))

    def greedySearch(self, photo, model):
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = [self.wordtoix[w] for w in in_text.split() if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.ixtoword[0][str(yhat)]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final

    def beam_search_predictions(self, image, model, beam_index = 3):
        start = [self.wordtoix[0]["startseq"]]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < self.max_length:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=self.max_length, padding='post')
                preds = model.predict([image,par_caps], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                # Getting the top <beam_index>(n) predictions and creating a 
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])
                    
            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]
    
        start_word = start_word[-1][0]
        intermediate_caption = [self.ixtoword[0][str(i)] for i in start_word]
        final_caption = []
    
        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption