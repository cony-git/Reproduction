# ======================Loading packages ============================== #
from __future__ import division, print_function, absolute_import

import csv
import os
import pickle

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import load_model, Model
from sklearn import metrics as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)


class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, dim=10240, header_len=2000, batch_size=32, max_section_num=8,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.dim = dim
        self.header_len = header_len
        self.max_section_num = max_section_num
        self.list_IDs = list_IDs
        self.datasets = datasets
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.zeros((self.batch_size, self.max_section_num, self.dim), dtype=float)
        y = np.zeros(self.batch_size, dtype=float)
        X_header = np.zeros((self.batch_size, self.header_len), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # base_path = "/home/malware_data/{0}/{1}{2}"
            # item = self.datasets.loc[ID]
            #
            # pointer = item['pointerto_raw_data'].split(',')
            # size = item['virtual_size'].split(',')
            #
            # file_path = base_path.format(item["mw_file_directory"], item["mw_file_hash"], item["mw_file_size"])
            base_path = "E:\panpan_static_sample"  # "/home/malware_data/{0}/{1}{2}"
            item = self.datasets.loc[ID]

            pointer = item['pointerto_raw_data'].split(',')
            size = item['virtual_size'].split(',')

            # file_path = base_path.format(item["mw_file_directory"], item["mw_file_hash"], item["mw_file_size"])
            file_path = os.path.join(base_path, os.path.basename(item["mw_file_directory"]))

            in_file = open(file_path, 'rb')
            bytes_data = [int(single_byte) for single_byte in in_file.read(self.header_len)]
            X_header[i, 0:len(bytes_data)] = bytes_data

            for j in range(min(8, len(pointer))):
                in_file.seek(int(pointer[j]))
                bytes_data = [int(single_byte) for single_byte in in_file.read(min(self.dim, int(size[j])))]
                X[i, j, 0:len(bytes_data)] = bytes_data

            y[i] = self.labels[ID]

        X = np.transpose(X, (0, 2, 1)) / 255.0
        return [X, X_header, y], y


class AutoencoderLossLayer(Layer):
    def __init__(self, **kwargs):
        self.results = None
        super(AutoencoderLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AutoencoderLossLayer, self).build(input_shape)

    def calculate_loss(self, y_true, y_pred):
        loss = tf.losses.mean_squared_error(y_true, y_pred)
        return loss

    def call(self, inputs):
        y_pred = inputs[0]
        y_true = inputs[1]
        y = inputs[2]
        loss = 5 * self.calculate_loss(y_true, y_pred)
        self.add_loss(loss, inputs=inputs)
        self.results = y
        return y

    # return output shape
    def compute_output_shape(self, input_shape):
        return tuple(K.int_shape(self.results))


class GradientBoostingDeicisionTreeLayer(Layer):
    def __init__(self, n_tree=200, depth=3, n_node=32, dropout_rate=0.5, batch_size=32, learning_rate=0.01,
                 **kwargs):
        self.is_placeholder = True
        self.n_tree = n_tree
        self.depth = depth
        self.n_leaf = 2 ** (self.depth + 1) - 1
        self.n_node = n_node
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.results = None
        super(GradientBoostingDeicisionTreeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_t_ensemble = self.add_weight(name='w_t_ensemble',
                                            shape=(self.n_tree, input_shape[0][1], self.n_node),
                                            initializer='uniform',
                                            trainable=True)
        self.b_t_ensemble = self.add_weight(name='b_t_ensemble',
                                            shape=(self.n_tree, self.n_node),
                                            initializer='uniform',
                                            trainable=True)
        self.w_d_ensemble = self.add_weight(name='w_d_ensemble',
                                            shape=(self.n_tree, self.n_node, self.n_leaf),
                                            initializer='uniform',
                                            trainable=True)
        self.b_d_ensemble = self.add_weight(name='b_d_ensemble',
                                            shape=(self.n_tree, self.n_leaf),
                                            initializer='uniform',
                                            trainable=True)
        self.w_l_ensemble = self.add_weight(name='w_l_ensemble',
                                            shape=(self.n_tree, self.n_leaf + 1),
                                            initializer='uniform',
                                            trainable=True)
        self.b_l_ensemble = self.add_weight(name='b_l_ensemble',
                                            shape=(self.n_tree,),
                                            initializer='uniform',
                                            trainable=True)
        self.f_0 = self.add_weight(name='f_0', shape=(1,), initializer=keras.initializers.Constant(value=0.5),
                                   trainable=True)
        super(GradientBoostingDeicisionTreeLayer, self).build(input_shape)

    def construct_decision_trees(self, input_layer):
        decision_p_e = []
        leaf_p_e = []
        for i in range(self.n_tree):
            w_t = self.w_t_ensemble[i]
            b_t = self.b_t_ensemble[i]
            w_d = self.w_d_ensemble[i]
            b_d = self.b_d_ensemble[i]
            w_l = self.w_l_ensemble[i]

            tree_layer = tf.nn.relu(tf.add(tf.matmul(input_layer, w_t), b_t))
            tree_layer = Dropout(self.dropout_rate)(tree_layer)
            decision_p = tf.nn.sigmoid(tf.add(tf.matmul(tree_layer, w_d), b_d))
            leaf_p = w_l

            decision_p_e.append(decision_p)
            leaf_p_e.append(leaf_p)
        return decision_p_e, leaf_p_e

    def neural_decision_trees(self, decision_p_e, n_depth, n_leaf, n_batch):
        flat_decision_p_e = []

        for decision_p in decision_p_e:
            decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)
            decision_p_pack = tf.stack([decision_p, decision_p_comp])
            flat_decision_p = tf.reshape(decision_p_pack, [-1])
            flat_decision_p_e.append(flat_decision_p)

        batch_0_indices = \
            tf.tile(tf.expand_dims(tf.range(0, n_batch * n_leaf, n_leaf), 1), [1, (n_leaf + 1)])

        in_repeat = int((n_leaf + 1) / 2)
        out_repeat = n_batch

        batch_complement_indices = \
            tf.reshape(tf.convert_to_tensor([[0] * in_repeat, [n_batch * n_leaf] * in_repeat]
                                            * out_repeat), (n_batch, (n_leaf + 1)))

        mu_e = []

        for i, flat_decision_p in enumerate(flat_decision_p_e):
            mu = tf.gather(flat_decision_p, tf.add(batch_0_indices, batch_complement_indices))
            mu_e.append(mu)

        for d in range(1, n_depth + 1):
            indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
            tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                              [1, 2 ** (n_depth - d + 1)]), [1, -1])
            batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [n_batch, 1]))

            in_repeat = int(in_repeat / 2)
            out_repeat = int(out_repeat * 2)

            batch_complement_indices = \
                tf.reshape(tf.convert_to_tensor([[0] * in_repeat, [n_batch * n_leaf] * in_repeat]
                                                * out_repeat), (n_batch, (n_leaf + 1)))

            mu_e_update = []
            for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
                mu = tf.multiply(mu, tf.gather(flat_decision_p, tf.add(batch_indices, batch_complement_indices)))
                mu_e_update.append(mu)

            mu_e = mu_e_update

        return mu_e

    def calculate_loss(self, y_true, y_pred):
        loss = tf.losses.mean_squared_error(y_true, y_pred)
        return loss

    def probability_y_x(self, mu_e, leaf_p_e, b_l_e, y_true, batch_size):
        predict = self.f_0
        loss = 0
        all_predict = []
        all_predict.append(tf.tile(predict, [batch_size, ]))

        for i in range(self.n_tree):
            mu = mu_e[i]
            leaf_p = leaf_p_e[i]
            b_l = b_l_e[i]

            f_i = tf.nn.tanh(tf.add
                             (tf.reduce_sum(tf.multiply(mu, tf.tile(tf.expand_dims(leaf_p, 0), [batch_size, 1])),
                                            keepdims=True, axis=1), b_l))
            error = y_true - predict
            predict = predict + self.learning_rate * f_i
            all_predict.append(predict[:, 0])
            loss += self.learning_rate * self.calculate_loss(error, f_i)

        final = tf.transpose(all_predict, (1, 0))
        return final, loss

    def call(self, inputs):
        input = inputs[0]
        y_true = inputs[1]

        decision_p_e, leaf_p_e = self.construct_decision_trees(input)
        mu_e = self.neural_decision_trees(decision_p_e, self.depth, self.n_leaf, self.batch_size)
        py_x, loss = self.probability_y_x(mu_e, leaf_p_e, self.b_l_ensemble, y_true, self.batch_size)
        self.results = py_x
        self.add_loss(loss, inputs=inputs)
        return py_x

    # return output shape
    def compute_output_shape(self, input_shape):
        return tuple(K.int_shape(self.results))


class FR_loss():
    def __init__(self, fpr=0.001, beta=30):
        self.fpr = fpr
        self.beta = beta

    def fr_loss_func(self):
        def loss_func(y_true, y_pred):
            fn_loss = K.sum(y_true * K.binary_crossentropy(y_true, y_pred), axis=-1)
            fnr_loss = fn_loss / (K.sum(y_true) + K.epsilon())

            fp_loss = K.sum((1 - y_true) * K.binary_crossentropy(y_true, y_pred), axis=-1)
            fpr_loss = fp_loss / (K.sum((1 - y_true)) + K.epsilon())

            loss = fnr_loss + K.maximum(0., self.beta * (fpr_loss - self.fpr))

            return loss

        return loss_func


fr_loss = FR_loss()

from sklearn.metrics import roc_curve


class RecallReportor(keras.callbacks.Callback):
    def __init__(self, val_generator, y_true):
        self.val_generator = val_generator
        self.y_true = y_true

    def on_train_begin(self, logs={}):
        self.recalls = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(generator=self.val_generator, max_queue_size=10,
                                              workers=2, use_multiprocessing=True, verbose=0)

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        fpr, tpr, _ = roc_curve(self.y_true[:len(y_pred)], y_pred)

        # recall
        idx = find_nearest(fpr, 0.001)
        self.recalls.append(tpr[idx])

        with open(proposed_model_name + "_val_recall.csv", "a", newline='') as result_csvfile:
            result_writer = csv.writer(result_csvfile)
            result_writer.writerow([str(epoch), str(tpr[idx])])
            result_csvfile.flush()

        print("current recall:" + str(tpr[idx]))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def place_holder_loss(y_true, y_pred):
    return K.variable(0.0)


class Proposed_Model():
    def __init__(self, input_shape, input_header_shape, batch_size):
        self.input_shape = input_shape
        self.input_header_shape = input_header_shape
        self.batch_size = batch_size
        self.model = self.get_model()

    def get_model(self):
        x_input = Input(shape=self.input_shape)
        x_header = Input(shape=self.input_header_shape)
        y_true = Input(shape=(1,))

        x = Conv1D(filters=256, kernel_size=4, padding='same')(x_input)
        x = MaxPooling1D(4, padding='same')(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=256, kernel_size=4, padding='same')(x)
        x = MaxPooling1D(4, padding='same')(x)

        encoded = Activation('relu')(x)

        x = Conv1D(filters=256, kernel_size=4, padding='same')(encoded)
        x = UpSampling1D(4)(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=256, kernel_size=4, padding='same')(x)
        x = UpSampling1D(4)(x)
        x = Activation('relu')(x)

        decoded = Conv1D(filters=8, kernel_size=4, padding='same')(x)

        encoded_feature = GlobalMaxPooling1D()(encoded)

        e_x = Embedding(256, 8, input_length=self.input_header_shape[0])(x_header)
        c_x = Conv1D(filters=256, kernel_size=16, strides=16)(e_x)
        c_x_sigmoid = Conv1D(filters=256, kernel_size=16, strides=16, activation='sigmoid')(e_x)
        g_x = Multiply()([c_x, c_x_sigmoid])
        p_x = GlobalMaxPooling1D()(g_x)

        merged_feature = Concatenate(axis=-1)([p_x, encoded_feature])
        x = BatchNormalization()(merged_feature)
        x = Dropout(0.2)(x)

        x = Dense(256, activation='sigmoid')(x)

        y = GradientBoostingDeicisionTreeLayer(batch_size=self.batch_size)([x, y_true])

        y = AutoencoderLossLayer()([x_input, decoded, y])

        y = Dense(1, activation='sigmoid')(y)

        model = Model(inputs=[x_input, x_header, y_true], outputs=y)

        model.compile(loss=fr_loss.fr_loss_func(), optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, model_name, x_train, y_train, x_val, y_val, max_epoch, batch_size=32):
        file_path = model_name + "-{epoch:02d}.h5"

        # Generators
        training_generator = DataGenerator(range(len(x_train)), x_train, y_train, dim=10240,
                                           header_len=self.input_header_shape[0], batch_size=batch_size,
                                           max_section_num=8, shuffle=True)

        validation_generator = DataGenerator(range(len(x_val)), x_val, y_val, dim=10240,
                                             header_len=self.input_header_shape[0], batch_size=batch_size,
                                             max_section_num=8,
                                             shuffle=False)

        recallReportor = RecallReportor(validation_generator, y_val)
        early_stopping = EarlyStopping("val_acc", patience=10, verbose=1, mode='auto')
        check_point = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=False, mode='auto')
        callbacks_list = [check_point, early_stopping, recallReportor]

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True, workers=8,
                                 epochs=max_epoch, verbose=1, callbacks=callbacks_list)

    def predict(self, model_name, x_test, y_test, batch_size=32):
        model = load_model(model_name,
                           custom_objects={'GradientBoostingDeicisionTreeLayer': GradientBoostingDeicisionTreeLayer,
                                           'AutoencoderLossLayer': AutoencoderLossLayer,
                                           'loss_func': fr_loss.fr_loss_func()})
        testing_generator = DataGenerator(range(len(x_test)), x_test, y_test, dim=10240,
                                          header_len=self.input_header_shape[0], batch_size=batch_size,
                                          max_section_num=8,
                                          shuffle=False)
        y_pred = model.predict_generator(generator=testing_generator, verbose=1, use_multiprocessing=True, workers=2)
        y_pred = y_pred.reshape((-1,))

        y_true = y_test[0:y_pred.shape[0]]
        fp_np_index = np.where(y_true == 0)
        print(y_true)
        print(y_pred)

        auc = sm.roc_auc_score(y_true, y_pred)

        fp_np = y_pred[fp_np_index[0]].shape[0]
        thre_index = int(np.floor(fp_np - fp_np * 0.001))
        sorted_pred_prob = np.sort(y_pred[fp_np_index[0]], axis=0)
        thre = sorted_pred_prob[thre_index]

        y_pred_prob = np.vstack((1 - y_pred, y_pred)).transpose()

        y_pred_prob[:, 0] = 0.5
        y_pred_label = np.argmax(y_pred_prob, axis=-1)
        accu = sm.accuracy_score(y_true, y_pred_label)

        y_pred_prob[:, 0] = thre
        y_pred_label = np.argmax(y_pred_prob, axis=-1)

        tn, fp, fn, tp = sm.confusion_matrix(y_true, y_pred_label).ravel()
        fp_rate = fp / (fp + tn)
        recall_rate = tp / (tp + fn)
        print('fp_rate:' + str(fp_rate))
        print('threshold:' + str(thre))
        print('recall_rate:' + str(recall_rate))
        print('auc:' + str(auc))
        print('accu:' + str(accu))
        print('\n')

        with open(proposed_model_name + ".csv", "a", newline='') as result_csvfile:
            result_writer = csv.writer(result_csvfile)
            result_writer.writerow([model_name, fp_rate, thre, recall_rate, auc, accu])
            result_csvfile.flush()

        return fp_rate, thre, recall_rate, auc, accu


# proposed_model_name = 'proposed_model_loss'
# prefix = "../saved_models/"
#
# for i in range(8):
#     header_len = 4000
#     proposed_model = Proposed_Model((10240, 8), (header_len,), batch_size=32)
#     proposed_model.train(prefix + proposed_model_name + str(i), x_train[i], y_train[i], x_val[i],
#                          y_val[i], max_epoch=100,
#                          batch_size=32)
proposed_model_name = 'proposed_model_loss'
prefix = "../saved_models/"

# for i in range(8):
header_len = 4000
with open("feature_true.pkl", 'rb') as file:
    feature_true = pickle.load(file)
with open("label_true.pkl", 'rb') as file:
    label_true = pickle.load(file)

with open("feature_false.pkl", 'rb') as file:
    feature_false = pickle.load(file)
with open("label_false.pkl", 'rb') as file:
    label_false = pickle.load(file)

x = pd.concat([feature_true,feature_false],axis=0,ignore_index=True)
y = np.concatenate([label_true,label_false],axis=0)
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=0)
x_train = x_train.reset_index(drop=True)
x_val = x_val.reset_index(drop=True)

proposed_model = Proposed_Model((10240, 8), (header_len,), batch_size=32)
proposed_model.train(prefix + proposed_model_name, x_train, y_train, x_val,
                        y_val, max_epoch=100,
                        batch_size=32)