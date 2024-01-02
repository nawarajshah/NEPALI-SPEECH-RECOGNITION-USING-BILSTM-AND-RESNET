# Install the necessary system library and librosa dependencies
!apt-get -y install libsndfile1
!pip install librosa soundfile

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import time
from tqdm import tqdm
import edit_distance as ed

from model.configs import SR, device_name, UNQ_CHARS, INPUT_DIM, MODEL_NAME, NUM_UNQ_CHARS
from model.utils import CER_from_mfccs, batchify, clean_single_wav, gen_mfcc, indices_from_texts, load_model
from model.model import get_model

# Enable TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

def train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, epochs=100, batch_size=50):

    with strategy.scope():
        for e in range(0, epochs):
            start_time = time.time()

            len_train = len(train_wavs)
            len_test = len(test_wavs)
            train_loss = 0
            test_loss = 0
            test_CER = 0
            train_batch_count = 0
            test_batch_count = 0

            print("Training epoch: {}".format(e+1))
            for start in tqdm(range(0, len_train, batch_size)):
                end = None
                if start + batch_size < len_train:
                    end = start + batch_size
                else:
                    end = len_train
                x, target, target_lengths, output_lengths = batchify(
                    train_wavs[start:end], train_texts[start:end], UNQ_CHARS)

                with tf.GradientTape() as tape:
                    output = model(x, training=True)
                    loss = K.ctc_batch_cost(
                        target, output, output_lengths, target_lengths)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                train_loss += np.average(loss.numpy())
                train_batch_count += 1

            print("Testing epoch: {}".format(e+1))
            for start in tqdm(range(0, len_test, batch_size)):
                end = None
                if start + batch_size < len_test:
                    end = start + batch_size
                else:
                    end = len_test
                x, target, target_lengths, output_lengths = batchify(
                    test_wavs[start:end], test_texts[start:end], UNQ_CHARS)

                output = model(x, training=False)
                loss = K.ctc_batch_cost(
                    target, output, output_lengths, target_lengths)

                test_loss += np.average(loss.numpy())
                test_batch_count += 1

                input_len = np.ones(output.shape[0]) * output.shape[1]
                decoded_indices = K.ctc_decode(output, input_length=input_len,
                                       greedy=False, beam_width=100)[0][0]

                target_indices = [sent[sent != 0].tolist() for sent in target]
                predicted_indices = [sent[sent > 1].numpy().tolist() for sent in decoded_indices]

                len_batch = end - start
                for i in range(len_batch):
                    pred = predicted_indices[i]
                    truth = target_indices[i]
                    sm = ed.SequenceMatcher(pred, truth)
                    ed_dist = sm.distance()
                    test_CER += ed_dist / len(truth)
                test_CER /= len_batch

            train_loss /= train_batch_count
            test_loss /= test_batch_count
            test_CER /= test_batch_count

            rec = "Epoch: {}, Train Loss: {:.2f}, Test Loss {:.2f}, Test CER {:.2f} % in {:.2f} secs.\n".format(
                e+1, train_loss, test_loss, test_CER*100, time.time() - start_time)

            print(rec)

            # Save the final trained model
            model.save("model/trained_model1.h5")

# ... Rest of your code remains unchanged
