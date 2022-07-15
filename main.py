import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import callbacks
import os
import numpy as np
import argparse
import pandas as pd
from utils.params_data import *
from utils.process import preprocess, text_cleaner, tokenizer
from models.kobert import kobert_model


def main(config):
    model = kobert_model()

    if config.mode == 'train':
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_data, train_label = train_df['reviews'], train_df['label']
        test_data, test_label = test_df['reviews'], test_df['label']
        train_inputs, train_labels = preprocess(train_data, train_label)
        checkpoint_path = 'save/'

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Save best model weights
        callback_checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path + 'best_Kobert_weights.h5',
                                                        monitor='val_sparse_categorical_accuracy',
                                                        save_best_only=True, 
                                                        save_weights_only=True,
                                                        verbose=1) 
                                                        
        # Early-stopping for preventing the over fitting
        callback_earlystop = callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5)

        # Fit
        history = model.fit(train_inputs, train_labels, validation_split=validation_split,
                            epochs=epochs, batch_size=batch_size,
                            verbose=verbose,
                            callbacks=[callback_checkpoint, callback_earlystop])

        # Test
        test_inputs, test_labels = preprocess(test_data, test_label)
        model.load_weights(filepath=checkpoint_path + 'best_Kobert_weights.h5')

        pred = model.predict(test_inputs)
        pred = tf.argmax(pred, axis=1)
    
        print(f"TEST DATA ACCURACY: {accuracy_score(pred, test_labels)}")

    if config.mode == 'predict':

        # Load model weights
        model.load_weights(filepath='./save/best_Kobert_weights.h5')
        # Preprocessing
        cleaned_sentence = text_cleaner(config.input)
        encoded_dict = tokenizer(cleaned_sentence)
        token_ids = np.array(encoded_dict['input_ids']).reshape(1, -1)  # tokens_tensor
        token_masks = np.array(encoded_dict['attention_mask']).reshape(1, -1)  # masks_tensor
        token_segments = np.array(encoded_dict['token_type_ids']).reshape(1, -1)  # segments_tensor
        # Predict sentiment
        prediction = model.predict((token_ids, token_masks, token_segments))

        predicted_probability = np.round(np.max(prediction) * 100, 2)
        predicted_class = ['부정', '긍정'][np.argmax(prediction, axis=1)[0]]
        print('입력한 문장: {}'.format(config.input))
        print('{}% 확률로 {} 리뷰입니다.'.format(predicted_probability, predicted_class))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Option
    parser.add_argument('--mode', type=str, default='predict')
    parser.add_argument('--input', type=str, default='감정을 분석할 문장입니다.')

    args = parser.parse_args()

    main(args)
