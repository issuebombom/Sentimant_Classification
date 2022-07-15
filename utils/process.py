from utils.tokenization_kobert import KoBertTokenizer
from utils.params_data import *
from tqdm import tqdm
import numpy as np
import re


def text_cleaner(sentence):
    # 특수 문자 제거
    return re.sub("[^\s0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]", "", sentence)


def tokenizer(cleaned_sentence):
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    # Tokenizing / Tokens to sequence numbers / Padding
    encoded_dict = tokenizer.encode_plus(text=cleaned_sentence, padding='max_length', truncation=True,
                                         max_length=SEQ_LEN)

    return encoded_dict


def preprocess(text_array, label_array):

    token_ids = []
    token_masks = []
    token_segments = []
    labels = []

    for idx in tqdm(range(len(text_array))):

        sentence = text_array.iloc[idx]

        # cleaned sentence
        cleaned_sentence = text_cleaner(sentence)
        # Tokenizing / Tokens to sequence numbers / Padding
        encoded_dict = tokenizer(cleaned_sentence)
      
        token_ids.append(encoded_dict['input_ids']) # tokens_tensor
        token_masks.append(encoded_dict['attention_mask']) # masks_tensor
        token_segments.append(encoded_dict['token_type_ids']) # segments_tensor

        labels.append(label_array.iloc[idx])

    data_inputs = (np.array(token_ids), np.array(token_masks), np.array(token_segments))
    data_labels = np.array(labels)

    return data_inputs, data_labels
