import os
import pandas as pd
from preprocessing import *

train_path = "./data/naf_activite.csv"
mapping_path = "./data/naf_mapping.csv"


if __name__ == "__main__":
    
    classes = get_naf2_to_label(mapping_path)

    df_train = read_naf(train_path)
    df_train = map_naf5_to_naf2(df_train, mapping_path)

    df_train = apply_one_hot_encoder(df_train, list(classes.keys()))

    if not os.path.exists('./intermediate_outputs'):
        os.mkdir('./intermediate_outputs') 
        
    df_train.to_csv('./intermediate_outputs/train_onehot.csv', index=False)


    # example for using tokenizer and pretrained model

    # from sklearn.model_selection import train_test_split
    # from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # from transformers import Trainer, TrainingArguments

    # model = "camembert-base"
    # nb_labels = 89

    # tokenizer = AutoTokenizer.from_pretrained(model)
    # clf_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=nb_labels)

    # train_texts, val_texts, train_labels, val_labels = train_val_split(df_train)
    # train_encodings, val_encodings =  tokenize_text(train_texts, val_texts, tokenizer)
    # train_dataset, val_dataset = load_dataset(train_encodings, val_encodings, train_labels, val_labels)