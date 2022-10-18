import os
import pandas as pd
from preprocessing import *


train_path = "./data/naf_activite.csv"
mapping_path = "./data/naf_mapping.csv"


if __name__ == "__main__":
    
    classes = get_naf2_to_label(mapping_path)

    df = read_naf(train_path)
    df = map_naf5_to_naf2(df_train, mapping_path)

    df = apply_one_hot_encoder(df_train, list(classes.keys()))

    if not os.path.exists('./intermediate_outputs'):
        os.mkdir('./intermediate_outputs') 
        
    df.to_csv('./intermediate_outputs/train_onehot.csv', index=False)
    


    # train test split of data
                    
    # train_texts, val_texts, train_labels, val_labels = train_val_split(df_train)
    # train_encodings, val_encodings =  tokenize_text(train_texts, val_texts, tokenizer)
    # train_dataset, val_dataset = load_dataset(train_encodings, val_encodings, train_labels, val_labels)