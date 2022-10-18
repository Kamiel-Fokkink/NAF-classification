"""
Todo: put parameters into config

"""

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from .data_loader import ClassificationDataset

def read_naf(train_path: str, sep='|', index_col=0) -> pd.DataFrame:
    """ Read naf_activite.csv"""
    df = pd.read_csv(train_path, sep=sep, index_col=index_col)
    df = df.astype({'ACTIVITE': 'str', 'SIREN': 'object'})
    return df

def map_naf5_to_naf2(df: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    """ Process target classes"""
    naf5_to_naf2 = get_naf5_to_naf2(mapping_path)
    df['NAF2_CODE'] =  df['NAF_CODE'].map(naf5_to_naf2)
    df = df.fillna(-1).astype({'NAF2_CODE': 'int'})
    
    return df

def apply_one_hot_encoder(df: pd.DataFrame, classes: list) -> pd.DataFrame:
    """Apply one hot encoder to naf2 code and append the result as a column of lists"""
    encoder = get_one_hot_encoder(data=list(classes))

    # Use the one-hot encoder to transform our target variable
    df['encoded_label'] = df["NAF2_CODE"].apply(
    lambda x: encoder.transform([[x]]).toarray().tolist()[0])

    return df

def train_val_split(df: pd.DataFrame):
    """Train validation split on dataset"""

    texts = df["ACTIVITE"].tolist()
    labels = df["encoded_label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=.4, random_state=17, stratify=labels
    )
    return train_texts, val_texts, train_labels, val_labels

def tokenize_text(train_texts, val_texts, tokenizer):
    """Apply tokenizer to text data"""

    train_encodings = tokenizer(
        train_texts, truncation=True, max_length=300, padding=True
    )

    val_encodings = tokenizer(
        val_texts, truncation=True, max_length=300, padding=True
    )

    return train_encodings, val_encodings

def load_dataset(train_encodings, val_encodings, train_labels, val_labels) -> ClassificationDataset:
    """Gnerate torch datasets"""
    train_dataset = ClassificationDataset(train_encodings, train_labels)
    val_dataset = ClassificationDataset(val_encodings, val_labels)

    return train_dataset, val_dataset
    

############################## Preprocessing Utils Functions ###########

def get_one_hot_encoder(data: list) -> OneHotEncoder:
    data_to_encode = [[_] for _ in data]
    encoder = OneHotEncoder()
    encoder.fit(data_to_encode)
    return encoder


def get_naf2_to_label(mapping_path: str, sep=';',  encoding='latin-1') -> dict:
    
    df_mapping = pd.read_csv(mapping_path, sep=sep,  encoding=encoding)

    classes = (
        df_mapping[['naf2', 'naf2_label']]
        .dropna()
        .drop_duplicates()
        .astype({'naf2': 'int'})
        .set_index('naf2')
        .to_dict()['naf2_label']
        )
        
    classes[-1] = ''

    return classes

def get_naf5_to_naf2(mapping_path: str, sep=';',  encoding='latin-1') -> dict:
    
    df_mapping = pd.read_csv(mapping_path, sep=sep,  encoding=encoding)

    classes = (
        df_mapping[['naf5', 'naf2']]
        .set_index('naf5')
        .fillna(-1)
        .to_dict()['naf2']
        )
        
    return classes



