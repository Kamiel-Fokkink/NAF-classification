"""
Todo: put parameters into config

"""
import nltk
import unidecode
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from data_loader import ClassificationDataset

# =====================Variables for cleaning text=================
characters_to_remove = ["@", "/", "#", ".", ",", "!", "?", "(", ")", "-", "_","’","'", "\"", ":"]
transformation_dict = {initial:" " for initial in characters_to_remove}
stop_words = nltk.corpus.stopwords.words("french")
stop_words = [unidecode.unidecode(stopword) for stopword in stop_words]
# =================================================================



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

def undersample(df, threshold=1200):
    sample_amounts = aux_get_class_count(df, threshold=threshold)
    df = df.groupby('NAF2_CODE').apply(lambda g: g.sample(
        n=sample_amounts[g.name],
        replace=len(g) < sample_amounts[g.name]
    )).reset_index(drop=True)

    return df

def apply_one_hot_encoder(df: pd.DataFrame, classes: list) -> pd.DataFrame:
    """Apply one hot encoder to naf2 code and append the result as a column of lists"""
    encoder = get_one_hot_encoder(data=list(classes))

    # Use the one-hot encoder to transform our target variable
    df['encoded_label'] = df["NAF2_CODE"].apply(
    lambda x: encoder.transform([[x]]).toarray().tolist()[0])

    return df


def apply_clean_paragraph(df: pd.DataFrame, rm_ponctuation=True, rm_accent=True, rm_stopword=True)  -> pd.DataFrame:
    """Apply clean paragraph to column activite"""
    
    df['ACTIVITE_clean'] = df['ACTIVITE'].apply(lambda x: clean_paragraph(x, rm_ponctuation=rm_ponctuation, rm_accent=rm_accent, rm_stopword=rm_stopword))

    return df


def train_val_split(df: pd.DataFrame, clean=True):
    """Train validation split on dataset"""
    if clean:
        texts = df["ACTIVITE_clean"].to_list()
    else:
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

def clean_paragraph(comment, rm_ponctuation, rm_accent, rm_stopword):
    '''Cleans a paragraphs'''

    comment = comment.lower()
    if rm_ponctuation:
        comment = aux_remove_ponctuation(comment)
    if rm_accent:
        comment = aux_remove_accent(comment)
    if rm_stopword:
        comment = aux_remove_stopword(comment)

    return comment


def aux_remove_ponctuation(comment):
   
    return comment.translate(str.maketrans(transformation_dict))

def aux_remove_accent(comment):
    return unidecode.unidecode(comment)

def aux_remove_stopword(comment):
    comment = [token for token in nltk.word_tokenize(comment) if token not in stop_words] 
    return " ".join(comment)


def aux_get_keep_num(class_cnt, threshold):
    base = threshold - 150 * np.log10(threshold)
    if class_cnt > threshold:
        return  int(base  + 150 * np.log10(class_cnt)) 
    else:
        return class_cnt

def aux_get_class_count(df, threshold) -> list:
    count_by_class = df.groupby(['NAF2_CODE']).count()['ACTIVITE']
    return {count_by_class.keys()[i]: aux_get_keep_num(count_by_class.values[i], threshold) for i in range(len(count_by_class)) }



def main_preprocessing(conf):

    input_file = conf['paths']['Inputs_path'] + conf['files_info']['naf_descriptions']['path_file']
    input_mapping = conf['paths']['Inputs_path'] + conf['files_info']['naf_descriptions']['y_name']
    output_dir = conf['paths']['Outputs_path']
    output_file = conf['paths']['Outputs_path'] + conf['files_info']['naf_descriptions']['path_file_preprocessed']

    classes = get_naf2_to_label(input_mapping)

    df_train = read_naf(input_file)
    df_train = map_naf5_to_naf2(df_train, input_mapping)
    df_train = undersample(df_train, threshold=1000)
    df_train.reset_index(inplace=True,drop=True)
    df_train.dropna(how='any',axis=0,inplace=True)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    df_train.to_csv(output_file, index=False)