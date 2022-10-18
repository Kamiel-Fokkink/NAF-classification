import pandas as pd


def read_naf(train_path, sep='|', index_col=0):
    """ Read naf_activite.csv"""
    train_df = pd.read_csv(train_path, sep, index_col=index_col)
    train_df = train_df.astype({'ACTIVITE': 'str', 'SIREN': 'object'})

    return train_df

def map_naf5_to_naf2(df, mapping_path):
    naf5_to_naf2 = get_naf5_to_naf2(mapping_path)
    df['NAF2_CODE'] =  df['NAF_CODE'].map(naf5_to_naf2)
    df = df.fillna(-1).astype({'NAF2_CODE': 'int'})
    
    return df

############################## Preprocessing Utils Functions ###########

def get_naf2_to_label(mapping_path, sep=';',  encoding='latin-1'):
    
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

def get_naf5_to_naf2(mapping_path, sep=';',  encoding='latin-1'):
    
    df_mapping = pd.read_csv(mapping_path, sep=sep,  encoding=encoding)

    classes = (
        df_mapping[['naf5', 'naf2']]
        .set_index('naf5')
        .fillna(-1)
        .to_dict()['naf2']
        )
        
    return classes



