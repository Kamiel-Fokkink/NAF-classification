import os
import pandas as pd
from preprocessing import *
from Augmentation import *

#train_path = "./data/naf_activite.csv"
#mapping_path = "./data/naf_mapping.csv"


def main_preprocessing(conf):

    input_file = conf['Inputs_path'] + conf['files_info']['naf_descriptions']['path_file']
    input_mapping = conf['Inputs_path'] + conf['files_info']['naf_description']['y_name']
    output_dir = conf['Outputs_path']
    output_file = conf['Outputs_path'] + conf['files_info']['naf_description']['path_file_preprocessed']

    classes = get_naf2_to_label(input_mapping)

    df_train = read_naf(input_file)
    df_train = map_naf5_to_naf2(df_train, input_mapping)
    df_train = undersample(df_train, threshold=1000)
    df_train.reset_index(inplace=True,drop=True)
    df_train.dropna(how='any',axis=0,inplace=True)
    df_train = back_trans_train(df_train)
    df_train = random_deletion_train(df_train)
    df_train = random_swap_train(df_train)
    df_train = apply_clean_paragraph(df_train, rm_ponctuation=True, rm_accent=True, rm_stopword=True)
    df_train = apply_one_hot_encoder(df_train, list(classes.keys()))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    df_train.to_csv(output_file, index=False)