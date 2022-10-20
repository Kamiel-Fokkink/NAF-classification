import os
import pandas as pd
import numpy as np
from srcd.Preprocessing import preprocessing
from srcd.Modeling import modeling

train_path = "./data/naf_activite.csv"
mapping_path = "./data/naf_mapping.csv"
test_path = "./data/test.csv"


if __name__ == "__main__":

    classes = preprocessing.get_naf2_to_label(mapping_path)

    df_train = preprocessing.read_naf(train_path)
    df_train = df_train.sample(10000)
    df_test = pd.read_csv(test_path)

    df_train = preprocessing.map_naf5_to_naf2(df_train, mapping_path)
    df_train = preprocessing.apply_one_hot_encoder(df_train, list(classes.keys()))

    if not os.path.exists('./intermediate_outputs'):
        os.mkdir('./intermediate_outputs') 

    df_train.to_csv('./intermediate_outputs/train_onehot.csv', index=False)
    model = modeling.auto_nlp_modeling(model='camembert-base', data=df_train, nb_labels = len(list(classes.keys())), labels = list(classes.keys()))
    trained_model = model.fit()
    df_test_final = model.predict(trained_model, df_test)
    df_test_final['NewsId'] = df_test['NewsId']
    df_test_final = df_test_final.drop(['text'], axis=1)
    df_test_final.to_csv('./Outputs/predictions_new.csv', index=False)