import os
import pandas as pd
import numpy as np
from src.Modeling import modeling
from src.Preprocessing import preprocessing

train_path = "./data/train_clean.csv"
test_path = "./data/test.csv"
mapping_path = "./data/naf_mapping.csv"


if __name__ == "__main__":
    
    classes = preprocessing.get_naf2_to_label(mapping_path)
    df_train = pd.read_csv(train_path).sample(10)
    df_test = pd.read_csv(test_path)
    model = modeling.auto_nlp_modeling(model='camembert-base', data=df_train, nb_labels = len(list(classes.keys())), labels = list(classes.keys()))
    trained_model = model.fit()
    df_test_final = model.predict(trained_model, df_test)
    df_test_final['NewsId'] = df_test['NewsId']
    df_test_final = df_test_final.drop(['text'], axis=1)
    df_test_final.to_csv('./Outputs/predictions_new.csv', index=False)