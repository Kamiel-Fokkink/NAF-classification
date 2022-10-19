import os
import pandas as pd
from src.Preprocessing import preprocessing
from src.Modeling import modeling

train_path = "./data/naf_activite.csv"
mapping_path = "./data/naf_mapping.csv"
test_path = "./data/test.csv"


if __name__ == "__main__":
    
    classes = preprocessing.get_naf2_to_label(mapping_path)

    df_train = preprocessing.read_naf(train_path)
    df_train = df_train.sample(1000)
    df_test = pd.read_csv(test_path)
    
    df_train = preprocessing.map_naf5_to_naf2(df_train, mapping_path)
    df_train = preprocessing.apply_one_hot_encoder(df_train, list(classes.keys()))

    if not os.path.exists('./intermediate_outputs'):
        os.mkdir('./intermediate_outputs') 
        
    df_train.to_csv('./intermediate_outputs/train_onehot.csv', index=False)
    model = modeling.auto_nlp_modeling(data=df_train, nb_labels = len(list(classes.keys())))
    trained_model = model.fit()
    output = model.predict(trained_model, df_test)
    df_test['predicted'] = output.predictions.tolist()
    df_test.to_csv('./Outputs/predictions.csv', index=False)