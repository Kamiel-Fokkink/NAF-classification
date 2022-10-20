import os
import pandas as pd
from src.Preprocessing import preprocessing
from src.Modeling import modeling
import torch
torch.cuda.empty_cache()
from numba import cuda
cuda.select_device(0)
cuda.close()
cuda.select_device(0)

train_path = "./data/naf_activite.csv"
mapping_path = "./data/naf_mapping.csv"
test_path = "./data/test.csv"


if __name__ == "__main__":
    
    classes = preprocessing.get_naf2_to_label(mapping_path)

    df_train = preprocessing.read_naf(train_path)
    df_train = df_train.sample(n=1000, random_state=0)
    df_test = pd.read_csv(test_path)
    
    df_train = preprocessing.map_naf5_to_naf2(df_train, mapping_path)
    df_train = preprocessing.apply_one_hot_encoder(df_train, list(classes.keys()))

    if not os.path.exists('./intermediate_outputs'):
        os.mkdir('./intermediate_outputs') 
        
    df_train.to_csv('./intermediate_outputs/train_onehot.csv', index=False)
    model_list = ['camembert-base']
    #'camembert-base', 'camembert/camembert-large', 'camembert/camembert-base-wikipedia-4gb', 'camembert/camembert-base-oscar-4gb', 'camembert/camembert-base-ccnet-4gb', 'camembert/camembert-base-ccnet'
    for mod in model_list:
        print('start training', mod)
        try:
            torch.cuda.empty_cache()
            model = modeling.auto_nlp_modeling(data=df_train, nb_labels = len(list(classes.keys())), model=mod)
            trained_model = model.fit()
            output = model.predict(trained_model, df_test)
            #output.drop('NewsId')
            #df_test['predicted'] = output.predictions.tolist()
            if not os.path.exists('./Outputs'):
                os.mkdir('./Outputs')
            path = r"./Outputs/predictions_ {}.csv".format(mod.replace('/', '_'))
            output.to_csv(path, index=False)
        except Exception as e: 
            print(e)