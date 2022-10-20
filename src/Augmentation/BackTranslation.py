# Backtranslation realized by googletrans==3.1.0a0
from doctest import OutputChecker
from googletrans import Translator
import pandas as pd

translator = Translator()

def back_translate(text: str,lang_inter=['en','es','ja',"zh-cn","ca",'ar','fi','el','cs','th']):
    """
    :param text: str, input
    :param lang_inter: list of intermediate languages
    :return: str result
    """
    backtrans = [text]
    for lang in lang_inter:
        trans1 = translator.translate(text, src='fr', dest=lang)
        trans2 = translator.translate(trans1.text, src=lang, dest='fr')
        backtrans.append(trans2.text)

    return  list({}.fromkeys(backtrans).keys()) #Only keep the unique results of backtranslations 

def back_trans_train(df: pd.DataFrame, labels:list):
    """
    Backtranslate all labels need to be augmented and generate a new dataframe with exploded translation rows
    :param df: DataFrame, trainset
    :param label: list of labels need to be augmented
    :return: new df, result
    """
    df1=df.copy()
    df1['ACTIVITE']=df1.apply(lambda x: back_translate(x.ACTIVITE) if (x.NAF2_CODE in labels) else x.ACTIVITE,axis=1)
    
    return df1.explode('ACTIVITE')
