# %pip install googletrans==3.1.0a0
from doctest import OutputChecker
from googletrans import Translator
import pandas as pd
import re
import random

translator = Translator()

def back_translate(text: str, num_lang=10):
    """
    Create a list of sentences generated by back translation through up to 10 intermediate languages
    :param text: str, input
    :param num_lang: interger, number of intermediate languages, no larger than 10
    :return: list, result
    """
    lang_inter = ['en','es','ja',"zh-cn","ca",'ar','fi','el','cs','th']
    lang_inter = lang_inter[:num_lang]
    backtrans = [text]
    for lang in lang_inter:
        trans1 = translator.translate(text, src='fr', dest=lang)
        trans2 = translator.translate(trans1.text, src=lang, dest='fr')
        backtrans.append(trans2.text)

    return  list({}.fromkeys(backtrans).keys()) #Only keep the unique results of backtranslations 

def back_trans_train(df: pd.DataFrame, threshold=250):
    """
    Augment train set by back translation
    :param df: DataFrame, trainset
    :param threshold: classes with number of samples less than threshold will be augmented
    :return: df, result
    """
    count_by_class = df.groupby(['NAF2_CODE']).count()['ACTIVITE']
    labels = list(count_by_class[count_by_class<threshold].index)
    labels1 = list(count_by_class[count_by_class<120].index)
    df1 = df.copy()
    df1 = df1[df1["NAF2_CODE"].isin(labels)]
    # Back translation through 10 languages if count by class less than 150, otherwise we take 5 to cut running time
    df1['ACTIVITE'] = df1.apply(lambda x: back_translate(x.ACTIVITE) if x.NAF2_CODE in labels1 else back_translate(x.ACTIVITE,num_lang=5) ,axis=1)
    df1 = df1.explode('ACTIVITE').reset_index(drop=True)

    return  pd.concat([df1,df],axis=0).drop_duplicates()

def random_deletion(text_line: str):
    """
    Radonmly delete one word from a text with more than 8 words
    :param text_line: str
    :return: output_text: str
    """
    text_array = text_line.split()
    if len(text_array)>8:
        delete_token_index = random.randint(0, len(text_array)-1)
        text_array[delete_token_index] = ''
        output_text = " ".join(text_array)
        return output_text

def random_deletion_train(df: pd.DataFrame, threshold=400):
    """
    Augment train set by randomly delete one word
    :param df: DataFrame, trainset
    :param threshold: classes with number of samples less than threshold will be augmented
    :return: df, result
    """
    count_by_class = df.groupby(['NAF2_CODE']).count()['ACTIVITE']
    labels = list(count_by_class[count_by_class<threshold].index)
    df1 = df.copy()
    df1 = df1[df1["NAF2_CODE"].isin(labels)]
    # Back translation through 10 languages if count by class less than 150, otherwise we take 6 to cut running time
    df1['ACTIVITE'] = df1.apply(lambda x: random_deletion(x.ACTIVITE) ,axis=1)
    df1 = df1.reset_index(drop=True)

    return  pd.concat([df1,df],axis=0).drop_duplicates()

def random_swap(text_line):
    """
    Radonmly swap 2 words from a text with more than 10 words
    :param text_line: str
    :return: output_text: str
    """
    text_array = text_line.split()
    if len(text_array)>10:
        swap_token_index = random.sample(range(0, len(text_array)-1),2)
        temp = text_array[swap_token_index[0]]
        text_array[swap_token_index[0]] = text_array[swap_token_index[1]]
        text_array[swap_token_index[1]] = temp
        output_text = " ".join(text_array)
        return output_text

def random_swap_train(df: pd.DataFrame, threshold=400):
    """
    Augment train set by randomly swap 2 words
    :param df: DataFrame, trainset
    :param threshold: classes with number of samples less than threshold will be augmented
    :return: df, result
    """
    count_by_class = df.groupby(['NAF2_CODE']).count()['ACTIVITE']
    labels = list(count_by_class[count_by_class<threshold].index)
    df1 = df.copy()
    df1 = df1[df1["NAF2_CODE"].isin(labels)]
    # Back translation through 10 languages if count by class less than 150, otherwise we take 6 to cut running time
    df1['ACTIVITE'] = df1.apply(lambda x: random_swap(x.ACTIVITE) ,axis=1)
    df1 = df1.reset_index(drop=True)

    return  pd.concat([df1,df],axis=0).drop_duplicates()