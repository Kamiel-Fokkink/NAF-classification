# Backtranslation realized by googletrans==3.1.0a0
from googletrans import Translator
translator = Translator()


def back_translate(text,lang_inter=['en','es','ja',"zh-cn","ca",'ar','fi','el','cs','th']):
    """
    :param text: str, input
    :param lang_inter: list of intermediate languages
    :return: list of str, result
    
    """
    backtrans = []
    for lang in lang_inter:
        trans1 = translator.translate(text, src='fr', dest=lang)
        trans2 = translator.translate(trans1.text, src=lang, dest='fr')
        backtrans.append(trans2.text)

    return list({}.fromkeys(backtrans).keys()) #Only keep the unique results of backtranslations
