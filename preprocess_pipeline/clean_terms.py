import re


def clean_word(text):
    target_sentence = text.lower()
    remove_newline = re.sub(r'\\n', ' ', target_sentence)
    contents_without_paren = re.sub(r'\(.*?\)', "", remove_newline)
    cleanNum = re.sub(r"[\d]+(\-(?=[A-Za-z])|:)+", "N ", contents_without_paren)
    cleanNum_2 = re.sub(r"[\d]+[\.\-?,%\d]*", "N", cleanNum)

    contents_without_extra_spaces = re.sub(r' +', " ", cleanNum_2)
    cleanNum_3 = re.sub(r"[\d]+[\-] ", "N", contents_without_extra_spaces)
    cleanNum_4 = re.sub(r"[\d]+[\-][/\.?:,%\d]*", "N ", cleanNum_3)
    cleanNum_5 = re.sub(r"[\d]+[/\.?:,%\d]*(?!>)", "N", cleanNum_4)

    reduced_n = re.sub(r'(?<!\w)N( N)+', 'N', cleanNum_5)
    reduced_n2 = re.sub(r'NN+', 'N', reduced_n)
    remove_amp = re.sub(r'&', ' and ', reduced_n2)
    remove_punc = re.sub(r'[,+\-\.;=:\*@#\?\$_\/%"^\(\)\\\|`\[\]{}]+ *', " ", remove_amp)
    cleaned_word = remove_punc
    return cleaned_word