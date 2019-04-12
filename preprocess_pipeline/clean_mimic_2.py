import string
import re
import os

mimic_file_src = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "mimic_paragraphs_cleaned.txt")
mimic_file_dest = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "mimic_paragraphs_cleaned2.txt")
f = open(mimic_file_src, 'r')
g = open(mimic_file_dest, 'w')
for line in f:
    line = line[:-1]
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    word = line.translate(translator)
    split_chars_1 = re.sub(r'(?<=\w)(?<!LOCATIO)N', " N", word)
    split_chars_2 = re.sub(r'N(?=\w)(?!AME)', "N ", split_chars_1)
    reduced_n = re.sub(r'(?<=N)( N)+', ' ', split_chars_2)
    reduced_name = re.sub(r'(NAME )+', 'NAME ', reduced_n)
    cleaned_line = ' '.join(reduced_name.split())
    g.write(cleaned_line + '\n')
f.close()
g.close()
