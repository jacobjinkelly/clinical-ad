import re
import string

f = open("mimic_paragraphs.txt", 'r')
g = open("mimic_paragraphs_cleaned.txt", 'w')
for line in f:
    target_sentence = line[:-1].lower()
    remove_newline = re.sub(r'\\n', ' ', target_sentence)
    contents_without_paren = re.sub(r'\(.*?\)', "", remove_newline)

    contents_without_names = re.sub(r'\[\*\*.*?(NAME|name|Name).*?\*\*\]( \[\*\*.*?(NAME|name|Name).*?\*\*\])*',
                                    " NAME ", contents_without_paren)
    contents_without_h_names = re.sub(
        r'\[.*?(HOSPITAL|Hospital|hospital).*?\]( \[\*\*.*?(HOSPITAL|Hospital|hospital).*?\*\*\])*', " HOSPITAL ",
        contents_without_names)
    contents_without_loc = re.sub(
        r'\[.*?(Location|LOCATION|location).*?\]( \[\*\*.*?(Location|LOCATION|location).*?\*\*\])*', " LOCATION ",
        contents_without_h_names)

    contents_without_sb = re.sub(r'\[.*?\]', "N", contents_without_loc)
    transform_abbrs = re.sub(r'(?<=\.\w)\. |(?<=\.\w\w)\. |(?<=\.\w\w\w)\. |(?<=\.\w\w\w\w)\. ', ' ',
                             contents_without_sb)
    transform_abbrs_5 = re.sub(r'(?<=\w)\.(?!\s)', '', transform_abbrs)
    cleanNum = re.sub(r"[\d]+(\-(?=[A-Za-z])|:)+", "N ", transform_abbrs_5)
    cleanNum_2 = re.sub(r"[\d]+[\.\-?,%\d]*", "N", cleanNum)

    transform_abbrs_6 = re.sub(r"(?<=\s\w)\/(?=\w\s)", "", cleanNum_2)
    transform_abbrs_7 = re.sub(r"(?<=\s\w)\/(?=\w\/\w\s)", "", transform_abbrs_6)
    transform_abbrs_8 = re.sub(r"(?<=\s\w\w)\/(?=\w\s)", "", transform_abbrs_7)

    contents_without_names = re.sub(r'_%#name#%_( _%#name#%_)*', "NAME", transform_abbrs_8)
    contents_without_cities = re.sub(r'_%#city#%_( _%#city#%_)*', "LOCATION", contents_without_names)

    contents_without_months = re.sub(r'_%#m.+#%_', "N", contents_without_cities)
    contents_without_days = re.sub(r'_%#d.+#%_', "N", contents_without_months)
    contents_without_extra_spaces = re.sub(r' +', " ", contents_without_days)
    cleanNum_1 = re.sub(r"[\d]+[\-] ", "N", contents_without_extra_spaces)
    cleanNum_2 = re.sub(r"[\d]+[\-][/\.?:,%\d]*", "N ", cleanNum_1)
    cleanNum_3 = re.sub(r"[\d]+[/\.?:,%\d]*(?!>)", "N", cleanNum_2)

    reduced_n = re.sub(r'(?<!\w)N( N)+', 'N', cleanNum_3)
    reduced_n2 = re.sub(r'NN+', 'N', reduced_n)

    remove_tab = re.sub(r'\\t', ' ', reduced_n2)
    remove_amp = re.sub(r'&', ' and ', remove_tab)
    remove_apos = re.sub(r"'", '', remove_amp)
    target_sentence = re.sub(r'[,+\-\.;=:\*@#\?\$_\/%"^\(\)\\\|`\[\]{}]+ *', " ", remove_apos)
    target_sentence = re.sub(r"N\)[ \)]", "", target_sentence)

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    word = target_sentence.translate(translator)
    split_chars_1 = re.sub(r'(?<=\w)(?<!LOCATIO)N', " N", word)
    split_chars_2 = re.sub(r'N(?=\w)(?!AME)', "N ", split_chars_1)
    reduced_n = re.sub(r'(?<=N)( N)+', ' ', split_chars_2)
    reduced_name = re.sub(r'(NAME )+', 'NAME ', reduced_n)
    cleaned_line = ' '.join(reduced_name.split())

    l = cleaned_line.split()
    if len(l) > 10 and len(l) < 150:
        g.write(' '.join(cleaned_line.split()) + '\n')
