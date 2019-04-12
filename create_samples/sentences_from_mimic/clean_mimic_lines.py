#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:44:18 2018

@author: Marta
"""

import sys, getopt, re


def createSentences(input, window):
    try:

        split_sent = input.split("|")
        abbr_meaning = split_sent[1]

        text = split_sent[2]
        doc = ''

        target_sentence = text.lower()
        remove_newline = re.sub(r'\\n', ' ', target_sentence)
        contents_without_paren = re.sub(r'\(.*?\)', "", remove_newline)
        contents_without_sb = re.sub(r'\[.*?\]', "N", contents_without_paren)
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

        remove_tab = re.sub(r'\\t', ' ', reduced_n)
        remove_amp = re.sub(r'&', ' and ', remove_tab)
        target_sentence = re.sub(r'[,+\-\.;=:\*@#\?\$_\/%"^\(\)\\\|`\[\]{}]+ *', " ", remove_amp)
        target_sentence = re.sub(r"N\)[ \)]", "", target_sentence)

        doc += target_sentence + ' '
        remove_nl = doc.split('\n')
        split_on_sense = remove_nl[0].split(abbr_meaning)
        try:
            left_of_sense = split_on_sense[0].strip().split()
            right_of_sense = split_on_sense[1].strip().split()

            doc_left = ' '.join(left_of_sense)
            doc_whole = ' '.join(left_of_sense + right_of_sense)

            left_window_start = max(0, len(left_of_sense) - window)
            left_window_end = max(0, len(left_of_sense))
            local_sentence_left = left_of_sense[left_window_start:left_window_end]

            right_window_start = 0
            right_window_end = min(window, len(right_of_sense))
            local_sentence_right = right_of_sense[right_window_start:right_window_end]

            output = [str(abbr_meaning), str(doc_whole), str(doc_left), str(' '.join(local_sentence_left)),
                      str(' '.join(local_sentence_right))]

            return output

        except IndexError:
            pass
    except:
        pass



