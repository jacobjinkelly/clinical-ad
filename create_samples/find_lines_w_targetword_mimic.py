#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 21:22:18 2018

@author: Marta
"""
import clean_mimic_lines as clean_text


def create_sentences(target_word, abbr, window, category, corpus, original_word="", custom_outputfile=""):
    inputfile = corpus
    corpus_name = inputfile.split("_")[0]

    # abbreviated word
    if category == 0:
        outputfile = abbr + "_mimic_abbr" + ".txt"
    # expanded words
    elif category == 1:
        outputfile = abbr + "_mimic_rs.txt"
    # expanded words similar to sense
    elif category == 2:
        target_word = ' '.join(target_word.split("_"))
        outputfile = abbr + "_mimic_rs_sim.txt"
    else:
        outputfile = custom_outputfile

    print("Input file name is: " + inputfile)
    print("Output file name is: " + outputfile + ". Current word being searched for is: " + target_word)

    f = open(inputfile, 'r')
    #g = open("/Users/Marta/80k_abbreviations/create_samples/i2b2_sentences_20190327/" +outputfile, 'a')
    g = open(outputfile, 'a')
    try:
        for line in f:
            line = line.lower()
            present = False
            if len(line.split(" " + target_word.lower() + " ")) > 1:
                present = True
            elif len(line) >= len(target_word) + 2:
                # target word at beginning of sentence
                if line[:len(target_word) + 1].lower() == target_word.lower() + " ":
                    present = True
                # target word at end of sentence
                elif line[len(line) - (len(target_word) + 2):-1].lower() == " " + target_word.lower():
                    present = True
            if present:
                line_len = len(line.split())
                if line_len > 10 and line_len < 150:
                    # if category == 2 and original_word != "":
                    #     target_word = original_word
                    text_to_clean = abbr + "|" + target_word.lower() + "|" + line + '\n'
                    cleaned_text = clean_text.createSentences(input=text_to_clean, window=window,
                                                              original_word=original_word, category=category)
                    g.write(str(cleaned_text) + '\n')

    finally:
        f.close()
        g.close()
