import os
import ast
import string
import re
import pickle
import argparse
import datetime

rootdir = './'

word_dict = {}


def clean(word):
    word = word.lower()
    word = re.sub(r'%2[72]', "", word)
    word = re.sub(r'%26', "&", word)
    word = re.sub(r'%\w\w', " ", word)
    word = re.sub(r'[0-9]+', " N ", word)
    reduced_n = re.sub(r'(?<!\w)N( N)+', 'N', word)
    remove_apos = re.sub(r"'", '', reduced_n)
    word = re.sub(r'&', ' and ', remove_apos)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    word = word.translate(translator)
    word = ' '.join(word.split())
    return word

def consolidate_abbr_files(opt):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file[:10] == '200kabbrs_':
                path = os.path.join(subdir, file)
                f = open(path, 'r')
                for line in f:
                    content = line.split(":::")
                    senses = ast.literal_eval(content[-1][:-1])
                    abbr = content[0]
                    cleaned_abbr = clean(abbr)
                    for word in senses:
                        cleaned_word = clean(word)
                        if opt.by_abbr:
                            try:
                                word_dict[cleaned_abbr].add(cleaned_word)
                            except KeyError:
                                word_dict[cleaned_abbr] = set()
                                word_dict[cleaned_abbr].add(cleaned_word)
                        if opt.by_length:
                            word_length = len(cleaned_word.split())
                            if word_length > 1:
                                try:
                                    word_dict[word_length].add(cleaned_word)
                                except KeyError:
                                    word_dict[word_length] = set()
                                    word_dict[word_length].add(cleaned_word)

    date = datetime.datetime.today().strftime('%Y%m%d')
    if opt.outputfile:
        output = opt.outputfile
    else:
        output = "allacronyms_consolidated_" + str(date) + ".pickle"

    print("Outputfile: " + output)
    pickle_out = open(output, "wb")
    pickle.dump(word_dict, pickle_out)
    pickle_out.close()

    if opt.abbr_list:
        file = "abbr_list_" + str(date) + ".txt"
        g = open(file, 'w')
        for key in word_dict:
            g.write(key + '\n')
        g.close()
        print("Abbr list outputfile: " + file)


    print("Successfully consolidated abbreviations \U0001F60E \U0001F4A5")

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-by_length', action="store_true", help="sort abbr expansions by length")
    parser.add_argument('-by_abbr', action="store_true", help="sort abbrs expansions by abbr")
    parser.add_argument('-abbr_list', action="store_true", help="sort list of abbrs")
    parser.add_argument('-outputfile', help=".pickle file to store dictionary with key as abbr or length")

    opt = parser.parse_args()
    consolidate_abbr_files(opt)

if __name__ == "__main__":
   main()

