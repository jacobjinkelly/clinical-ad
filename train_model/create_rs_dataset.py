import sys, ast, re, getopt
import find_lines_w_targetword_mimic as find_tw
import pexpect
import argparse
import pickle
import os


FASTTEXT_PATH = "./fastText/fasttext"
MODEL_PATH = "./word_embeddings_joined.bin"

src_dir = "/Users/Marta/80k_abbreviations/allacronyms"
n = open(os.path.join(src_dir, "allacronyms_name2meta_20190402_NEW.pickle"), 'rb')
name2meta = pickle.load(n)
n.close()
all_abbrs = list(name2meta.keys())

# o = open(os.path.join(src_dir,"allacronyms_meta2name_20190318.pickle"), 'rb')
o = open(os.path.join(src_dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
meta2name = pickle.load(o)
o.close()

class FT():
    def __init__(self, num_neighbours):
        self.nn_process = pexpect.spawn('%s nn %s %d' % (FASTTEXT_PATH, MODEL_PATH, num_neighbours))
        self.nn_process.expect('Query word?')  # Flush the first prompt out.
    def get_nn(self, word):
        word = word.encode()
        self.nn_process.sendline(word)
        self.nn_process.expect('Query word?')
        output = self.nn_process.before
        return [word.decode()] + [line.strip().split()[0] for line in output.decode().strip().split('\n')[1:]]


def format_senses(opt):
    # f = open(opt.inputfile, 'r')
    # abbr_dict = {}
    # for line in f:
    #     content = line.split(":::")
    #     abbr = content[0].lower()
    #     possible_senses = ast.literal_eval(content[1])
    #     abbr_dict[abbr] = possible_senses


    abbr_dict = ["ivf","dm", "le","ra","pcp","op","dt","dc","pa","pda","rt","sma","ac","pe",
                 "otc","im","pac","pr","asa","ir","sbp","cea","ca","er","bal","avr,","cvp","av"]
    for abbr in abbr_dict:

        # if opt.find_a:
        #     find_tw.create_sentences(abbr, abbr, int(opt.window), 0, opt.corpus)
        senses = list(name2meta[abbr].keys())
        senses_joined = []
        for i in senses:
            senses_joined.append("_".join(i.split()))
        senses_seen = []
        for sense in senses:

            # sense = ' '.join(sense.split("_"))
            # sense = re.sub(r'\-', ' ', sense)
            # sense = sense.lower()
            # if sense in senses_seen:
            #     continue
            # else:
            #     senses_seen.append(sense)
            # if opt.find_custom != "":
            #     find_tw.create_sentences(sense, abbr, int(opt.window), -1, opt.corpus, custom_outputfile=opt.find_custom)
            # else:
            #     find_tw.create_sentences(sense, abbr, int(opt.window), 1, opt.corpus)
            if opt.find_n:
                model = FT(opt.n)
                joined_sense = "_".join(sense.split())
                print(joined_sense)
                nn = model.get_nn(joined_sense)
                for i in range(1, len(nn)):
                    if str(nn[i]) not in senses_joined and str(nn[i]) not in all_abbrs:
                        find_tw.create_sentences(str(nn[i]), abbr, opt.window, 2, opt.corpus, original_word=sense)


def main(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-inputfile', required=True, help="format: ABBR:::[list_of_expansions]")
    parser.add_argument('-find_n', action="store_true", help="find neighbouring words")
    parser.add_argument('-find_a', action="store_true", help="find sentences containing target abbreviation")
    parser.add_argument('-n', default=3, help="number of neighbouring words to find")
    parser.add_argument('-window', required=True, help="number of surrounding words to take")
    parser.add_argument('-corpus', required=True, help="file to find sentences in (uncleaned MIMIC)")
    parser.add_argument('-find_custom', default="", help="find list of random words; name of outputfile")

    opt = parser.parse_args()
    # print("Input file name is:", opt.inputfile)
    format_senses(opt)


if __name__ == "__main__":
    main(sys.argv[1:])
