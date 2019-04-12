import argparse
import pickle
import os
import sys

alpha_map = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
             14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
             26: 'z', 27: 'num'}

NUM_ID = 1000
NUM_CUIS = 0

#src_dir = '/Volumes/terminator/hpf/rs_sorted_alpha_cleaned/'
#dest_dir = '/Volumes/terminator/hpf/cuis_rs/'

src_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha_cleaned/'
dest_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/cuis_rs/'

def get_abbr_list():
    abbr_list = []

    for i in range(27):
        single_char = alpha_map[i+1]
        if single_char == "num":
            single_char = "N"
        abbr_list.append(single_char)

    f = open("abbr_list_20190310.txt", 'r')
    for line in f:
        abbr = line[:-1]
        abbr_list.append(abbr)
        if abbr[-1] == 's':
            abbr_list.append(abbr[:-1])
        else:
            abbr_list.append(abbr + 's')
    f.close()
    return abbr_list

def get_cuis():
    f = open("list_of_cuis_20190310.txt", 'r').readlines()
    cui_list = f[0].split(",")
    return cui_list

def organize_files_by_cui(opt):
    abbr_list = get_abbr_list()
    cui_list = get_cuis()

    #umls_id2name_src = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), 'umls_id2name_20190310.pickle')
    umls_id2name_src = "umls_id2name_20190310.pickle"
    pickle_in = open(umls_id2name_src, 'rb')
    id2name = pickle.load(pickle_in)
    pickle_in.close()

    global NUM_CUIS
    NUM_CUIS = len(cui_list)

    job_id = int(opt.id)
    chunk = int(NUM_CUIS // NUM_ID)
    start = int(chunk * (job_id - 1))
    end = int(start + chunk)
    if job_id == NUM_ID:
        end = NUM_CUIS

    print("Covering indices: " + str(start) + " to " + str(end))
    for i in range(start, end):
        cui = cui_list[i]
        cui_terms = id2name[cui]
        filenames = set()
        for term in cui_terms:
            if term in abbr_list:
                continue
            try:
                char_dir = term[0] + '/'
                if "N" in term:
                    new_term = term.replace("N", "")
                    new_term = ' '.join(new_term.split())
                    if new_term in abbr_list:
                        continue
                if term[0] == "N":
                    char_dir = "num/"
            except:
                continue
            filenames.add(src_dir + char_dir + term)

        existing_filenames = set()
        for fname in filenames:
            if os.path.isfile(fname):
                existing_filenames.add(fname)
        if len(existing_filenames) == 0:
            continue

        outfname = dest_dir + str(cui) + ".txt"

        '''
        f = open(outfname, 'w')
        for fname in existing_filenames:
            g = open(fname, 'r')
            for line in g:
                f.write(line)
            g.close()
        f.close()
        '''
        with open(outfname, 'w') as outfile:
            for fname in existing_filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)



    print("Done writing CUIs: " + str(cui_list[start]) + " to " + str(cui_list[end]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of cui terms to create files for;"
                                                    "e.g.: '1'")

    opt = parser.parse_args()
    organize_files_by_cui(opt)
    print("Done writing rs samples to cui files \U0001F4AA \U00002600")

if __name__ == "__main__":
    main()
