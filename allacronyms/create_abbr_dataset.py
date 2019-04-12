import pickle
import datetime
import os
import argparse



date = datetime.datetime.today().strftime('%Y%m%d')
NUM_ID = 100
NUM_CUIS = 0

def get_abbrs():
    m = open("cleaned_allacronyms_dict_20190318.pickle", 'rb')
    abbr_dict = pickle.load(m)
    m.close()

    return list(abbr_dict.keys())

def create_abbr_datasets(opt):

    n = open("allacronyms_name2meta_20190318.pickle", 'rb')
    name2meta = pickle.load(n)
    n.close()

    abbrs_to_get = get_abbrs()
    global NUM_CUIS
    NUM_CUIS = len(abbrs_to_get)

    job_id = int(opt.id)
    chunk = int(NUM_CUIS // NUM_ID)
    start = int(chunk * (job_id - 1))
    end = int(start + chunk)
    if job_id == NUM_ID:
        end = NUM_CUIS
    print("Covering indices: " + str(start) + " to " + str(end))

    for i in range(start, end):
        abbr = abbrs_to_get[i]
        dest_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/abbr_dataset/'
        dest_fname = abbr + ".txt"
        foutput = open(os.path.join(dest_dir, dest_fname), 'w')

        abbr_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha_cleaned/" + str(abbr[0]) + '/'
        try:
            abbr_file = open(os.path.join(abbr_dir, abbr), 'r').readlines()
            for item in abbr_file:
                foutput.write(item)
        except:
            print("*********************************")
            print(abbr + " file does not exist!")
            print("*********************************")

        expansions = list(name2meta[abbr]) # list of cuis
        for expansion in expansions:
            exp_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha_cleaned/" + str(expansion[0]) + '/'
            try:
                exp_file = open(os.path.join(exp_dir, expansion), 'r').readlines()
                for item in exp_file:
                    foutput.write(item)
            except:
                print(expansion + " file does not exist!")

        foutput.close()

    print("Done writing CUIs: " + str(abbrs_to_get[start]) + " to " + str(abbrs_to_get[end-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of abbreviation files to get;"
                                                    "e.g.: '1'")

    opt = parser.parse_args()
    create_abbr_datasets(opt)
    print("Done making abbr dataset! \U0001F335 \U0001F33A")

if __name__ == "__main__":
    main()