import os
import random
import argparse
import datetime

date = datetime.datetime.today().strftime('%Y%m%d')
NUM_ID = 100
NUM_CUIS = 0

def get_files():
    files = open("cuis_identified.txt", 'r').readlines()
    #files = open("some_cuis.txt", 'r').readlines()
    for i in range(len(files)):
        files[i] = files[i][:-1]
    return files

def sample_rs_dataset(opt):
    #src_dir = '/Volumes/terminator/hpf/cuis_rs/'
    #dest_dir = '/Volumes/terminator/hpf/cuis_rs_' + str(opt.ns) + 'Samples_' + date + '/'
    src_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/cuis_rs/'
    dest_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/cuis_rs_' + str(opt.ns) + 'Samples_' + date + '/'

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    file_names = get_files()
    global NUM_CUIS
    NUM_CUIS = len(file_names)

    job_id = int(opt.id)
    chunk = int(NUM_CUIS // NUM_ID)
    start = int(chunk * (job_id - 1))
    end = int(start + chunk)
    if job_id == NUM_ID:
        end = NUM_CUIS
    print("Covering indices: " + str(start) + " to " + str(end))
    for i in range(start, end):
        file = file_names[i]
        all_cui_samples = open(os.path.join(src_dir, file), 'r').readlines()
        random.Random(16).shuffle(all_cui_samples)  # shuffles all_cui_samples array with random seed = 16
        num_samples = min(len(all_cui_samples), int(opt.ns))
        subset_cui_samples = all_cui_samples[:num_samples]

        foutput = file[:-4] + "_" + str(opt.ns) + ".txt"
        fdest = open(os.path.join(dest_dir, foutput), 'w')
        for sample in subset_cui_samples:
            fdest.write(sample)
        fdest.close()

    print("Done writing CUIs: " + str(file_names[start]) + " to " + str(file_names[end-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of cui files to sample from;"
                                                    "e.g.: '1'")
    parser.add_argument('-ns', default=500, help='number of max. samples to get from each cui file to generate'
                                                 'concept model dataset')

    opt = parser.parse_args()
    sample_rs_dataset(opt)
    print("Done sampling CUI files! \U0001F335 \U0001F33A")

if __name__ == "__main__":
    main()