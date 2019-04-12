from bs4 import BeautifulSoup
import requests
import time
import pickle
import getopt, sys
TIMEOUT = 1
TIMEOUT_2 = 2

def get_page(next_page):
    r = None
    should_repeat = True
    while r is None or should_repeat:
        try:
            r = requests.get(next_page, timeout=TIMEOUT)
            print(r)
            should_repeat = False
            r.raise_for_status()
        except requests.exceptions.Timeout as timeout_err:
            print(timeout_err)
            time.sleep(TIMEOUT)
            should_repeat = True
        except requests.exceptions.HTTPError as err:
            print(err)
            time.sleep(TIMEOUT)
            should_repeat = True
        except requests.exceptions.ConnectionError as connection_err:
            print(connection_err)
            time.sleep(TIMEOUT_2)
            should_repeat = True

        if r is None:
            print("NO TEXT RETURNED from current get request")
            time.sleep(TIMEOUT)
            should_repeat = True

    return r


def get_alphabet_to_cover(version):
    # Create list of double alphabet chars (e.g. AA, AB,...,ZZ)
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    double_alphabet = []
    for char_1 in alphabet:
        for char_2 in alphabet:
            double_alphabet.append(char_1+char_2)

    # load characters of the alphabet already covered
    try:
        pickle_in = open("alphabet_covered_" + str(version) + ".pickle", 'rb')
        alph_covered = pickle.load(pickle_in)
        pickle_in.close()
    except FileNotFoundError:
        alph_covered = []

    if version == 0:
        double_alphabet_version = double_alphabet[:338]
    elif version == 1:
        double_alphabet_version = double_alphabet[338:390]
    elif version == 2:
        double_alphabet_version = double_alphabet[442:546]
    else:
        double_alphabet_version = double_alphabet[390:442]
    # subtract seen characters from list of characters to see
    unseen_alphabet = (list(set(double_alphabet_version) - set(alph_covered)))
    unseen_alphabet.sort()

    return unseen_alphabet, alph_covered


def scrape_pages(unseen_alphabet, alph_covered, version):

    for i in unseen_alphabet:
        print("NEW CHARACTER CLASS: " + i)
        abbr_sense = []

        possible_abbrs = []
        next_page = "https://www.allacronyms.com/_medical/aa-index-alpha/" + i
        print(next_page)
        r = get_page(next_page)
        soup = BeautifulSoup(r.content, 'html.parser')
        possible_page_nums = []
        for a_tag in soup.find_all('a', href=True):
            url_split = str(a_tag['href']).split('/_medical/')
            if len(url_split) > 1 and url_split[1][0:2] == i:
                possible_abbrs.append(str(a_tag['href']))
            elif len(url_split) > 1 and url_split[1].split('/')[-1].isdigit():
                possible_page_nums.append(int(url_split[1].split('/')[-1]))

        if len(possible_page_nums) > 0:
            num_pages = max(possible_page_nums)
            print("NUM PAGES: " + str(num_pages))
            for j in range(2,int(num_pages)+1):
                next_page = "https://www.allacronyms.com/_medical/aa-index-alpha/" + i + '/' + str(j)
                print(next_page)
                r = get_page(next_page)
                soup = BeautifulSoup(r.content, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    url_split = str(a_tag['href']).split('/_medical/')
                    if len(url_split) > 1 and url_split[1][0:2] == i:
                        possible_abbrs.append(str(a_tag['href']))

        end = False
        counter = -1
        # print(possible_abbrs)
        # possible_abbrs = [possible_abbrs[0]]
        for z in possible_abbrs:
            print("NEW ABBR:")
            counter += 1

            next_page = "https://www.allacronyms.com" + z
            print(next_page)
            r = get_page(next_page)
            soup = BeautifulSoup(r.content, 'html.parser')

            abbr_sense.append(soup)

            possible_mini_page_nums = []

            # get number of pages
            for a_tag in soup.find_all('a', href=True):
                url_split = str(a_tag['href']).split('/_medical/')
                if len(url_split) > 1 and url_split[1].split('/')[-1].isdigit():
                    possible_mini_page_nums.append(int(url_split[1].split('/')[-1]))

            if len(possible_mini_page_nums) > 0:
                num_mini_pages = max(possible_mini_page_nums)
                for v in range(2, int(num_mini_pages) + 1):
                    next_page = "https://www.allacronyms.com" + z + '/' + str(v)
                    print(next_page)
                    r = get_page(next_page)
                    soup = BeautifulSoup(r.content, 'html.parser')
                    abbr_sense.append(soup)

            if counter == len(possible_abbrs)-1:
                end = True

            if end:
                print("FINISHED CHARACTER CLASS! " + i)
                letter = i[0]
                f = open("abbrs_master_doc_" + letter + ".txt", 'a')
                f.write("------------------" + str(i) + "------------------" + '\n' + str(abbr_sense) + '\n')
                f.close()
                alph_covered.append(str(i))
                pickle_out = open("alphabet_covered_" + str(version) + ".pickle", 'wb')
                pickle.dump(alph_covered, pickle_out)
                pickle_out.close()


def main(argv):
    version = 0
    try:
        opts, args = getopt.getopt(argv, "v:")
    except getopt.GetoptError:
        print("python3 scrape_allacronyms_pages.py -v <version>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-v":
            version = int(arg)

    unseen_alphabet, alph_covered = get_alphabet_to_cover(version)
    # temp_double_alphabet = [unseen_alphabet[0]]  # AA
    scrape_pages(unseen_alphabet, alph_covered, version)

if __name__ == "__main__":
    main(sys.argv[1:])