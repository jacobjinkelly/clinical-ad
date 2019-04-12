from bs4 import BeautifulSoup
import urllib.request as urllib2
import random
import os
import sys
import requests
import time
alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
double_alphabet = []
for char_1 in alphabet:
    for char_2 in alphabet:
        double_alphabet.append(char_1+char_2)

temp_double_alphabet = [double_alphabet[0]] # AA
abbr_sense = {} # dictionary that stores abbrs-senses

seen_alphabet = []
try:
    l = open("alphabet_covered.txt", 'r')
    for line in l:
        seen_alphabet.append(line[:-1])
    l.close()
except FileNotFoundError:
    pass
print(seen_alphabet)
l = open("alphabet_covered.txt", 'a')

for i in double_alphabet:
    if i not in seen_alphabet:

        #-----abbreviation pages we have already visited-----#
        seen_abbrs = []
        try:
            g = open("trackedabbrs_" + i +".txt", 'r')
            for line in g:
                seen_abbrs.append(line[:-1])
            g.close()
        except FileNotFoundError:
            pass
        #----------------------------------------------------#

        #----abbreviation category pages we have visited-----#
        seen_pages = []
        try:
            q = open("trackedpages_" + i +".txt", 'r')
            for line in q:
                seen_pages.append(line[:-1])
            q.close()
        except FileNotFoundError:
            pass
        #----------------------------------------------------#

        f = open("200kabbrs_" + i +".txt", 'a')
        g = open("trackedabbrs_" + i +".txt", 'a')
        q = open("trackedpages_" + i +".txt", 'a')
        m = open("trackedabbrcat_" + i +".txt", 'a')


        # time.sleep(120)
        next_page = "https://www.allacronyms.com/_medical/aa-index-alpha/" + i
        r = None
        should_repeat = False
        try:
            r = requests.get(next_page, timeout=10)
        except:
            while r is None or should_repeat:
                try:
                    r = requests.get(next_page, timeout=10)
                    should_repeat = False
                except:
                    should_repeat = True

        #if str(r) != "<Response [200]>":
            #sys.exit(1)
        response = str(r).split()
        response_number = response[1]
        print(response_number[1])
        while response_number[1] != str(2):
            time.sleep(10)
            r = requests.get(next_page)
            response = str(r).split()
            response_number = response[1]
        soup = BeautifulSoup(r.content, 'html.parser')
        print(soup)
        #possible_abbrs = []
        possible_page_nums = []
        for a_tag in soup.find_all('a', href=True):
            url_split = str(a_tag['href']).split('/_medical/')
            if len(url_split) > 1 and url_split[1][0:2] == i:
                #possible_abbrs.append(a_tag['href'])
                if next_page not in seen_pages:
                    m.write(str(a_tag['href']) + '\n')
            elif len(url_split) > 1 and url_split[1].split('/')[-1].isdigit():
                possible_page_nums.append(int(url_split[1].split('/')[-1]))
        print(possible_page_nums)
        if next_page not in seen_pages:
            q.write(str(next_page) + '\n')
        if len(possible_page_nums) > 0:
            num_pages = max(possible_page_nums)
            print(num_pages)
            for j in range(2,int(num_pages)+1):
                next_page  = "https://www.allacronyms.com/_medical/aa-index-alpha/" + i + '/' + str(j)
                if next_page not in seen_pages:
                    r = None
                    should_repeat = False
                    try:
                        r = requests.get(next_page, timeout=10)
                    except:
                        while r is None or should_repeat:
                            try:
                                r = requests.get(next_page, timeout=10)
                                should_repeat = False
                            except:
                                should_repeat = True
                    print(r)
                    response = str(r).split()
                    response_number = response[1]
                    while response_number[1] != str(2):
                        time.sleep(10)
                        r = requests.get(next_page)
                        response = str(r).split()
                        response_number = response[1]
                    #if str(r) == "<Response [200]>":
                    soup = BeautifulSoup(r.content, 'html.parser')
                    for a_tag in soup.find_all('a', href=True):
                        url_split = str(a_tag['href']).split('/_medical/')
                        if len(url_split) > 1 and url_split[1][0:2] == i:
                            #possible_abbrs.append(a_tag['href'])
                            m.write(str(a_tag['href']) + '\n')
                    q.write(str(next_page) + '\n')
                    '''
                    else:
                        q.close()
                        f.close()
                        g.close()
                        m.close()
                        l.close()
                        sys.exit(1)
                    '''
        # ------abbreviation categories we have visited-------#
        possible_abbrs = []
        m = open("trackedabbrcat_" + i +".txt", 'r')
        for line in m:
            possible_abbrs.append(line[:-1])
        m.close()
        # ----------------------------------------------------#

        end = False
        counter = -1
        for z in possible_abbrs:
            counter += 1
            print(z)
            abbr_cat = z.split('/')[-1]
            if abbr_cat not in seen_abbrs:
                #if counter%5 == 0:
                    #time.sleep(int(random.random()*5 + 1))
                next_page = "https://www.allacronyms.com" + z
                print(next_page)
                r = None
                should_repeat = False
                try:
                    r = requests.get(next_page, timeout=10)
                except:
                    while r is None or should_repeat:
                        try:
                            r = requests.get(next_page, timeout=10)
                            should_repeat = False
                        except:
                            should_repeat = True

                #if str(r) == "<Response [200]>":
                response = str(r).split()
                response_number = response[1]
                while response_number[1] != str(2):
                    time.sleep(10)
                    r = requests.get(next_page)
                    response = str(r).split()
                    response_number = response[1]

                soup = BeautifulSoup(r.content, 'html.parser')

                abbreviations_and_expansions = []
                possible_mini_page_nums = []

                #get number of pages
                for a_tag in soup.find_all('a', href=True):
                    url_split = str(a_tag['href']).split('/_medical/')
                    if len(url_split) > 1 and url_split[1].split('/')[-1].isdigit():
                        possible_mini_page_nums.append(int(url_split[1].split('/')[-1]))

                #GET ALL WORDS ON CURRENT PAGE
                # a_tag = soup.find_all(class_='pairAbb')

                a_tag = soup.find_all(class_='pairAbb')
                for y in a_tag:
                    s1 = str(y).split('/')

                    # s1 = str(y).split('title="')[1]
                    # s2 = s1.split('">')[0].split(" stands for ")
                    # abbr = s2[0]
                    key = z.split('/')[-1]
                    if len(s1) > 3 and s1[1] == '_medical' and s1[2].lower() == key.lower():
                        sense = s1[3].split('" title=')[0]
                    #if abbr.lower() == key.lower():
                        try:
                            abbr_sense[key].append(sense)
                        except KeyError:
                            abbr_sense[key] = [sense]
                seen_all_pages = False
                r_tag = soup.find_all(id='related-abbreviations')
                if len(r_tag) > 0:
                    possible_mini_page_nums = []
                    seen_all_pages = True
                if len(possible_mini_page_nums) > 0:
                    num_mini_pages = max(possible_mini_page_nums)
                    for v in range(2, int(num_mini_pages) + 1):
                        time.sleep(0.5)
                        next_page = "https://www.allacronyms.com" + z + '/' + str(v)
                        r = None
                        should_repeat = False
                        try:
                            r = requests.get(next_page, timeout=10)
                        except:
                            while r is None or should_repeat:
                                try:
                                    r = requests.get(next_page, timeout=10)
                                    should_repeat = False
                                except:
                                    should_repeat = True
                        #if str(r) != "<Response [200]>":
                            #sys.exit(1)
                        response = str(r).split()
                        response_number = response[1]
                        while response_number[1] != str(2):
                            time.sleep(10)
                            r = requests.get(next_page)
                            response = str(r).split()
                            response_number = response[1]
                        if v == int(num_mini_pages):
                            seen_all_pages = True
                        soup = BeautifulSoup(r.content, 'html.parser')
                        a_tag = soup.find_all(class_='pairAbb')
                        for y in a_tag:
                            s1 = str(y).split('/')
                            # s2 = s1.split('">')[0].split(" stands for ")
                            # abbr = s2[0]
                            key = z.split('/')[-1]
                            if len(s1) > 3 and s1[1] == '_medical' and s1[2].lower() == key.lower():
                                sense = s1[3].split('" title=')[0]
                            #if abbr.lower() == key.lower():
                                try:
                                    abbr_sense[key].append(sense)
                                except KeyError:
                                    abbr_sense[key] = [sense]
                        r_tag = soup.find_all(id='related-abbreviations')
                        if len(r_tag) > 0:
                            v = int(num_mini_pages) -1

                if seen_all_pages == True:
                    g.write(abbr_cat + '\n')
                    f.write(abbr_cat + ":::" + str(abbr_sense[abbr_cat]) + '\n')
                '''
                else:
                    f.close()
                    g.close()
                    m.close()
                    l.close()
                    q.close()
                    print("STOPPED HERE:")
                    print(z)
                    sys.exit(1)
                '''
            if counter == len(possible_abbrs)-1:
                end = True


        if end:
            l.write(str(i) + '\n')
            f.close()
            g.close()
            m.close()

            q.close()
            #os.remove("trackedabbrs_" + i +".txt")
            #os.remove("trackedpages_" + i + ".txt")
            #os.remove("trackedabbrcat_" + i + ".txt")


