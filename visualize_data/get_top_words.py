import getopt, sys, ast
import operator
N = 10

def get_top_words(inputfile):
    f = open(inputfile, 'r')
    coordinates_set = {}
    curr_coordinate_set = ""
    for line in f:
        content = line
        check_coordinates = line.split(":")
        if check_coordinates[0] == "COORDINATES":
            curr_coordinate_set = content[:-1]
            coordinates_set[curr_coordinate_set] = {}
        else:
            content = ast.literal_eval(content) # create list
            for word in content:
                try:
                    coordinates_set[curr_coordinate_set][word] += 1
                except KeyError:
                    coordinates_set[curr_coordinate_set][word] = 1

    for key in coordinates_set:
        sorted_d = sorted(coordinates_set[key].items(), key=operator.itemgetter(1), reverse=True)
        print(key)
        print(sorted_d[:N])

def main(argv):
    inputfile = ''
    global N
    try:
        opts, args = getopt.getopt(argv, "i:n:")
    except getopt.GetoptError:
        print('python3 get_top_words.py -i <inputfile> -n <top_n_words>')
        print('Inputfile has format: Line 1: COORDINATES, Line 2: [list_of_words], etc.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            inputfile = arg
        elif opt == "-n":
            N = int(arg)


    print("Input file name is:", inputfile)

    get_top_words(inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])