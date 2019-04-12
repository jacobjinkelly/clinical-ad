import getopt, sys, ast, pickle
from visualize_data import AbbrRep
COORDINATES = []
INPUTFILE = ""
PARAM_MSSG = 'python3 get_samples_at_coordinates.py -p <pickle_file> -c <[[x_val_10, x_val_11, y_val_10, y_val_11]' \
             ',..., [x_val_N0, x_val_N1, y_val_N0, y_val_N1]]>'

def print_samples_at_coordinates():

    pickle_in = open(INPUTFILE, "rb")
    contents = pickle.load(pickle_in)
    pickle_in.close()

    vis_x = contents[0]
    vis_y = contents[1]
    all_X_words = contents[2]

    filename = "SAMPLES_AT_" + INPUTFILE[:-7] + ".txt"
    q = open(filename, 'a')
    results_to_view = []
    batch_num = len(COORDINATES)
    for curr_batch in range(batch_num):
        try:
            x = [COORDINATES[curr_batch][0], COORDINATES[curr_batch][1]]
            y = [COORDINATES[curr_batch][2], COORDINATES[curr_batch][3]]

            t = max(x[0], x[1])
            b = min(x[0], x[1])
            l = min(y[0], y[1])
            r = max(y[0], y[1])
            q.write("COORDINATES: x=" + str([b, t]) + " y=" + str([l, r]) + '\n')
            for i in range(len(vis_x)):
                if vis_x[i] < t and vis_x[i] > b and vis_y[i] < r and vis_y[i] > l:
                    results_to_view.append(i)

            for z in results_to_view:
                #all_X_words[z] = AbbrRep(all_X_words[z])
                q.write(str(all_X_words[z].features_doc) + '\n')
        except:
            print("Coordinates not given in proper format.")
            print(COORDINATES)
            print(PARAM_MSSG)
            sys.exit(2)

def main(argv):
    global COORDINATES, INPUTFILE
    try:
        opts, args = getopt.getopt(argv, "p:c:")
    except getopt.GetoptError:
        print(PARAM_MSSG)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-p":
            INPUTFILE = arg
        elif opt == "-c":
            COORDINATES = ast.literal_eval(arg)

    print_samples_at_coordinates()
    print("Successfully found samples within given coordinates.")

if __name__ == "__main__":
    main(sys.argv[1:])