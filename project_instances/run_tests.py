import os
import sys

def write_scores_output(description):
  with open("score_tracker.txt", "a") as f:
    f.write('\n')
    f.write(description + '\n')

if __name__ == '__main__':
    file_name = "problem"

    args = sys.argv

    start = 1
    end = 22

    if len(args) != 2:
        print("Not tracking scores")
        print()
    else:
        description = args[1]
        write_scores_output(description)
    # if len(args) > 1:
    #     start = int(args[1])
    #     end = int(args[2]) + 1
    #
    # print("Files from range {} to {}".format(start, end - 1))

    for i in range(start, end):
        input_file = file_name + str(i)
        output_file = file_name + str(i) + ".out"
        print('final_proj_solver.py ' + input_file + " " + output_file)
        os.system('final_proj_solver.py ' + input_file + " " + output_file)
