import os

if __name__ == '__main__':
    file_name = "problem"

    for i in range(1, 22):
        input_file = file_name + str(i) + ".in"
        output_file = file_name + str(i) + ".out"
        print('sample_solver_constraints.py ' + input_file + " " + output_file)
        os.system('sample_solver_constraints.py ' + input_file + " " + output_file)
