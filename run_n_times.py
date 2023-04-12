import os
import sys

sys.stdout = open(os.devnull , 'w')

script_name = "run_models.py"
output_prefix = "out"
n_iter = 100

for i in range(n_iter):
    #output_file = output_prefix + '_' + str(i) + '.txt'
    #sys.stdout = open(output_file, 'w')
    #subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)
    os.system("python3 "+ script_name + " " + str(i) )
