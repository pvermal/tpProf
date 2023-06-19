import os
import sys

sys.stdout = open(os.devnull , 'w')

script_name = "run_models.py"
output_prefix = "out"
n_iter = 100

for i in range(n_iter):
    os.system("python3 "+ script_name + " " + str(i) )
