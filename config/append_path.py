from pathlib import PurePath, Path
import numpy as np
import sys


main_dir = 'bachelor_rep'
script_dir = 'scripts'
p = PurePath(Path().cwd())
parts = np.array([str(par) for par in p.parts])
idx = np.where(parts[::-1] == script_dir)[0][0] 

idx_main = np.where(parts[::-1] == main_dir)[0][0]
path_dir_script = p.parents[idx - 1]
sys.path.insert(0, str(path_dir_script))
MAIN_PATH = p.parents[idx_main - 1]
