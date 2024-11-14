#from append_path import *
from pylab import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib import font_manager

from pathlib import PurePath, Path
import numpy as np
import sys

MAIN_PATH = PurePath(Path().cwd())

style_path = (MAIN_PATH / 'config' / 'BA_mplrc2').as_posix()

mpl.style.use(style_path)

# Edit the font, font size, and axes width
#FONT_FAMILY = 'CMU Serif'
FONT_SIZE= 18

#mpl.rcParams['font.family'] = FONT_FAMILY
mpl.rcParams['font.size'] = FONT_SIZE


# symbols definition

s_auc = r'$\mathcal{A}$'
s_target = r'$\tau$'
s_prediction = r'$\hat \tau$'
s_dt = r'$\Delta t$'
s_sigma = r'$\hat \sigma_k$'
s_var_emp = r'$\sigma^2_{\mathrm{emp.}}$'
s_beta_red = r'$\beta_\epsilon$'
s_beta_red_hat = r'$\hat \beta_\epsilon$'
s_beta_hat = r'$\hat \beta$'
s_beta = r'$\beta$'
s_data_points = r'$N$'
s_N_exp =r'$\tilde N$'
s_new = 'NEW'


# Using 


# plt.rcParams.update({
#     "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  # red   with alpha = 30%
#     # "axes.facecolor":    (0.0, 1.0, 0.0, 0.5),  # green with alpha = 50%
#     "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),  # blue  with alpha = 20%
# })

# Setting LaTex Font
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": font_family,
#     "font.sans-serif": []})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
# It's also possible to use the reduced notation by directly setting font.family:

COLORS = cm.get_cmap('tab10', 10)
print('Loading plot config 2', COLORS)
