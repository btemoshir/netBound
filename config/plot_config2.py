#from append_path import *
from pylab import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib import font_manager

#font_dirs = ['C:\\Users\\thanatos\\Documents\\Fonts\\computer-modern']
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

#for font_file in font_files:
#    font_manager.fontManager.addfont(font_file)

#style_path = (MAIN_PATH / 'scripts' / 'config' / 'BA_mplrc').as_posix()

style_path = (MAIN_PATH / 'memoryFunctions' / 'config' / 'BA_mplrc').as_posix()

mpl.style.use(style_path)

# Edit the font, font size, and axes width
FONT_FAMILY = 'CMU Serif'
FONT_SIZE= 18

mpl.rcParams['font.family'] = FONT_FAMILY
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
print('Loading plot config', COLORS)


def calc_axis(N, space_ratio):

    if not isinstance(N, tuple): 
        ax_arr_list = []
        dx =  (1 - space_ratio) / N
        dw = space_ratio/(N-1)

        x = 0 
        for _ in range(N): 

            ax_arr_list.append([x, 0, dx, 1])
            x += dx + 2*dw
    
        return  ax_arr_list

    else : 
        Nw, Nh = N
        srw, srh = space_ratio
        ax_arr_list = []

        dx = (1 - srw) / Nw
        dw = srw / (Nw - 1)

        dy = (1 - srh) / Nh
        dh = srh / (Nh - 1)
        
        x = 0 
        y = 1 - dy - dh 

        for h in range(Nh): 
            for w in range(Nw): 
                ax_arr_list.append([x, y, dx, dy])
                x += dx + 2*dw
            y -= dy + 2*dh 
            x = 0 
        
        return ax_arr_list