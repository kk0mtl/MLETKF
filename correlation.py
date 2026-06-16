import numpy as np
import os
from scipy import signal 
from scipy.fftpack import rfft, irfft, fft, ifft
from scipy import interpolate
from collections import OrderedDict
import string
from plot import *

"""
Reference
1. matplotlib (https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py)
2. python interpolation (https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)
"""

def calc_corr(x, x_intp, x_axis_shift, y, intp):
    N = len(x)
    
    if x_axis_shift:
        y[0:N-2] = y[1:N-1]

    if not intp:
        return y
        
    f_intp = interpolate.interp1d(x, y, kind='cubic')
    y_intp = f_intp(x_intp)        

    return y_intp

    
def plot_corr(enkf_method, h_type, nem, nobs, corr):
    pass   


def plot_corr_paper(collection, data_dir, plot_show, plot_outdir, plot_save):
    fname = os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_6_nEnKF_1_nem_81_nobs_40_obs_loc_dist_0.95_h_type_1_y")

    N = len(np.loadtxt(fname)) if os.path.exists(fname) else 40
    
    x = np.arange(N)
    x_intp = np.arange(0, N+0.1-1, 0.1)
        
    x_axis_shift = True         # 1 grid point shift for x axis 
    intp = False                # interpolation for y value

    x_axs = x_intp.copy() if intp else x.copy()

    if x_axis_shift:
        x_axs += 1    

    import matplotlib
    import matplotlib.pyplot as plt

    set_plot_font(plt,0)

    #-------------------------------------------------------------------------------------------------------------------------------------
    # 1. collected figure 
    #-------------------------------------------------------------------------------------------------------------------------------------         
    if collection:   
        obs_pos_dst = 1             # 0: one observation over each state location, 
                                    # 1: Gaussian distribution centered on variable 20 with a standard deviation of 1/5 the domain length
                                    # 2: uniform distribution over the domain length 
        nobs_list = [40, 200, 200]
        alpb_list = list(string.ascii_lowercase)

        #---------------------------------------------------------------------------------------------------------------------------------
        # 1) read data 
        #---------------------------------------------------------------------------------------------------------------------------------
        i = 0
        
        # PF (truth)
        trth_intp_list =[]

        for h_type in [0,1,2]:
            for nem in [10000]:
                y = np.loadtxt(os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_9_nEnKF_1_nem_{}_nobs_{}_obs_loc_dist_0.95_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], h_type)))
                trth_intp_list.append({"label": "Optimal least squares", "corr": calc_corr(x, x_intp, x_axis_shift, y, intp)})
                                       
        # ETKF (global)
        etkf_intp_list =[]

        for h_type in [0,1,2]:
            for nem in [10,100]: 
                y = np.loadtxt(os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_2_nEnKF_1_nem_{}_nobs_{}_obs_loc_dist_3.7_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], h_type)))
                etkf_intp_list.append({"text": "{}) {}".format(alpb_list[i], h_mathexp_str(h_type)), "label_corr": "ETKF (${N}_{e}$=%d)"%(nem), \
                                       "corr": calc_corr(x, x_intp, x_axis_shift, y, intp)})     
            i += 1

        # LETKF (local)
        letkf_intp_list =[]
        obs_loc_dists = {0:{3:1.1,10:3.6}, 1:{3:1.1,10:3.5}, 2:{3:1.4,10:2.8}}
        
        for h_type in [0,1,2]:
            #for nem in [3,10,100]:
            for nem in [3,10]: 
                cy = np.loadtxt(os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_3_nEnKF_1_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], obs_loc_dists[h_type][nem], h_type)))
                ry = np.loadtxt(os.path.join(data_dir,"[obs_rloc_x20]enkf_method_3_nEnKF_1_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], obs_loc_dists[h_type][nem], h_type)))
                letkf_intp_list.append({"text": "{}) {}, localized".format(alpb_list[i], h_mathexp_str(h_type)), "label_corr": "LETKF (${N}_{e}$=%d)"%(nem), "label_rloc": "GC (LETKF with ${N}_{e}$=%d)"%(nem),\
                                         "corr": calc_corr(x, x_intp, x_axis_shift, cy, intp), "rloc": calc_corr(x, x_intp, x_axis_shift, ry, intp)})     
            i += 1

        # UTKF (global)
        utkf_intp_list =[]

        for h_type in [0,1,2]:
            for nem in [81]: 
                y = np.loadtxt(os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_6_nEnKF_1_nem_{}_nobs_{}_obs_loc_dist_0.95_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], h_type)))
                #utkf_intp_list.append({"text": "{}) {}".format(alpb_list[i], h_list[h_type]), "label": "UTKF (${N}_{e}$=%d) or SPKF (${N}_{e}$ = 121)"%(nem), 
                utkf_intp_list.append({"text": "{}) {}".format(alpb_list[i], h_mathexp_str(h_type)), "label_corr": "UTKF (${N}_{e}$=%d)"%(nem), 
                                       "corr": calc_corr(x, x_intp, x_axis_shift, y, intp)})     
            i += 1    

        # LUTKF (local)
        lutkf_intp_list =[]

        obs_loc_dist = {0:{100:0.95,200:1},1:{100:1.1,200:1},2:{100:1.8,200:1.2}}
        for h_type in [0,1,2]:
            for nem in [3]: 
                cy = np.loadtxt(os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_7_nEnKF_30_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], obs_loc_dist[h_type][nobs_list[obs_pos_dst]], h_type)))
                ry = np.loadtxt(os.path.join(data_dir,"[obs_rloc_x20]enkf_method_7_nEnKF_30_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y".format(nem, nobs_list[obs_pos_dst], obs_loc_dist[h_type][nobs_list[obs_pos_dst]], h_type)))
                lutkf_intp_list.append({"text": "{}) {}, localized".format(alpb_list[i], h_mathexp_str(h_type)), "label_corr": "LUTKF (${N}_{e}$=%d)"%(nem), "label_rloc": "GC (LUTKF with ${N}_{e}$=%d)"%(nem),\
                                        "corr": calc_corr(x, x_intp, x_axis_shift, cy, intp), "rloc": calc_corr(x, x_intp, x_axis_shift, ry, intp)})  
            i += 1
            
        #---------------------------------------------------------------------------------------------------------------------------------   
        # 2) plot
        #---------------------------------------------------------------------------------------------------------------------------------
        #colors = ['#d62728','#1f77b4','#2ca02c','#7f7f7f'] # https://matplotlib.org/gallery/lines_bars_and_markers/markevery_prop_cycle.html#sphx-glr-gallery-lines-bars-and-markers-markevery-prop-cycle-py
        colors = ['k','g','r','k']
        #colors = ['C1','C2','C3','C4']
        linestyles = [(0, (4,2)), (0, (1,1)), (0, (4,1)), '-', '--', ':', '-.']
        text_fontsize = 16
        tick_fontsize = 13
        label_fontsize = 19
        legend_fontsize = 11
        text_pos = (1.0, 0.9)
        
        fig_nrows = 4
        fig_ncols = 3

        plt.rc('font', family='Arial')
        fig, axs = plt.subplots(fig_nrows, fig_ncols, figsize=(18,12))
        
        for irow in range(fig_nrows):
            for icol in range(fig_ncols):
                if irow == 0:        # ETKF
                    axs[irow, icol].plot(x_axs, trth_intp_list[icol]["corr"], color=colors[0], label=trth_intp_list[icol]["label"]) 
                    axs[irow, icol].plot(x_axs, etkf_intp_list[2*icol]["corr"], color=colors[1], label=etkf_intp_list[2*icol]["label_corr"])
                    axs[irow, icol].plot(x_axs, etkf_intp_list[2*icol+1]["corr"], color=colors[2], label=etkf_intp_list[2*icol+1]["label_corr"])
                    axs[irow, icol].text(text_pos[0], text_pos[1], etkf_intp_list[2*icol]["text"], fontsize=text_fontsize)
                elif irow == 1:      # LETKF
                    axs[irow, icol].plot(x_axs, trth_intp_list[icol]["corr"], color=colors[0], label=trth_intp_list[icol]["label"]) 
                    #axs[irow, icol].plot(x_axs, letkf_intp_list[3*icol]["corr"], color=colors[1], label=letkf_intp_list[3*icol]["label_corr"])
                    #axs[irow, icol].plot(x_axs, letkf_intp_list[3*icol+1]["corr"], color=colors[2], label=letkf_intp_list[3*icol+1]["label_corr"])
                    #axs[irow, icol].plot(x_axs, letkf_intp_list[3*icol+2]["corr"], color=colors[3], label=letkf_intp_list[3*icol+2]["label_corr"])                    
                    axs[irow, icol].plot(x_axs, letkf_intp_list[2*icol]["corr"], color=colors[1], label=letkf_intp_list[2*icol]["label_corr"])          # Ne = 3
                    axs[irow, icol].plot(x_axs, letkf_intp_list[2*icol+1]["corr"], color=colors[2], label=letkf_intp_list[2*icol+1]["label_corr"])      # Ne = 10
                    #axs[irow, icol].plot(x_axs, letkf_intp_list[3*icol]["rloc"], color=colors[3], linestyle=linestyle[0], label=letkf_intp_list[3*icol]["label_rloc"])
                    #axs[irow, icol].plot(x_axs, letkf_intp_list[3*icol+1]["rloc"], color=colors[3], linestyle=linestyle[1], label=letkf_intp_list[3*icol+1]["label_rloc"])
                    #axs[irow, icol].plot(x_axs, letkf_intp_list[3*icol+2]["rloc"], color=colors[3], linestyle=linestyle[2], label=letkf_intp_list[3*icol+2]["label_rloc"])                    
                    axs[irow, icol].plot(x_axs, letkf_intp_list[2*icol]["rloc"], color=colors[3], linestyle='--', label=letkf_intp_list[2*icol]["label_rloc"])          # Ne = 3
                    axs[irow, icol].plot(x_axs, letkf_intp_list[2*icol+1]["rloc"], color=colors[3], linestyle=linestyles[1], label=letkf_intp_list[2*icol+1]["label_rloc"])      # Ne = 10
                    axs[irow, icol].text(text_pos[0], text_pos[1], letkf_intp_list[2*icol]["text"], fontsize=text_fontsize)
                elif irow == 2:      # UTKF (SPKF)
                    axs[irow, icol].plot(x_axs, trth_intp_list[icol]["corr"], color=colors[0], label=trth_intp_list[icol]["label"]) 
                    axs[irow, icol].plot(x_axs, utkf_intp_list[icol]["corr"], color=colors[2], label=utkf_intp_list[icol]["label_corr"])
                    axs[irow, icol].text(text_pos[0], text_pos[1], utkf_intp_list[icol]["text"], fontsize=text_fontsize)
                elif irow == 3:      # LUTKF
                    axs[irow, icol].plot(x_axs, trth_intp_list[icol]["corr"], color=colors[0], label=trth_intp_list[icol]["label"]) 
                    axs[irow, icol].plot(x_axs, lutkf_intp_list[icol]["corr"], color=colors[2], label=lutkf_intp_list[icol]["label_corr"])
                    axs[irow, icol].plot(x_axs, lutkf_intp_list[icol]["rloc"], color=colors[3], linestyle='--', label=lutkf_intp_list[icol]["label_rloc"])
                    axs[irow, icol].text(text_pos[0], text_pos[1], lutkf_intp_list[icol]["text"], fontsize=text_fontsize)
                    axs[irow, icol].set_xlabel('Variables', fontsize=label_fontsize)

                if icol == 0:
                    axs[irow, icol].set_ylabel('Correlation [$\mathrm{x}_{20}$,$\mathbf{x}$]',fontsize=label_fontsize)
                axs[irow, icol].set_xlim(0., 40.)
                #axs[irow, icol].set_ylim(-0.245, 1.1)
                axs[irow, icol].set_ylim(-0.3, 1.1)
                axs[irow, icol].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
                axs[irow, icol].tick_params(axis='x', labelsize=tick_fontsize)
                axs[irow, icol].tick_params(axis='y', labelsize=tick_fontsize)
                #axs[irow, icol].legend(loc='upper right', fontsize=legend_fontsize)
                legend = axs[irow, icol].legend(loc='upper right', edgecolor='k', fancybox=False, framealpha=1., fontsize=legend_fontsize)
                legend.get_frame().set_linewidth(0.5)
                #axs[irow, icol].grid(True)

        fig.tight_layout()

        fn = '1_2_correlation_enkf_method_3_7_h_type_0_1_2_nem_10_3_nobs_200'
        if plot_save:
            save_plot(plt, os.path.join(plot_outdir,fn), ['png','svg','eps'])
        if plot_show:
            plt.show()

    #-------------------------------------------------------------------------------------------------------------------------------------
    # 2. single figure
    #-------------------------------------------------------------------------------------------------------------------------------------  
    else:
        enkf_method = 7
        h_type = 0
        nem = 3
        nobs = 200

        #---------------------------------------------------------------------------------------------------------------------------------
        # 1) read data 
        #---------------------------------------------------------------------------------------------------------------------------------
        truth_fn = os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_9_nEnKF_1_nem_10000_nobs_200_obs_loc_dist_0.95_h_type_0_y")
        fn = os.path.join(data_dir,"[xxcorr_b_mean_x20]enkf_method_{}_nEnKF_30_nem_{}_nobs_{}_obs_loc_dist_0.95_h_type_{}_y".format(enkf_method,nem,nobs,h_type))
        rloc_fn = os.path.join(data_dir,"[obs_rloc_x20]enkf_method_7_nEnKF_30_nem_3_nobs_200_obs_loc_dist_0.95_h_type_0_y")

        #---------------------------------------------------------------------------------------------------------------------------------   
        # 2) plot
        #---------------------------------------------------------------------------------------------------------------------------------
        plt.figure(figsize=(7,4))        
        plt.plot(x_axs, calc_corr(x, x_intp, x_axis_shift, np.loadtxt(truth_fn), intp), color='r', label='Optimal least squares')
        plt.plot(x_axs, calc_corr(x, x_intp, x_axis_shift, np.loadtxt(fn), intp), color='g', label='Local UTKF')
        plt.plot(x_axs, calc_corr(x, x_intp, x_axis_shift, np.loadtxt(rloc_fn), intp), color='k', linestyle=(0, (4, 2)), label='Localization function (GC)')

        plt.legend(loc='upper right',fontsize=12)
        plt.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        plt.text(1.0, 0.9, '%s, ${N}_{e}$ = 50'%(h_mathexp_str(0)), fontname="Arial", fontsize=14)
        plt.xlim(0., 40.)
        plt.xticks(fontsize=12)
        #plt.ylim(-0.3, 1.)
        plt.yticks(fontsize=12)

        plt.xlabel('Variable', fontsize=15)
        plt.ylabel('Correlation[$\mathrm{x}_{20}$,$\mathbf{x}$]',fontsize=15)
        plt.grid()
        plt.tight_layout()
        
        fn = '1_3_correlation_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(enkf_method, h_type, nem, nobs)
        if plot_save:
            save_plot(plt, os.path.join(plot_outdir,fn), ['png','svg','eps'])
        if plot_show:
            plt.show()

            
if __name__ == "__main__":
    #---------------------------------------------------------------------------------------------------
    # 1. plot collected or single correlation gragh (for paper)
    #---------------------------------------------------------------------------------------------------
    plot_corr_paper(True, os.path.join('data_corr_paper','obs_pos_dst_9'), True, 'fig', True)
