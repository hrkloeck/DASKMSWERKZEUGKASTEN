# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
# this is a first attemp to a PLOT library to
# handle waterfall and a normal spectrum
#
# Hope you enjoy it
# 
# --------------------------------------------------------------------


import numpy as np
import sys
import json

from astropy.time import Time
from astropy import units as u
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

from mpl_toolkits.axes_grid1 import make_axes_locatable
# here you will find the trick for the colorbar and the size
# http://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib


def group_multiple_spwds(dyn_specs):
    """
    check if the data has different obs and spwd
    """

    # get the overall accessing
    #
    scans = dyn_specs.keys()

    multi_obs_info   = []
    group_scan_spwds = []
    spwd_match       = 1
    #
    for sc in scans:

        # get entire spwds of the file
        #
        spwds_in_file = dyn_specs[sc]['INFO_SPWD']
        
        # check the keys per scan
        #
        sscan_keys    = dyn_specs[sc].keys()

        scan_key_info   = []
        spwd_per_scan   = []
        #
        for ssc in sscan_keys:
            if ssc.count('INFO') > 0:
                scan_key_info.append(ssc)
            else:
                spwd_per_scan.append(ssc)

        if len(spwd_per_scan) != len(spwds_in_file):
            full_spwd_match = -1
            spwd_match      = -1
        else:
            full_spwd_match = 1

        multi_obs_info.append([sc,scan_key_info,spwd_per_scan,full_spwd_match])
        

    if spwd_match == -1:
        for s in range(len(multi_obs_info)-1):
            if s == 0:
                group_scan_spwds.append(multi_obs_info[s][2])
            if multi_obs_info[s][2] == multi_obs_info[s+1][2]:
                isthere = -1
                for g in group_scan_spwds:
                    if g == multi_obs_info[s+1][2]:
                        isthere = 1
                if isthere == -1:
                    group_scan_spwds.append(multi_obs_info[s+1][2])
    else:
        
       group_scan_spwds.append(multi_obs_info[0][2]) 

    return multi_obs_info,group_scan_spwds



def concats_scan_spwds(dataspec,newspwd,axis=0):
    """
    concatenate spwd use axis=1 
    concatenate time use axis=0
    """
    if len(dataspec) != 0: 
        if len(newspwd) != 0:
                dataspec = np.concatenate((dataspec,newspwd),axis)        
    else:
        dataspec = newspwd

    return dataspec




def combine_averaged_data(dyn_specs,select_freq,select_time,select_spwd,print_info=True):
    """
    will combine the averaged data
    """

    multi_obs_info,group_scan_spwds = group_multiple_spwds(dyn_specs)


    # build a merged waterfall spectrum
    #
    concat_freq       = []
    concat_time       = []
    concat_data       = []
    concat_datastd    = []
    concat_data_flag  = []

    # new
    concat_freq_per_sw = []
    concat_chan_per_sw = []

    #
    for grspwd in group_scan_spwds:

        concat_freq_t      = []
        concat_time_t      = [] 
        concat_data_t      = [] 
        concat_datastd_t   = []
        concat_data_flag_t = []
        time_sc            = []

        # new
        concat_freq_t_psw  = []
        concat_chan_t_psw  = []

        # check the scans
        #
        for sc in multi_obs_info:

            data_type    = dyn_specs[sc[0]][sc[1][3]]
            info_corr    = dyn_specs[sc[0]][sc[1][1]]

            concat_sw_data      = [] 
            concat_sw_datastd   = [] 
            concat_sw_flag      = [] 
            concat_sw_freq      = [] 
            
            # new
            concat_sw_freq_sw   = [] 
            concat_sw_chan_sw   = [] 


            # check selection
            #
            if select_spwd == -1:

                # combine all spwd
                #
                if grspwd == sc[2]:   # check that only equal blocks of SPWD are combined

                    for sw in sc[2]:

                            if print_info:
                                print('sc',sc)
                                print('spwd',sw)
                                print('datatype',data_type)
                                print('info_corr',info_corr)
                                print('time',np.array(dyn_specs[sc[0]][str(sw)]['time_range']).shape)
                                print('freq',np.array(dyn_specs[sc[0]][str(sw)]['chan_freq']).shape)
                                print('flag',np.array(dyn_specs[sc[0]][str(sw)]['flag']).shape)
                                print('data',np.array(dyn_specs[sc[0]][str(sw)][data_type]).shape)


                            # concatenate the spwds
                            #
                            concat_sw_data      = concats_scan_spwds(concat_sw_data,dyn_specs[sc[0]][str(sw)][data_type],axis=1)   
                            concat_sw_datastd   = concats_scan_spwds(concat_sw_datastd,dyn_specs[sc[0]][str(sw)][data_type+'STD'],axis=1)  
                            concat_sw_flag      = concats_scan_spwds(concat_sw_flag,dyn_specs[sc[0]][str(sw)]['flag'],axis=1)   
                            concat_sw_freq      = concats_scan_spwds(concat_sw_freq,dyn_specs[sc[0]][str(sw)]['chan_freq'],axis=0)
                            #

                            # needs some bookkeeping of the channels and freq
                            #
                            len_freqs           = len(dyn_specs[sc[0]][str(sw)]['chan_freq'])
                            frqspsw             = np.ones(len_freqs)*float(sw)
                            chanspsw            = np.arange(len_freqs)
                            concat_sw_freq_sw   = concats_scan_spwds(concat_sw_freq_sw,frqspsw,axis=0)
                            concat_sw_chan_sw   = concats_scan_spwds(concat_sw_chan_sw,chanspsw,axis=0)
                            #

                            # time
                            time_sc             = dyn_specs[sc[0]][str(sw)]['time_range']

                else:
                    time_sc  = []

            else:


                if str(select_spwd) in sc[2]:

                    # extract the selected spwd
                    #
                    concat_sw_data      = dyn_specs[sc[0]][str(select_spwd)][data_type]
                    concat_sw_datastd   = dyn_specs[sc[0]][str(select_spwd)][data_type+'STD']
                    concat_sw_flag      = dyn_specs[sc[0]][str(select_spwd)]['flag']
                    concat_sw_freq      = dyn_specs[sc[0]][str(select_spwd)]['chan_freq']
                    #


                    # needs some bookkeeping of the channels and freq
                    #
                    len_freqs           = len(dyn_specs[sc[0]][str(select_spwd)]['chan_freq'])
                    frqspsw             = np.ones(len_freqs)*float(select_spwd)
                    chanspsw            = np.arange(len_freqs)
                    concat_sw_freq_sw   = concats_scan_spwds(concat_sw_freq_sw,frqspsw,axis=0)
                    concat_sw_chan_sw   = concats_scan_spwds(concat_sw_chan_sw,chanspsw,axis=0)
                    #
                    time_sc             = dyn_specs[sc[0]][str(select_spwd)]['time_range']

                else:
                    concat_sw_data      = [] 
                    concat_sw_datastd   = [] 
                    concat_sw_flag      = []
                    concat_sw_freq      = []
                    #
                    concat_sw_freq_sw   = []
                    concat_sw_chan_sw   = []
                    time_sc             = []


            # combine all scans
            #
            concat_data_t      = concats_scan_spwds(concat_data_t,concat_sw_data,axis=0) 
            concat_datastd_t   = concats_scan_spwds(concat_datastd_t,concat_sw_datastd,axis=0) 
            concat_data_flag_t = concats_scan_spwds(concat_data_flag_t,concat_sw_flag,axis=0) 

            concat_time_t      = concats_scan_spwds(concat_time_t,time_sc,axis=0) 
            if len(concat_sw_freq) > 0:
                concat_freq_t      = concat_sw_freq #concats_scan_spwds(concat_freq_t,concat_sw_freq,axis=0) 

            if len(concat_sw_freq_sw) > 0:
                concat_freq_t_psw = concat_sw_freq_sw

            if len(concat_sw_chan_sw) > 0:
                concat_chan_t_psw = concat_sw_chan_sw

            #print(grspwd,sc[2],sc[0],'data -- ',np.array(concat_data_t).shape)
            #print(grspwd,sc[2],sc[0],'freq -- ', np.array(concat_freq_t).shape)
            #print(grspwd,sc[2],sc[0],'time -- ',np.array(concat_time_t).shape)
            #print('\n')

        # put everything togethe per group
        #
        concat_data.append(concat_data_t)
        concat_datastd.append(concat_datastd_t)
        concat_data_flag.append(concat_data_flag_t)

        concat_freq.append(concat_freq_t)
        concat_time.append(concat_time_t)
        
        concat_freq_per_sw.append(concat_freq_t_psw)
        concat_chan_per_sw.append(concat_chan_t_psw)

    return concat_data,concat_datastd,concat_data_flag,concat_freq,concat_time,concat_freq_per_sw,concat_chan_per_sw







def plot_waterfall_spec(avg_dynspec,sel_time_range,frequency,select_spwd,data_type,showparameter,stokes,source_name,plt_filename,cwd='',dofigureswap=False,doplt=True):

                    #
                    # Plot waterfall spectrum 
                    #




                    # convert the Julian time 
                    #
                    time_iso = Time(sel_time_range/(24.*3600.),scale='utc',format='mjd').ymdhms


                    # define the number of tick labels in time 
                    #
                    if len(sel_time_range) > 100:
                        nth_y = 10
                    else:
                        nth_y = 5

                    every_nth_y = int(len(sel_time_range)/nth_y)
                    time_plt_axis_labels = []
                    time_plt_axis_ticks  = []
                    for i in range(len(time_iso)):
                        if i % every_nth_y == 0:
                            sec = int(time_iso[i][5])
                            plt_time = str(time_iso[i][0])+'-'+str(time_iso[i][1])+'-'+str(time_iso[i][2])+' '+str(time_iso[i][3])+':'+str(time_iso[i][4])+':'+str(sec)
                            time_plt_axis_labels.append(datetime.strptime(plt_time,'%Y-%m-%d %H:%M:%S'))
                            # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
                            time_plt_axis_ticks.append(i)
                            #time_plt_axis_ticks.append(sel_time_range[i])
                    #
                    # ####################



                    # define the number of tick labels in frequency 
                    #
                    if len(frequency) > 100:
                        nth_y = 7
                    else:
                        nth_y = 4

                    every_nth_y = int(len(frequency)/nth_y)

                    freq_plt_axis_labels = []
                    freq_plt_axis_ticks  = []

                    for i in range(len(frequency)):
                        if i % every_nth_y == 0:
                            freq_plt_axis_labels.append('%.4e'%frequency[i])
                            # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
                            freq_plt_axis_ticks.append(i)
                    #
                    # ####################





                    # figure setup 
                    #
                    if len(sel_time_range) > 100 and dofigureswap == False:
                        im_size  = (8.27, 11.69)       # A4 portrait
                    else:
                        im_size  = (8.27, 11.69)[::-1]  # A4 landscape
                    plt.rcParams['figure.figsize'] = im_size


                    # plt filename 

                    pltname =  cwd + plt_filename+'.png'

                    # the figures
                    #
                    fig, ax = plt.subplots()
                    #
                    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
                    #
                    cmap = mpl.cm.cubehelix
                    #cmap = mpl.cm.CMRmap
                    #cmap = mpl.cm.nipy_spectral
                    cmap.set_bad(color='black')
                    #
                    
                    # title
                    #
                    ax.set_title(source_name+' '+data_type+', '+showparameter+', corr '+stokes)

                    #
                    ax.minorticks_on()

                    # channel axis
                    #
                    ax.set_xlabel('channel')
                    #

                    # set frequency as top axis 
                    #
                    ax2 = ax.secondary_xaxis("top")

                    ax2.set_xticks(freq_plt_axis_ticks)
                    ax2.set_xticklabels(freq_plt_axis_labels,size=8)
                    #


                    # time axis
                    #
                    ax.set_ylabel('time')
                    ax.yaxis_date()
                    ax.yaxis.set_tick_params(which='minor', bottom=False)
                    ax.set_yticks(time_plt_axis_ticks)
                    ax.set_yticklabels(time_plt_axis_labels,size=8)
                    ax.yaxis.set_tick_params(which="major", rotation=0)


                    # image
                    if showparameter == 'AMP':
                        image = plt.imshow(avg_dynspec,origin='lower',interpolation='nearest',cmap=cmap,norm=mpl.colors.LogNorm())
                    else:
                        image = plt.imshow(avg_dynspec,origin='lower',interpolation='nearest',cmap=cmap)


                    divider = make_axes_locatable(ax)
                    cax     = divider.append_axes("right", size="5%", pad=0.15)
                    fig.colorbar(image,cax=cax)

                    if doplt:
                        # save
                        plt.savefig(pltname)
                        plt.close()
                    else:
                        plt.show()
                    # clean matplotlib up
                    plt.cla()
                    plt.clf()
                    plt.close('all')



def spectrum_average(avg_dynspec,sel_time_range,frequency,select_spwd,data_type,showparameter,stokes,source_name,plt_filename,cwd='',dofigureswap=False,doplt=True):


            if len(frequency) > 100 and dofigureswap == False:
                im_size  = (8.27, 11.69)       # A4 portrait
            else:
                im_size  = (8.27, 11.69)[::-1]  # A4 landscape

            plt.rcParams['figure.figsize'] = im_size

            # plt filename 
            #
            pltname =  cwd + plt_filename+'.png'


            #
            fig, ax = plt.subplots()


            # spectrum 
            #
            plt.scatter(np.arange(len(avg_dynspec)),avg_dynspec)

    
            # title
            #
            ax.set_title(source_name+' '+data_type+', '+showparameter+', corr '+stokes)


            ax.minorticks_on()
            #ax.tick_params(direction="in")
            #
            if showparameter == 'AMP':
                ax.set_ylabel(showparameter.lower()+' [Jy]')
            elif showparameter == 'FLAG':
                ax.set_ylabel(showparameter.lower())
            elif showparameter == 'PHASE':
                ax.set_ylabel(showparameter.lower()+' [deg]')
            else:
                ax.set_ylabel(showparameter.lower())

            ax.set_xlabel('channel')
            #


            # set frequency as top axis 
            #
            ax2 = ax.secondary_xaxis("top")

            ax2.set_xticks(ax.get_xticks())

            frq_ticks_org   = ax.get_xticks()
            frq_ticks       = []
            frq_ticks_label = []
            
            for f in frq_ticks_org:
                    if f > 0 and f < len(frequency) -1:
                            frq_ticks_label.append('%.4e'%frequency[int(f)])
                            frq_ticks.append(f)

            ax2.set_xticks(frq_ticks)
            ax2.set_xticklabels(frq_ticks_label,size=8)
            #


            if doplt:
                # save
                plt.savefig(pltname)
                plt.close()
            else:
                plt.show()

            # clean matplotlib up
            plt.cla()
            plt.clf()
            plt.close('all')


