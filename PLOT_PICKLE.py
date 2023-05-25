
# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
# Only works if numpy astropy matplotlib is in your python environment installed !!!
#
# This program plots waterfall and averged spectra of a measurement set (MS) using 
# DASK-MS
#
# Hope you enjoy it
# 
# --------------------------------------------------------------------

# this is just a small script to plot the pickle output spectra or waterfall from
# the DYNAMIC_SPECTRUM_PLOTTER.py

# e.g. singularity exec --bind $PWD HRK_CASA_6.5_DASK.simg python DYNAMIC_SPECTRUM_PLOTTER.py --MS_FILE=1491291289.1ghz.1.1ghz.4hrs.ms --DOPLOTAVGSPECTRUM --DOPLOTAVGWATERFALLSPEC --DO_SAVE_AVERAGE_DATA 
#
# This will generate some pickle files 
# e.g. PLT_0252-712_SPECTRUM_AMP_SPWD_0_XX_pickle.py
#
import sys
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.time import Time
from datetime import datetime

def getparameter(filename):
    """
    get the parameter out again
    """
    import pickle
    if filename.count('.py') == 0:
       pfile  = open(filename+'.py','rb')
    else:
       pfile  = open(filename,'rb')
    data =  pickle.load(pfile)
    pfile.close()
    return(data)


# take the input as filename
#
filename    = sys.argv[1]

pickle_data = getparameter(filename)

dofigureswap = False

if  pickle_data['DATA']['type'] == 'SPECTRUM':

        sel_freq_range = pickle_data['DATA']['sel_freq_range']
        avg_spectrum   = pickle_data['DATA']['avg_spectrum']

        scan_num       = pickle_data['DATA']['scan_num']
        showparameter  = pickle_data['DATA']['showparameter']
        corr           = pickle_data['DATA']['corr']
        source_name    = pickle_data['DATA']['source_name'] 
        select_spwd    = pickle_data['DATA']['select_spwd']
        set_scan       = pickle_data['DATA']['set_scan']


        if len(sel_freq_range) > 100 and dofigureswap == False:
            im_size  = (8.27, 11.69)       # A4 portrait
        else:
            im_size  = (8.27, 11.69)[::-1]  # A4 landscape

        plt.rcParams['figure.figsize'] = im_size

        # the figures
        #
        #if scan_num  == -1:
        #    pltname = pltf_marker +'SPECTRUM_'+'SPWD_'+str(select_spwd)+'_'+stokes[polr]+'.png'
        #else:
        #    pltname = pltf_marker +'SPECTRUM_'+'SCAN_'+str(set_scan)+'_SPWD_'+str(select_spwd)+'_'+stokes[polr]+'.png'
        #
        fig, ax = plt.subplots()

        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        #
        cmap = mpl.cm.cubehelix
        cmap.set_bad(color='black')

        # spectrum 
        #
        plt.scatter(np.arange(len(avg_spectrum)),avg_spectrum)


        # title 
        if scan_num  == -1:
            ax.set_title(source_name+' '+', corr '+corr+', spwd '+str(select_spwd))
        else:
            ax.set_title(source_name+', scan '+str(set_scan)+', corr '+corr+', spwd '+str(select_spwd))


        ax.minorticks_on()
        #
        if showparameter == 'AMP':
            ax.set_ylabel(showparameter.lower()+' [Jy]')
        else:
            ax.set_ylabel(showparameter.lower()+' [deg]')

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
                if f > 0 and f < len(sel_freq_range) -1:
                        frq_ticks_label.append('%e'%sel_freq_range[int(f)])
                        frq_ticks.append(f)

        ax2.set_xticks(frq_ticks)
        ax2.set_xticklabels(frq_ticks_label,size=8)
        #

        print('kala')
        plt.show()
        sys.exit(-1)
        # save
        plt.savefig(pltname)
        plt.close()
        # clean matplotlib up
        plt.cla()
        plt.clf()
        plt.close('all')

    



if  pickle_data['DATA']['type'] == 'WATERFALL':

            sel_freq_range = pickle_data['DATA']['sel_freq_range']
            sel_time_range = pickle_data['DATA']['sel_time_range']

            avg_dynamic_spectrum   = pickle_data['DATA']['avg_dynamic_spectrum']

            scan_num       = pickle_data['DATA']['scan_num']
            showparameter  = pickle_data['DATA']['showparameter']
            corr           = pickle_data['DATA']['corr']
            source_name    = pickle_data['DATA']['source_name'] 
            select_spwd    = pickle_data['DATA']['select_spwd']
            set_scan       = pickle_data['DATA']['set_scan']
                



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





            # Start the plotting
            #
            if len(sel_time_range) > 100 and dofigureswap == False:
                im_size  = (8.27, 11.69)       # A4 portrait
            else:
                im_size  = (8.27, 11.69)[::-1]  # A4 landscape

            plt.rcParams['figure.figsize'] = im_size

            # the figures
            #
        
            #if scan_num == -1:
            #    pltname = pltf_marker +'AVERAGE_WATERFALL_'+'SPWD_'+str(select_spwd)+'_'+stokes[polr]+'.png'
            #else:
            #    pltname = pltf_marker +'AVERAGE_WATERFALL_'+'SCAN_'+str(set_scan)+'_SPWD_'+str(select_spwd)+'_'+stokes[polr]+'.png'



            #
            fig, ax = plt.subplots()

            # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            #
            cmap = mpl.cm.cubehelix
            cmap.set_bad(color='black')


            # image
            #
            if showparameter == 'AMP':
                image1 = plt.imshow(avg_dynamic_spectrum,origin='lower',interpolation='nearest',cmap=cmap,norm=mpl.colors.LogNorm())
            else:
                image1 = plt.imshow(avg_dynamic_spectrum,origin='lower',interpolation='nearest',cmap=cmap)

            #
            if scan_num == -1:
                ax.set_title(source_name+' '+showparameter+', corr '+corr+', spwd '+str(select_spwd))
            else:
                ax.set_title(source_name+' '+showparameter+', scan '+str(set_scan)+', corr '+corr+', spwd '+str(select_spwd))

            ax.minorticks_on()
            #
            ax.set_xlabel('channel')
            #

            # set frequency as top axis 
            #
            frq_ticks_org   = ax.get_xticks()
            if len(frq_ticks_org) > 2:
                ax2 = ax.secondary_xaxis("top")
                #
                frq_ticks       = []
                frq_ticks_label = []

                for f in frq_ticks_org:
                    if f > 0 and f < len(sel_freq_range) -1:
                        frq_ticks_label.append('%e'%sel_freq_range[int(f)])
                        frq_ticks.append(f)

                ax2.set_xticks(frq_ticks)
                ax2.set_xticklabels(frq_ticks_label,size=8)
                #

            ax.set_ylabel('time')
            ax.yaxis_date()
            ax.yaxis.set_tick_params(which='minor', bottom=False)
            ax.set_yticks(time_plt_axis_ticks)
            ax.set_yticklabels(time_plt_axis_labels,size=8)
            ax.yaxis.set_tick_params(which="major", rotation=0)
           

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            # here you will find the trick for the colorbar and the size
            # http://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="5%", pad=0.15)
            fig.colorbar(image1,cax=cax)


            print('kala')
            plt.show()
            sys.exit(-1)
            # save
            plt.savefig(pltname)
            plt.close()
            # clean matplotlib up
            plt.cla()
            plt.clf()
            plt.close('all')
