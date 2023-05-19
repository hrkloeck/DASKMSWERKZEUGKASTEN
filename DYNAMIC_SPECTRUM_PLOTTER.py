# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
#
#
# This program plots waterfall and averged spectra of a measurement set (MS) using 
# DASK-MS
#
# Hope you enjoy it
# 
# --------------------------------------------------------------------


def main():

    import sys

    from optparse import OptionParser

    import yaml

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from astropy.time import Time
    from datetime import datetime
    import matplotlib.dates as mdates


    from daskms import xds_from_ms,xds_from_table
    import dask
    import dask.array as da

    import DASK_MS_WERKZEUGKASTEN as INFMS


    # argument parsing
    #
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)


    parser.add_option('--MS_FILE', dest='msfile', type=str,
                      help='MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms')

    parser.add_option('--DATA_TYPE', dest='datacolumn', default='DATA',type=str,
                      help='which data column to use [defaul DATA]')

    parser.add_option('--FIELD_ID', dest='field_id', default=0,type=int,
                      help='if MS contains muliple field define on field')

    parser.add_option('--DOBSLWATERFALLSPEC', dest='dobslwfspec', action='store_true', default=False,
                      help='produce waterfall spectrum per baseline')

    parser.add_option('--DOPLOTAVGWATERFALLSPEC', dest='doavgwfspec', action='store_true', default=False,
                      help='produce an averaged waterfall sepctrum')

    parser.add_option('--DOPLOTAVGSPECTRUM', dest='doavgspec', action='store_true', default=False,
                      help='produce an average spectrum')
    
    parser.add_option('--PLOTFILEMARKER', dest='pltf_marker', default='PLT_',type=str,
                      help='add file indicator in front of the file [defaut = PLT_]')

    parser.add_option('--SHOW', dest='showparameter', default='AMP',type=str,
                      help='= Show the amplitude [AMP] or the [PHASE] [default is AMP]')

    parser.add_option('--DONOTSORTUVDIS', dest='dosortuvdis', action='store_false', default=True,
                      help='use original sequence of baselines. [default sort versus UV-distance ]')

    parser.add_option('--CHANNELSLIDE', dest='chnslide', default='[0,0]', type=str,
                      help='select channel range to plot [channel1,channel2]')

    parser.add_option('--SELECT_SPWD', dest='select_spwd', default=0, type=float,
                      help='select spectral window (default 0)')

    parser.add_option('--SELECT_BSL', dest='select_bsl', default='[[]]', type=str,
                      help='select baselines (e.g. [[ANT1,ANT2],[ANT3,ANT8]])')

    parser.add_option('--SELECT_ANT', dest='select_ant', default='[]', type=str,
                      help='select antennas (e.g. [ANT1,ANT2,ANT3])')

    parser.add_option('--SELECT_UVDIS', dest='select_uvdis', default='[0,0]', type=str,
                      help='select baselines via UV distance (e.g. [0,100] in meter)')

    parser.add_option('--TESTFLAG', dest='testfg', default='[[0,0]]', type=str,
                      help='test flag data channels [[channel1,channel2],[channel1,channel2]]')


    parser.add_option('--DOPROGRESSBAR', dest='progbar', action='store_true',default=False,
                      help='show progress bar ')


    
    # ----

    (opts, args)         = parser.parse_args()

     
    if opts.msfile == None:        
        parser.print_help()
        sys.exit()


    # set the parmaters
    #
    MSFN                = opts.msfile
    data_type           = opts.datacolumn
    field               = opts.field_id

    dobslwfspec         = opts.dobslwfspec
    doavgwfspec         = opts.doavgwfspec
    doavgspec           = opts.doavgspec
    testfg              = opts.testfg
    chnslide            = opts.chnslide
    select_bsl          = opts.select_bsl
    select_ant          = opts.select_ant
    select_uvdis        = opts.select_uvdis
    select_spwd         = int(opts.select_spwd)

    pltf_marker         = opts.pltf_marker
    showparameter       = opts.showparameter
    dosort_versus_uvdis = opts.dosortuvdis
    doshowprogressbar   = opts.progbar
    

    # ------------------------------------------------------------------------------
    

    # Source Information
    msource_info = INFMS.ms_source_info(MSFN)
    fie_sour     = INFMS.ms_field_info(MSFN)


    # check if multiple sources are present
    #
    if len(msource_info.keys()) > 1 and field <= len(msource_info.keys()):
        print('\nMultiple sources present in the MS file')

        print('\t- Please define FIELD_ID')
        fie_sour = INFMS.ms_field_info(MSFN)
        for fi in fie_sour:
            print('\t\t--FIELD_ID ',fi,' ',fie_sour[fi])
        print('\n\t- Use default FIELD_ID: ',field,' ',fie_sour[str(field)])
        source_name = fie_sour[str(field)]
    else:
        source_name = fie_sour[str(list(fie_sour.keys())[0])]
    #
    #
    if field > len(msource_info.keys()):
        print('\nMultiple sources present in the MS file')
        print('\t- FIELD_ID ',field,'unkown')
        print('\t- Please define FIELD_ID')
        fie_sour = INFMS.ms_field_info(MSFN)
        for fi in fie_sour:
            print('\t\t--FIELD_ID ',fi,' ',fie_sour[fi])
        print('\n')

        sys.exit(-1)

    # check which data column to use
    if data_type != 'DATA':
        if INFMS.ms_check_col(MSFN,data_type) == -1:
            print('\t- DATA column ',data_type,' not present in dataset')
            sys.exit(-1)


    # Add the source name of the plotfiles
    #
    pltf_marker += source_name+'_'

    
    # get the baseline information
    #
    ms_bsls         = np.array(INFMS.ms_baselines(MSFN,tabs='FEED'))
    ms_bsls_antname = INFMS.ms_baselines(MSFN,tabs='ANTENNA')

    #
    if np.cumprod(ms_bsls.shape)[-1] == 0:
        print('Caution no baselines defined -- ABORT')
        sys.exit(-1)

    # get bsl length 
    ms_bsls_length  = np.array(INFMS.ms_baselines_length(MSFN,ms_bsls))

    # switch to sort baseline versus UV distance 
    if dosort_versus_uvdis:
        bsl_index = np.argsort(ms_bsls_length)
    else:
        bsl_index = np.arange(len(ms_bsls))
    
    #
    # -------------------------


    # Get data description (e.g. spectral windows)
    #
    dades          = xds_from_table(MSFN+'::DATA_DESCRIPTION')
    didesinfo      = dades[0].compute()
    spwd_idx       = didesinfo.SPECTRAL_WINDOW_ID.data

    if len(spwd_idx) > 1:
        print('Dataset consist of various spectra windows: ',spwd_idx,' use ',int(select_spwd))
        if int(select_spwd) > spwd_idx[-1]:
            print('SPWD does not exsist!!!')
            sys.exit(-1)
    else:
        select_spwd = 0

    # select data based on baselines
    #
    if len(select_bsl) > 4:
        print('\t\t-- select on baselines')
        sel_bsl = eval(select_bsl)
        bsl_index_new = []
        #
        for bsl in bsl_index:
            for sbsel in sel_bsl:
                if (ms_bsls_antname[bsl][0] == sbsel[0] and ms_bsls_antname[bsl][1] == sbsel[1] ) or  (ms_bsls_antname[bsl][0] == sbsel[1] and ms_bsls_antname[bsl][1] == sbsel[0]):
                    bsl_index_new.append(bsl)

                
        bsl_index = bsl_index_new
        if len(bsl_index) == 0:
            print('Baseline not present in dataset: ',sel_bsl)
            sys.exit(-1)

    # select data based on antenna
    #
    if len(select_ant) > 2:
        print('\t\t-- select on antenna')
        sel_ant = eval(select_ant)
        bsl_index_new = []
        #
        for bsl in bsl_index:
            for sant in sel_ant:
                if ms_bsls_antname[bsl][0] == sant  or ms_bsls_antname[bsl][1] == sant:
                    bsl_index_new.append(bsl)
                
        bsl_index = bsl_index_new
        if len(bsl_index) == 0:
            print('Antenna not present in dataset: ',sel_ant)
            sys.exit(-1)


    # select data based on antenna
    #
    #
    if len(select_uvdis) > 2:
        sel_uvdis = eval(select_uvdis)
        if sel_uvdis[0] != sel_uvdis[1]:
            print('\t\t-- select on uvdistance')
            sel_uvdis = eval(select_uvdis)
            bsl_index_new = []
            #
            for bsl in bsl_index:
                if ms_bsls_length[bsl] > sel_uvdis[0] and ms_bsls_length[bsl] <= sel_uvdis[1]:
                        bsl_index_new.append(bsl)

            bsl_index = bsl_index_new
            if len(bsl_index) == 0:
                print('Baselines with UV distance not present in dataset: ',select_uvdis)
                sys.exit(-1)


    # get the polarisation info
    dapolinfo = INFMS.ms_pol_info(MSFN)
    stokes    = dapolinfo['STOKES']

    # get the frequency info
    daspc           = xds_from_table(MSFN+'::SPECTRAL_WINDOW')
    daspcinfo       = daspc[0].compute()
    spwd_freq       = daspcinfo.CHAN_FREQ.data
    spwd_freq_shape = spwd_freq.shape
    

    # check if MODEL DATA is present 
    get_model_data = INFMS.ms_check_col(MSFN,'MODEL_DATA')

  
    # get a data dictionary ordered by baselines 
    ms_bsl_data    = INFMS.ms_get_bsl_data(MSFN,field_idx=field,setspwd=int(select_spwd),bsls=ms_bsls,bsl_idx=bsl_index)


    # optain the data from Dask 
    bsl_data = dask.compute(ms_bsl_data)[0]


    print('\n=== Spectrum plotter ===')
    average_dynspec = {}
    average_dynspec['DATA'] = {}
    average_dynspec['MASK'] = {}



    #
    flagged_bsl = 0
    bslnidx     = 0
    #
    #
    # sweaps over the baslines
    for bsl_idx in bsl_index:

        if doshowprogressbar:
            print('Work on baseline: ',ms_bsls[bsl_idx])
            INFMS.progressBar(bslnidx,len(bsl_index))
    

        # ----- Here starts the investigation
        bsltime  = np.array(bsl_data[bsl_idx]['TIME_CENTROID'])
        bslfreq  = np.array(bsl_data[bsl_idx]['CHAN_FREQ'])


        # get data and model and merge all spwds into a large one 
        bsldata  = INFMS.merge_spwds(np.array(bsl_data[bsl_idx][data_type]))
        bslflag  = INFMS.merge_spwds(np.array(bsl_data[bsl_idx]['FLAG']))   # Note CASA flag data if boolean value is True
        #
        if get_model_data != -1:
            bslmodel  = INFMS.merge_spwds(np.array(bsl_data[bsl_idx]['MODEL']))
            
        # set the slicer of the data, its defined at the top
        #        
        ch_rang  = [0,bslfreq.shape[0]]
        tim_rang = [0,bsltime.shape[0]]

        chsel = eval(chnslide)
        if chsel[0] != chsel[1]:
            ch_rang  = chsel

        time_r   = slice(tim_rang[0],tim_rang[1])
        chan_r   = slice(ch_rang[0],ch_rang[1])

        # loop over all stokes parameters
        #
        for polr in range(len(stokes)):

            # define polarisation slicer
            pol_r = slice(polr,polr+1)

            # init the dic for each baseline and stokes values

            if stokes[polr] in average_dynspec == False:
                average_dynspec['DATA'][stokes[polr]] = {}
                average_dynspec['MASK'][stokes[polr]] = {}
        

            # Check number of flags and if entire selection is flagged
            #
            data_fully_flagged = -1
            bslfg_shape        = bslflag[time_r,chan_r,pol_r].shape
            #
            if bslfg_shape[0] * bslfg_shape[1] == np.count_nonzero(bslflag[time_r,chan_r,pol_r]):
                data_fully_flagged = 1
                flagged_bsl += 1
                if doshowprogressbar:
                    print('\n\t=== CAUTION NO DATA ON ', ms_bsls[bsl_idx],' ',stokes[pol_r])



            # use the selection also to get the time and frequency information
            #
            sel_time_range = bsltime[time_r]
            sel_freq_range = bslfreq[chan_r]
            
            # determine the averages
            #
            if data_fully_flagged == -1:

                # select subsection of the data 
                #
                cdata       = np.ma.masked_array(bsldata[time_r,chan_r,pol_r],mask=bslflag[time_r,chan_r,pol_r],fill_value=np.nan)
                cdata_shape = cdata.shape
                #
                if cdata_shape[2] == 1:
                    cdata = cdata.reshape(cdata_shape[0],cdata_shape[1])

                # Flag table indicates False as not flagged and True as flagged
                # need to invert to be used for future processing
                #
                mult_mask = np.invert(bslflag[time_r,chan_r,pol_r]).astype(dtype=int)
                mult_mask_shape = mult_mask.shape
                #
                if mult_mask_shape[2] == 1:
                    mult_mask = mult_mask.reshape(cdata_shape[0],cdata_shape[1])

                if (stokes[polr] in average_dynspec['DATA']) == False:
                    # Initiate the first array to be filled 
                    #
                    average_dynspec['DATA'][stokes[polr]] = np.copy(cdata * mult_mask)
                    average_dynspec['MASK'][stokes[polr]] = np.copy(mult_mask)
                    
                else:
                    # add each dynamic spectra with each other
                    #
                    average_data = np.add(average_dynspec['DATA'][stokes[polr]],cdata * mult_mask)
                    average_dynspec['DATA'][stokes[polr]] = average_data
                    #
                    average_mask = np.add(average_dynspec['MASK'][stokes[polr]],mult_mask)
                    average_dynspec['MASK'][stokes[polr]] = average_mask
                    #
                    if bslnidx == len(bsl_index) -1:
                        average_mask[average_mask == 0] = 1
                        average_dynspec[stokes[polr]] = average_data/average_mask
                        
                        
                if dobslwfspec:
                    #
                    # Plot  individual waterfall spectrum per baseline
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


                    if showparameter == 'AMP':
                        dynamic_spectrum   = np.absolute(cdata) * mult_mask
                    else:
                        dynamic_spectrum   = np.angle(cdata,deg=True)


                    # figure setup 
                    #
                    if len(sel_time_range) > 100:
                        im_size  = (8.27, 11.69)       # A4 portrait
                    else:
                        im_size  = (8.27, 11.69)[::-1]  # A4 landscape
                    plt.rcParams['figure.figsize'] = im_size

                    # plt filename 
                    #
                    
                    pltname = pltf_marker +showparameter+'_WATERFALL_'+'SPWD_'+str(select_spwd)+'_'+ms_bsls_antname[bsl_idx][0]+'_'+ms_bsls_antname[bsl_idx][1]+'_'+stokes[polr]+'.png'

                    #pltname = pltf_marker +str(bsl_idx)+'_'+stokes[polr]+'.png'

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
                    ax.set_title(source_name+' '+showparameter+', '+str(ms_bsls_antname[bsl_idx])+', corr '+stokes[polr]+', spwd '+str(select_spwd))
                    ax.minorticks_on()
                    #
                    ax.set_xlabel('channel')
                    #
                    ax.set_ylabel('time')
                    ax.yaxis_date()
                    ax.yaxis.set_tick_params(which='minor', bottom=False)
                    ax.set_yticks(time_plt_axis_ticks)
                    ax.set_yticklabels(time_plt_axis_labels,size=8)
                    ax.yaxis.set_tick_params(which="major", rotation=0)

                    # image
                    if showparameter == 'AMP':
                        image = plt.imshow(dynamic_spectrum,origin='lower',interpolation='nearest',cmap=cmap,norm=mpl.colors.LogNorm())
                    else:
                        image = plt.imshow(dynamic_spectrum,origin='lower',interpolation='nearest',cmap=cmap)

                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    # here you will find the trick for the colorbar and the size
                    # http://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib
                    divider = make_axes_locatable(ax)
                    cax     = divider.append_axes("right", size="5%", pad=0.15)
                    fig.colorbar(image,cax=cax)

                    # save
                    plt.savefig(pltname)
                    plt.close()
                    # clean matplotlib up
                    plt.cla()
                    plt.clf()
                    plt.close('all')
                    
        bslnidx += 1       



    # HERE WE CAN DO THE SPECTRUM AS ITSELF
    #
    if doavgspec:


        # loop over all stokes parameters
        #
        for polr in range(len(stokes)):

            # average the data in time
            avgerage_spectrum = INFMS.average_cdata(average_dynspec[stokes[polr]],axis=0)


            if showparameter == 'AMP':
                avg_spectrum = np.absolute(avgerage_spectrum)
            else:
                avg_spectrum   = np.angle(avgerage_spectrum,deg=True)



            # apply some test flagging
            #
            tfg = eval(testfg)
            if tfg[0][0] != tfg[0][1]:

                for f in tfg:
                    # nice thing to do test flagging
                    fg_chan = slice(f[0],f[1])        
                    avg_spectrum[time_r,fg_chan] = -9999


                avg_spectrum = np.ma.masked_where(avg_spectrum == -9999,avg_spectrum)


            

            if len(sel_time_range) > 100:
                im_size  = (8.27, 11.69)       # A4 portrait
            else:
                im_size  = (8.27, 11.69)[::-1]  # A4 landscape

            plt.rcParams['figure.figsize'] = im_size

            # the figures
            #
            pltname = pltf_marker +'SPECTRUM_'+'SPWD_'+str(select_spwd)+'_'+stokes[polr]+'.png'

            #
            fig, ax = plt.subplots()

            # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            #
            cmap = mpl.cm.cubehelix
            cmap.set_bad(color='black')

            # spectrum 
            #
            plt.scatter(np.arange(len(avg_spectrum)),avg_spectrum)


            #
            ax.set_title(source_name+' '+' corr '+stokes[polr]+' spwd '+str(select_spwd))

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

            # save
            plt.savefig(pltname)
            plt.close()
            # clean matplotlib up
            plt.cla()
            plt.clf()
            plt.close('all')

    



    # Plot the average waterfall spectrum
    #
    if doavgwfspec:
        
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




        # loop over all stokes parameters
        #
        for polr in range(len(stokes)):


            if showparameter == 'AMP':
                avg_dynamic_spectrum = np.absolute(average_dynspec[stokes[polr]])
            else:
                avg_dynamic_spectrum   = np.angle(average_dynspec[stokes[polr]],deg=True)


            # apply some test flagging
            #
            tfg = eval(testfg)
            if tfg[0][0] != tfg[0][1]:

                for f in tfg:
                    # nice thing to do test flagging
                    fg_chan = slice(f[0],f[1])        
                    avg_dynamic_spectrum[time_r,fg_chan] = -9999


                avg_dynamic_spectrum = np.ma.masked_where(avg_dynamic_spectrum == -9999,avg_dynamic_spectrum)


            if len(sel_time_range) > 100:
                im_size  = (8.27, 11.69)       # A4 portrait
            else:
                im_size  = (8.27, 11.69)[::-1]  # A4 landscape

            plt.rcParams['figure.figsize'] = im_size

            # the figures
            #
            pltname = pltf_marker +'AVERAGE_WATERFALL_'+'SPWD_'+str(select_spwd)+'_'+stokes[polr]+'.png'

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
            ax.set_title(source_name+' '+showparameter+', corr '+stokes[polr]+', spwd '+str(select_spwd))

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


            # save
            plt.savefig(pltname)
            plt.close()
            # clean matplotlib up
            plt.cla()
            plt.clf()
            plt.close('all')



    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------






if __name__ == "__main__":
    main()


