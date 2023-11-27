# HRK 2021
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
#
# This program extract information out of a measurement set (MS) using 
# DASK-MS
#
# 
# --------------------------------------------------------------------

import sys

from optparse import OptionParser



def main():


    import yaml

    import numpy as np
    import matplotlib.pyplot as plt

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

    parser.add_option('--DATA_TYPE', dest='datacolumn', default='CORRECTED_DATA',type=str,
                          help='which data column to use e.g. DATA [default CORRECTED_DATA]')

    parser.add_option('--SHOW', dest='showparameter', default='AMP',type=str,
                      help='= Show the amplitude [AMP] or the [PHASE] [default is AMP]')

    parser.add_option('--FIELD_ID', dest='field_id', default=0,type=int,
                      help='if MS contains muliple field define on field')

    parser.add_option('--SCAN_NUMBER', dest='scan_num', default=-1,type=int,
                      help='select scan number [no default]. Note overwrites FIELD_ID.')

    parser.add_option('--SELECT_BSL', dest='select_bsl', default='[[]]', type=str,
                      help='select baselines (e.g. [[ANT1,ANT2],[ANT3,ANT8]])')

    parser.add_option('--SELECT_ANT', dest='select_ant', default='[]', type=str,
                      help='select antennas (e.g. [ANT1,ANT2,ANT3])')

    parser.add_option('--SELECT_UVDIS', dest='select_uvdis', default='[0,0]', type=str,
                      help='select baselines via UV distance (e.g. [0,100] in meter)')

    parser.add_option('--CHANNELSLIDE', dest='chnslide', default='[0,0]', type=str,
                      help='select channel range to plot [channel1,channel2]')

    parser.add_option('--SELECT_SPWD', dest='select_spwd', default=-1, type=float,
                      help='select spectral window (default all)')

    parser.add_option('--PLOTFILEMARKER', dest='pltf_marker', default='PLT_',type=str,
                      help='add file indicator in front of the file [defaut = PLT_]')

    parser.add_option('--SWAPFIGURESIZE', dest='figureswap', action='store_false',default=True,
                        help='show progress bar ')

    parser.add_option('--DONOTSORTUVDIS', dest='dosortuvdis', action='store_false', default=True,
                      help='use original sequence of baselines. [default sort versus UV-distance ]')

    parser.add_option('--DOPROGRESSBAR', dest='progbar', action='store_true',default=False,
                      help='show progress bar ')

    parser.add_option('--WORK_DIR', dest='cwd', default='',type=str,
                      help='Points to the working directory if output is produced (e.g. usefull for containers)')



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
    showparameter       = opts.showparameter
    scan_num            = opts.scan_num
    doshowprogressbar   = opts.progbar
    pltf_marker         = opts.pltf_marker

    dofigureswap        = opts.figureswap
    chnslide            = opts.chnslide
    select_bsl          = opts.select_bsl
    select_ant          = opts.select_ant
    select_uvdis        = opts.select_uvdis    
    select_spwd         = int(opts.select_spwd)

    cwd                 = opts.cwd        # used to write out information us only for container

    dosort_versus_uvdis = opts.dosortuvdis    
    # ------------------------------------------------------------------------------
    

    # Source Information
    msource_info = INFMS.ms_source_info(MSFN)
    fie_sour     = INFMS.ms_field_info(MSFN)


    # define the scan  and overwrites the field number
    #
    if scan_num != -1:
        tot_scans = 0
        set_scan  = -1
        print('\nSelect scans ',scan_num)
        for so in msource_info:
            tot_scans += len(msource_info[so]['SCAN_ID'])
            if scan_num in msource_info[so]['SCAN_ID']:
                field = msource_info[so]['SOURCE_ID'][0]
                set_scan = scan_num
        if tot_scans < scan_num:
            print('\t--Caution seems that input scan number is to high!')
        if set_scan == -1:
            print('\t--Caution that scan seems not to be in the MS-set!')
            sys.exit(-1)
    else:
        set_scan = scan_num

    # check if multiple sources are present
    #
    if len(msource_info.keys()) > 1 and field <= len(msource_info.keys()):
        print('\nMultiple sources present in the MS file')
        print('\t- Please define FIELD_ID')
        for fi in fie_sour:
            print('\t\t--FIELD_ID ',fi,' ',fie_sour[fi])
        #
        print('\n\t- Use FIELD_ID: ',field,' ',fie_sour[str(field)])
        source_name = fie_sour[str(field)]
    else:
        source_name = fie_sour[str(list(fie_sour.keys())[0])]


    # just check if field ID has a match
    #
    if field > len(msource_info.keys()):
        print('\nMultiple sources present in the MS file')
        print('\t- FIELD_ID ',field,'unkown')
        print('\t- Please define FIELD_ID')
        for fi in fie_sour:
            print('\t\t--FIELD_ID ',fi,' ',fie_sour[fi])
        print('\n')
        sys.exit(-1)


    # check which data column to use
    if data_type != 'DATA':
        if INFMS.ms_check_col(MSFN,data_type) == -1:
            print('\t- DATA column ',data_type,' not present in dataset')
            sys.exit(-1)
        
    
    pltf_marker += source_name+'_'


    # do some selection
    data_timidex = [0,None]
    field        = 0


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
        if select_spwd == -1:
            print('Dataset consist of various spectra windows: ',spwd_idx,' use all')
        else:
            print('Dataset consist of various spectra windows: ',spwd_idx,' select ',select_spwd)
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
                    #print(bsl,ms_bsls[bsl], ms_bsls_antname[bsl])
                
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


    # select data based on uvdistance
    #
    if len(select_uvdis) > 2:
        sel_uvdis = eval(select_uvdis)
        if sel_uvdis[0] != sel_uvdis[1]:
            print('\t\t-- select on uvdistance')
            bsl_index_new = []
            #
            for bsl in bsl_index:
                if ms_bsls_length[bsl] > sel_uvdis[0] and ms_bsls_length[bsl] <= sel_uvdis[1]:
                        bsl_index_new.append(bsl)

            bsl_index = bsl_index_new
            if len(bsl_index) == 0:
                print('Baselines with UV distance not present in dataset: ',select_uvdis)
                sys.exit(-1)
    #
    # ####

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
    # not sure how to cleverly sort the data that it can do both
    #
    if scan_num == -1:
        ms_bsl_data    = INFMS.ms_get_bsl_data(MSFN,field_idx=field,spwd=int(select_spwd),bsls=ms_bsls,bsl_idx=bsl_index)
    else:
        ms_bsl_data    = INFMS.ms_get_bsl_data_scan(MSFN,field_idx=field,scan_num=scan_num,spwd=int(select_spwd),bsls=ms_bsls,bsl_idx=bsl_index)


    # optain the data from Dask 
    bsl_data = dask.compute(ms_bsl_data)[0]


    print('\n=== Generate baseline measurements versus time plots')

    stats_info = {}
    i          = 0
    flagged_bsl = 0
    #
    # sweaps over the baslines
    for bsl_idx in bsl_index:

        
        if doshowprogressbar:
            # print('Work on baseline: ',ms_bsls[bsl_idx])
            INFMS.progressBar(i,len(bsl_index))
            i += 1

        # init the dic for each baseline
        if stats_info.get(bsl_idx) == None:
            stats_info[bsl_idx] = {}

        # ----- get the time info 
        bsltime  = np.array(bsl_data[bsl_idx]['TIME_CENTROID'])


        # get the freq info
        bslfreq_spwd_chan  = np.array(bsl_data[bsl_idx]['CHAN_FREQ'])
        bslfreq = INFMS.merge_spwds_freqs(bslfreq_spwd_chan)



        # get data and model and merge all spwds into a large one 
        bsldata  = INFMS.merge_spwds(np.array(bsl_data[bsl_idx][data_type]))
        bslflag  = INFMS.merge_spwds(np.array(bsl_data[bsl_idx]['FLAG']))   # Note CASA flag data if boolean value is True
        if get_model_data != -1:
            bslmodel  = INFMS.merge_spwds(np.array(bsl_data[bsl_idx]['MODEL_DATA']))

            
    
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
            if stats_info[bsl_idx].get(stokes[polr]) == None:
                stats_info[bsl_idx][stokes[polr]] = {}
        

            # Check number of flags and if entire selection is flagged
            #
            data_fully_flagged = -1
            bslfg_shape        = bslflag[time_r,chan_r,pol_r].shape
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

                # Average the data in frequency
                cdata_avg_freq = INFMS.average_cdata(cdata,axis=1)


                if get_model_data != -1:
                    cmodel       = np.ma.masked_array(bslmodel[time_r,chan_r,pol_r],mask=bslflag[time_r,chan_r,pol_r],fill_value=np.nan)
                    cmodel_shape = cmodel.shape

                    if cmodel_shape[2] == 1:
                        cmodel = cmodel.reshape(cmodel_shape[0],cmodel_shape[1])

                    # Average the model in frequency
                    cmodel_avg_freq = INFMS.average_cdata(cmodel,axis=1)

                    # store baseline averages
                    #
                    stats_info[bsl_idx][stokes[polr]]['DATA_AMP_time']     = np.abs(cdata_avg_freq)
                    stats_info[bsl_idx][stokes[polr]]['DATA_PHASE_time']   = np.angle(cdata_avg_freq,deg=True)
                    #
                    stats_info[bsl_idx][stokes[polr]]['MODEL_AMP_time']    = np.abs(cmodel_avg_freq)
                    stats_info[bsl_idx][stokes[polr]]['MODEL_PHASE_time']  = np.angle(cmodel_avg_freq,deg=True)

                else:
                    stats_info[bsl_idx][stokes[polr]]['DATA_AMP_time']     = np.abs(cdata_avg_freq)
                    stats_info[bsl_idx][stokes[polr]]['DATA_PHASE_time']   = np.angle(cdata_avg_freq,deg=True)
                    #
                    stats_info[bsl_idx][stokes[polr]]['MODEL_AMP_time']    = np.array([np.nan])
                    stats_info[bsl_idx][stokes[polr]]['MODEL_PHASE_time']  = np.array([np.nan])

            else:

                    stats_info[bsl_idx][stokes[polr]]['DATA_AMP_time']     = np.array([np.nan])
                    stats_info[bsl_idx][stokes[polr]]['DATA_PHASE_time']   = np.array([np.nan])
                    #
                    stats_info[bsl_idx][stokes[polr]]['MODEL_AMP_time']    = np.array([np.nan])
                    stats_info[bsl_idx][stokes[polr]]['MODEL_PHASE_time']  = np.array([np.nan])

    #
    # --- 



    # sweap over the baslines
    #
    for bsl_idx in bsl_index:

        if doshowprogressbar:
            # print('Work on baseline: ',ms_bsls[bsl_idx])
            INFMS.progressBar(i,len(bsl_index))
            i += 1
                    

        import matplotlib.pyplot as plt

        # loop over all stokes parameter
        for polr in range(len(stokes)):
            
            
            if showparameter == 'AMP':
                data_versus_time   = stats_info[bsl_idx][stokes[polr]]['DATA_AMP_time']
                model_versus_time  = stats_info[bsl_idx][stokes[polr]]['MODEL_AMP_time']

            else:
                data_versus_time   = stats_info[bsl_idx][stokes[polr]]['DATA_PHASE_time']
                model_versus_time  = stats_info[bsl_idx][stokes[polr]]['MODEL_PHASE_time']



            # old commented it out if np.isnan(data_versus_time[0]) == False:

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
                    #time_plt_axis_ticks.append(i)
                    time_plt_axis_ticks.append(sel_time_range[i])


            pltname = cwd + pltf_marker +showparameter+'_VPLOT_'+'SPWD_'+str(select_spwd)+'_'+ms_bsls_antname[bsl_idx][0]+'_'+ms_bsls_antname[bsl_idx][1]+'_'+stokes[polr]+'.png'

            # figure setup 
            #
            # default is landscape
            #
            if dofigureswap:
                    im_size  = (8.27, 11.69)[::-1]  # A4 landscape
            else:
                    im_size  = (8.27, 11.69)       # A4 portrait

            plt.rcParams['figure.figsize'] = im_size


            fig, ax = plt.subplots()



            ax.set_title(source_name+' '+showparameter+', '+str(ms_bsls_antname[bsl_idx])+', corr '+stokes[polr]+', spwd '+str(select_spwd))

            plt.scatter(sel_time_range,data_versus_time)
            if data_versus_time.shape == model_versus_time.shape:
                plt.plot(sel_time_range,model_versus_time,color='red')

            #ax.set_xlabel('time')
            ax.xaxis_date()
            #ax.xaxis.set_tick_params(which='minor', bottom=False)
            ax.set_xticks(time_plt_axis_ticks)
            ax.set_xticklabels(time_plt_axis_labels,size=8)
            ax.xaxis.set_tick_params(which="major", rotation=90)

            if showparameter == 'AMP':
                ax.set_ylabel(showparameter.lower()+' [Jy]')
            else:
                ax.set_ylabel(showparameter.lower()+' [deg]')

            plt.savefig(pltname)
            plt.close()
                    


    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------




if __name__ == "__main__":
    main()


