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
#
# In a singularity container you can run it like
#
# singularity exec --bind ${PWD}:/data HRK_CASA_6.5_DASK.simg python DYNAMIC_SPECTRUM_PLOTTER.py --MS_FILE= --DOPLOTAVGSPECTRUM --DOPLOTAVGWATERFALLSPEC --DO_SAVE_AVERAGE_DATA= --WORK_DIR='/data/'
#
# --------------------------------------------------------------------
#
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
    import WERKZEUGKASTEN_PLOT_LIB as WZKPL
    import RFI_MITILIB as RFIM


    # argument parsing
    #
    usage = "usage: %prog [options]\
            This programme can be used to plot and store an average (over baseline or antenna) waterfall spectrum"

    parser = OptionParser(usage=usage)


    parser.add_option('--MS_FILE', dest='msfile', type=str,
                      help='MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms')

    parser.add_option('--DATA_TYPE', dest='datacolumn', default='DATA',type=str,
                      help='which data column to use e.g. CORRECTED_DATA [default DATA]')

    parser.add_option('--FIELD_ID', dest='field_id', default=0,type=int,
                      help='if MS contains muliple field define on field')

    parser.add_option('--SCAN_NUMBER', dest='scan_num', default=-1,type=int,
                      help='select scan number [no default]. Note overwrites FIELD_ID.')

    parser.add_option('--DOBSLWATERFALLSPEC', dest='dobslwfspec', action='store_true', default=False,
                      help='produce waterfall spectrum')

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

    parser.add_option('--SELECT_SPWD', dest='select_spwd', default=-1, type=float,
                      help='select spectral window (default all)')

    parser.add_option('--SELECT_BSL', dest='select_bsl', default='[[]]', type=str,
                      help='select baselines (e.g. [[ANT1,ANT2],[ANT3,ANT8]])')

    parser.add_option('--SELECT_ANT', dest='select_ant', default='[]', type=str,
                      help='select antennas (e.g. [ANT1,ANT2,ANT3])')

    parser.add_option('--SELECT_UVDIS', dest='select_uvdis', default='[0,0]', type=str,
                      help='select baselines via UV distance (e.g. [0,100] in meter)')

    parser.add_option('--DO_SAVE_AVERAGE_DATA', dest='dodatainfoutput',type=str,default='',
                      help='Generate dump of the averaged data via pickle file.')

    parser.add_option('--SWAPFIGURESIZE', dest='figureswap', action='store_true',default=False,
                      help='show progress bar ')

    parser.add_option('--DOPROGRESSBAR', dest='progbar', action='store_true',default=False,
                      help='show progress bar ')

    parser.add_option('--WORK_DIR', dest='cwd', default='',type=str,
                      help='Points to the working directory if output is produced (e.g. usefull for containers)')

    parser.add_option('--DOPLTFIG', dest='dopltfig', action='store_true',default=False,
                      help='Plot Figure instead of printing.')

    # ----

    (opts, args)         = parser.parse_args()

     
    if opts.msfile == None:        
        parser.print_help()
        sys.exit()


    # set the parmaters
    #
    cwd                 = opts.cwd        # used to write out information us only for container
    MSFN                = opts.msfile
    data_type           = opts.datacolumn
    field               = opts.field_id
    scan_num            = opts.scan_num
    dobslwfspec         = opts.dobslwfspec
    doavgspec           = opts.doavgspec
    chnslide            = opts.chnslide
    select_bsl          = opts.select_bsl
    select_ant          = opts.select_ant
    select_uvdis        = opts.select_uvdis
    select_spwd         = int(opts.select_spwd)

    dodatainfoutput     = opts.dodatainfoutput

    pltf_marker         = opts.pltf_marker
    showparameter       = opts.showparameter
    dosort_versus_uvdis = opts.dosortuvdis
    dofigureswap        = opts.figureswap
    doshowprogressbar   = opts.progbar
    
    dopltfig            = opts.dopltfig   # default setting always plotting into file if plotting is activated

    # ------------------------------------------------------------------------------
    

    # Source and Field Information
    #
    msource_info = INFMS.ms_source_info(MSFN)
    fie_sour     = INFMS.ms_field_info(MSFN)



    if data_type == 'DATA' or data_type == 'CORRECTED_DATA' or data_type == 'MODEL':
        showparameter = showparameter
    else:
        showparameter = data_type


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
        print('\t- FIELD_ID ',field,' unkown')
        print('\t- Please define FIELD_ID')
        for fi in fie_sour:
            print('\t\t--FIELD_ID ',fi,' ',fie_sour[fi])
        print('\n')
        sys.exit(-1)



    # define the scan and overwrites the field number
    #
    set_scan  = -1
    if scan_num != -1:
        tot_scans = 0
        print('\nSelect scan number ',scan_num)
        for so in msource_info:
            tot_scans += len(msource_info[so]['SCAN_ID'])
            if scan_num in msource_info[so]['SCAN_ID']:
                field = msource_info[so]['SOURCE_ID'][0]
                scans = msource_info[so]['SCAN_ID']
                set_scan = scan_num
                print('\t - use FIELD_ID ',field,' ',so)

        if tot_scans < scan_num:
            print('\t--Caution seems that input scan number is to high!')
        if set_scan == -1:
            print('\t--Caution that scan seems not to be in the MS-set!')
            sys.exit(-1)

    # check which data column to use
    if data_type != 'DATA':
        if INFMS.ms_check_col(MSFN,data_type) == -1:
            print('\t- ',data_type,' not present in dataset')
            sys.exit(-1)


    # Add the source name of the plotfiles
    #
    pltf_marker += source_name+'_'

    
    # get the baseline information
    #
    ms_bsls         = np.array(INFMS.ms_baselines(MSFN,tabs='FEED'))  # provides antenna ID values
    ms_bsls_antname = INFMS.ms_baselines(MSFN,tabs='ANTENNA')

    #
    if np.cumprod(ms_bsls.shape)[-1] == 0:
        print('Caution no baselines defined -- ABORT')
        sys.exit(-1)


    # get bsl length 
    #
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
    #
    if len(spwd_idx) > 1:
        if select_spwd == -1:
            print('Dataset consist of various spectra windows: ',spwd_idx,' use all')
        else:
            print('Dataset consist of various spectra windows: ',spwd_idx,' select ',select_spwd)
        if int(select_spwd) > spwd_idx[-1]:
            print('SPWD does not exsist!!!')
            sys.exit(-1)
    else:
        select_spwd = -1


    # select data based on baselines
    #
    if len(select_bsl) > 4:
        print('\t\t-- select on baselines')
        sel_bsl       = eval(select_bsl)
        #
        bsl_index_new = []
        #
        for bsl in bsl_index:
            for idx_bsl,sbsel in enumerate(sel_bsl):
                if ( ms_bsls_antname[bsl][0] == sbsel[0] and ms_bsls_antname[bsl][1] == sbsel[1] ):
                    bsl_index_new.append(bsl)

        bsl_index = bsl_index_new
        
        if len(bsl_index) == 0:
            print('Baseline not present in dataset: ',sel_bsl)
            sys.exit(-1)


    # select data based on antenna
    #
    if len(select_ant) > 2:
        print('\t\t-- select on antenna')
        sel_ant       = eval(select_ant)
        bsl_index_new = []
        #
        for bsl in bsl_index:
            for sant in sel_ant:
                if ms_bsls_antname[bsl][0] == sant  or ms_bsls_antname[bsl][1] == sant:
                    bsl_index_new.append(bsl)
                
        bsl_index = bsl_index_new

        if len(bsl_index) == 0:
            print('Antenna not present in dataset: ',sel_ant)
            print('use',INFMS.ms_unique_antenna(MSFN,tabs='ANTENNA'))
            sys.exit(-1)


    # select data based on uvdistance
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

    if data_type == 'MODEL_DATA' and get_model_data == -1:
        print('no model data present')
        sys.exit(-1)

    # check if CORRECTED DATA is present 
    get_corr_data = INFMS.ms_check_col(MSFN,'CORRECTED_DATA')

    if data_type == 'CORRECTED_DATA' and get_corr_data == -1:
        print('no corecetd data present')
        sys.exit(-1)
  

    # get spectral data averaged over baselines 
    #
    dyn_specs = INFMS.ms_average_data(MSFN,field,scan_num,np.array(ms_bsls)[np.array(bsl_index)],data_type,showparameter,print_info=doshowprogressbar)


    # save the entire dataset 
    #
    if len(dodatainfoutput) > 0:

        now = datetime.today().isoformat()
        
        pickle_data = {}
        pickle_data['type']            = 'SPECTRUM'
        pickle_data['data_type']       = data_type
        pickle_data['dyn_specs']       = dyn_specs
        pickle_data['scan_num']        = set_scan
        pickle_data['showparameter']   = showparameter
        pickle_data['corr']            = stokes
        pickle_data['source_name']     = source_name
        pickle_data['field']           = field
        pickle_data['baselines_id']    = np.array(ms_bsls)[np.array(bsl_index)]
        pickle_data['baselines_name']  = np.array(ms_bsls_antname)[np.array(bsl_index)]
        pickle_data['select_spwd']     = select_spwd
        pickle_data['MSFN']            = MSFN
        pickle_data['produced']        = str(now)

        picklename = cwd + pltf_marker +dodatainfoutput+'_pickle'
            
        INFMS.saveparameter(picklename,'WFDATA',pickle_data)

    
    # merge the multi-sw datasets
    #
    select_freq = []
    select_time = []
    #
    
    # this is the old one dealing with concat data of s0 and s4
    # concat_data,concat_datastd,concat_data_flag,concat_freq,concat_time = INFMS.combine_averaged_data(dyn_specs,select_freq,select_time,select_spwd,print_info=True)

    concat_data,concat_datastd,concat_data_flag,concat_freq,concat_time,concat_freq_per_sw,concat_chan_per_sw \
      = WZKPL.combine_averaged_data(dyn_specs,select_freq,select_time,select_spwd,print_info=False)

    # HRK comment in case there is a concatenated dataset (s0,s4)
    #
    for m in range(len(concat_data)):
        #
        # Prepare the data and do the plotting 
        #

        for st, sto in enumerate(stokes):

            # get the data
            #
            dyn_spec            = np.array(concat_data[m])[:,:,st]
            dyn_spec_std        = np.array(concat_datastd[m])[:,:,st]
            dyn_spec_avmask     = np.array(concat_data_flag[m])[:,:,st]

            # determine threshold used to init first mask
            #
            threshold = np.median(dyn_spec_avmask.std(axis=0))

            final_mask = mask = RFIM.mask_true_false(dyn_spec_avmask,threshold).astype(bool)


            # masked the waterfall data
            #
            dyn_spec_masked     = np.ma.masked_array(dyn_spec,mask=final_mask)
            dyn_spec_masked_std = np.ma.masked_array(dyn_spec_std,mask=final_mask)
            #
            # =========================================


            # determine and masked the spectrum data
            #
            avg_dynspec    = dyn_spec_masked.mean(axis=0)
            avg_dynspecstd = dyn_spec_masked_std.mean(axis=0)
            avg_final_mask = final_mask.mean(axis=0)

            # use a threshold to mask the data
            #
            nf_mask        = RFIM.mask_true_false(avg_final_mask,threshold=np.median(final_mask.std(axis=0)))

            # mask spectrum
            #
            spectrum_masked     = np.ma.masked_array(avg_dynspec,mask=nf_mask)
            spectrum_masked_std = np.ma.masked_array(avg_dynspecstd,mask=nf_mask)
            #
            # =========================================


            # WATERFALL plot
            #
            if dobslwfspec == True:

                # print('\n=== Average Waterfall Spectrum plotting ===')
                #
                plt_filename    =  pltf_marker +showparameter+'_WATERFALL_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                WZKPL.plot_waterfall_spec(dyn_spec_masked,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type,showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)
                #
                plt_filename    =  pltf_marker +showparameter+'_STD_WATERFALL_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                WZKPL.plot_waterfall_spec(dyn_spec_masked_std,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type+'STD',showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)

            # SPECTRUM plot
            #
            if doavgspec == True:

                #print('\n=== Spectrum plotting ===')
                #
                plt_filename    =  pltf_marker +showparameter+'_SPECTRUM_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                WZKPL.spectrum_average(spectrum_masked,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type,showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)
                #
                plt_filename    =  pltf_marker +showparameter+'_SPECTRUM_STD_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                WZKPL.spectrum_average(spectrum_masked_std,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type+'STD',showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)


    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------



if __name__ == "__main__":
    main()


