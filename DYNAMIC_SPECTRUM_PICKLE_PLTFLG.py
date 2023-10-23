
# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
# The programme can be used to plot (1) and to produce a flag table for further processing
#
# (1) This program plots waterfall and averged spectra using the pickle dataset produced 
#     with the BASIC_SPECTRUM_PLOTTER.py with the setting --DO_SAVE_AVERAGE_DATA=
#
# (2) This program flag on the waterfall spectrum and a flag mask can be stored for further 
#     processing
#
# Required a python environment with the following packages: numpy,scipy, matplotlib
#
# Hope you enjoy it
# 
# --------------------------------------------------------------------
#
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#
import WERKZEUGKASTEN_PLOT_LIB as WZKPL
import RFI_MITILIB as RFIM



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


def saveparameter(filename,para,data):
    """
    purpose is to save some data into a file for testing issues
    """
    import pickle
    pfile  = open(filename+'.py','wb')
    ddata = {}
    ddata[para]=data
    pickle.dump(ddata,pfile)
    pfile.close()
    return(filename+'.py')


def main():


    from optparse import OptionParser


    # argument parsing
    #
    usage = "usage: %prog [options]\
            This programme can be used to plot and store a flag mask based on the average (over baseline or antenna) waterfall spectrum"

    parser = OptionParser(usage=usage)


    parser.add_option('--AVG_FILE', dest='avgfile', type=str,
                      help='pickle - file of an averaged MS file')

    parser.add_option('--DOPLTSTDDATA', dest='doplotstddata', action='store_true', default=False,
                      help='plot the stddata instead of the data')
    
    parser.add_option('--DOUSEDATA', dest='dousestddata', action='store_false', default=True,
                      help='use the data instead of the stddata to produce the flag mask')

    parser.add_option('--DOBSLWATERFALLSPEC', dest='dobslwfspec', action='store_true', default=False,
                      help='produce waterfall spectrum')

    parser.add_option('--DOPLOTAVGSPECTRUM', dest='doavgspec', action='store_true', default=False,
                      help='produce an average spectrum')

    # flag the data
    parser.add_option('--DOFLAGDATA', dest='doflagthedata', action='store_true',default=False,
                      help='Do flag the data')

    parser.add_option('--FLAG_CHAN', dest='flagbyhand', default='[[]]', type=str,
                      help='flag channel ranges by hand (e.g. [[55,55],[10,30]])')

    parser.add_option('--DO_SAVE_FLAG_MASK', dest='dosaveflagmask',type=str,default='',
                      help='Generate dump of the averaged data via pickle file.')

    parser.add_option('--PRE_FLAG_MASK', dest='fgmaskfile', type=str,default='',
                      help='pickle - file of the mask FG file')

    # some internal orga stuff
    parser.add_option('--SWAPFIGURESIZE', dest='figureswap', action='store_false',default=True,
                        help='show progress bar ')

    parser.add_option('--PLOTFILEMARKER', dest='pltf_marker', default='PLT_',type=str,
                      help='add file indicator in front of the file [defaut = PLT_]')


    parser.add_option('--WORK_DIR', dest='cwd', default='',type=str,
                      help='Points to the working directory if output is produced (e.g. usefull for containers)')


    parser.add_option('--PRTINFO', dest='prtinfo', action='store_true',default=False,
                      help='Print out information during processing')

    # ----

    (opts, args)         = parser.parse_args()

     
    if opts.avgfile == None:        
        parser.print_help()
        sys.exit()


    # set the parmaters
    #
    filename            = opts.avgfile

    dousestddata        = opts.dousestddata
    doplotstddata       = opts.doplotstddata

    dobslwfspec         = opts.dobslwfspec
    doavgspec           = opts.doavgspec

    fgfilename          = opts.fgmaskfile

    #
    doflagthedata       = opts.doflagthedata
    flagbyhand          = opts.flagbyhand
    dosaveflagmask      = opts.dosaveflagmask
    #
    dofigureswap        = opts.figureswap
    pltf_marker         = opts.pltf_marker
    cwd                 = opts.cwd        # used to write out information us only for container
    prtinfo             = opts.prtinfo  

    # hard coded 
    #
    dopltfig            = False            # default setting always plotting into file if plotting is activated
    doflagonbsls        = True            # this setting can be overwritten in the actual FLAG_IT programe
    # ------------------------------------------------------------------------------
    

    # in case flaging is used
    #
    casa_flag_coordinates = []


    if len(fgfilename):
        # load the flag mask 
        #
        pickle_data_mask = getparameter(fgfilename)

        pickle_keys = pickle_data_mask.keys()

        for pk in pickle_keys:
            if pk == 'WFDATA':
                print('\n === You are using the wrong pickle file ===\n')
                print('\n === use DYNAMIC_SPECTRUM_PICKLE_PLTFLG.py  ===\n')
                sys.exit(-1)

        data_type_m      = pickle_data_mask['FGDATA']['data_type']       
        flag_mask_m      = pickle_data_mask['FGDATA']['flag_mask'] 

        fg_wf_mask      = pickle_data_mask['FGDATA']['fg_wf_mask'] 

        timerange_m      = pickle_data_mask['FGDATA']['timerange']
        doflagonbsls_m   = pickle_data_mask['FGDATA']['flagonbsl']   
        bsls_ant_id_m    = pickle_data_mask['FGDATA']['baselines_id'] 
        bsls_ant_name_m  = pickle_data_mask['FGDATA']['baselines_name']
        set_scan_m       = pickle_data_mask['FGDATA']['scan_num']
        showparameter_m  = pickle_data_mask['FGDATA']['showparameter']
        stokes_m         = pickle_data_mask['FGDATA']['corr']
        source_name_m    = pickle_data_mask['FGDATA']['source_name']
        field_m          = pickle_data_mask['FGDATA']['field']
        FGMSFILE_m       = pickle_data_mask['FGDATA']['MSFN']
        produced_m       = pickle_data_mask['FGDATA']['produced']

        if prtinfo:
            print('\n === FLAGER INPUT === ')
            print('- use ',fgfilename,' for flagging ')
            print('- produced     ',produced_m)
            print('- used MS File ',FGMSFILE_m)
            print('- flag field   ',field_m)
            print('- flag source  ',source_name_m)
            print('- Stokes       ',stokes_m)
            print('- flag on baseline selection ',doflagonbsls_m)

        # 
        # ------------------------------------------------------------------------------



    # load the averaged data
    #
    pickle_data = getparameter(filename)

    if  dobslwfspec == True or doavgspec == True or doflagthedata == True:


        pickle_keys = pickle_data.keys()

        for pk in pickle_keys:
            if pk == 'FGDATA':
                print('\n === You are using the FG pickle file ===\n')
                sys.exit(-1)



        data_type      = pickle_data['WFDATA']['data_type']
        dyn_specs      = pickle_data['WFDATA']['dyn_specs']
        scan_num       = pickle_data['WFDATA']['scan_num']
        select_spwd    = pickle_data['WFDATA']['select_spwd']
        showparameter  = pickle_data['WFDATA']['showparameter']
        stokes         = pickle_data['WFDATA']['corr']
        source_name    = pickle_data['WFDATA']['source_name']
        field          = pickle_data['WFDATA']['field']
        bsls_ant_id    = pickle_data['WFDATA']['baselines_id']
        bsls_ant_name  = pickle_data['WFDATA']['baselines_name']
        usedmsfile     = pickle_data['WFDATA']['MSFN']
        produced       = pickle_data['WFDATA']['produced']
        
        select_freq    = []
        select_time    = []
        set_scan       = scan_num

        if set_scan == -1:
                skeys = dyn_specs.keys()
                tot_num_spwd =  dyn_specs[list(skeys)[0]]['INFO_SPWD']
        else:
                tot_num_spwd =  dyn_specs[str(set_scan)]['INFO_SPWD']


        # Add the source name of the plotfiles
        #
        pltf_marker += source_name+'_'


        # merge the multi-sw datasets
        #
        select_freq = []
        select_time = []
        #
        concat_data,concat_datastd,concat_data_flag,concat_freq,concat_time,concat_freq_per_sw,concat_chan_per_sw \
          = WZKPL.combine_averaged_data(dyn_specs,select_freq,select_time,select_spwd,print_info=False)


        if len(concat_data) > 1:
                print('Need to look at the data structure seems to be from multiple times')
                sys.exit(-1)

        flag_data = {}

        if prtinfo:

                if  dobslwfspec == True or doavgspec == True:
                        print('\n=== Average Waterfall/Spectrum plotting ===\n')
                else:
                        print('\n=== Average Spectrum flagging  ===\n')

        for m in range(len(concat_data)):

            # leave for the time being to check the data
            #
            #print('WF shapes')
            #print('array ',np.array(concat_data[m]).shape)
            #print('time ',np.array(concat_time[m]).shape)
            #print('freq ',np.array(concat_freq[m]).shape)

            flag_data[str(m)] = {}

            for st, sto in enumerate(stokes):

                # get the data
                #
                dyn_spec            = np.array(concat_data[m])[:,:,st]
                dyn_spec_std        = np.array(concat_datastd[m])[:,:,st]
                dyn_spec_avmask     = np.array(concat_data_flag[m])[:,:,st]

                # determine threshold used to init first mask
                #
                threshold = np.median(dyn_spec_avmask.std(axis=0))

                if doflagthedata:

                    if dousestddata:
                        datatoflag     = dyn_spec_std
                    else:
                        datatoflag     = dyn_spec

                    #
                    # Flagging the data
                    #
                    sigma          = 3
                    stats_type     = 'mean'
                    smooth_kernels = ['robx','roby','scharrx','scharry','sobelx','sobely','canny','prewittx','prewitty']
                    percentage     = 50
                    #
                    final_mask  = RFIM.flag_data(datatoflag,dyn_spec_avmask,sigma,stats_type,percentage,smooth_kernels,threshold,flagbyhand)
                    # ==============================


                    # check the impact
                    f_mask_info = RFIM.flag_impact(final_mask,dyn_spec_avmask)
                    #
                    if prtinfo:
                        print('Flag Info [',sto,'] ',f_mask_info[2]/f_mask_info[0]*100,'%')

                    # create the flagging info for further Flagging 
                    # 
                    fg_mask_spwd,fg_concat_freq_spwd,fg_concat_time,fg_concat_freq_per_sw_spwd = RFIM.mask_into_spwd(final_mask,concat_time[m],concat_freq[m],concat_freq_per_sw[m],concat_chan_per_sw[m])
                    #
                    flag_data[str(m)][sto] = {}                    
                    flag_data[str(m)][sto]['mask_spwd']                  = fg_mask_spwd
                    flag_data[str(m)][sto]['fg_concat_freq_spwd']        = fg_concat_freq_spwd
                    flag_data[str(m)][sto]['fg_concat_time']             = fg_concat_time
                    flag_data[str(m)][sto]['fg_concat_freq_per_sw_spwd'] = fg_concat_freq_per_sw_spwd
                    flag_data[str(m)][sto]['fg_info']                    = f_mask_info[2]/f_mask_info[0]*100
                    flag_data[str(m)][sto]['final_mask']                 = final_mask
                else:
                    final_mask = RFIM.mask_true_false(dyn_spec_avmask,threshold).astype(bool)


                if len(fgfilename) > 0:
                    final_mask = fg_wf_mask[str(m)][sto]['final_mask']

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

                    plt_filename    =  pltf_marker +showparameter+'_WATERFALL_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                    WZKPL.plot_waterfall_spec(dyn_spec_masked,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type,showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)
                    #
                    if doplotstddata:
                        plt_filename    =  pltf_marker +showparameter+'_STD_WATERFALL_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                        WZKPL.plot_waterfall_spec(dyn_spec_masked_std,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type+'STD',showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)


                # SPECTRUM plot
                #
                if doavgspec == True:

                    # just to play with if needed
                    #
                    #get_outer_bondaries,interp_avg_dynspecmean = RFIM.interpolate_mask_data(np.ma.masked_array(concat_freq[m],nf_mask),np.ma.masked_array(avg_dynspec,nf_mask),concat_freq[m],nf_mask)
                    #sm_data = RFIM.convolve_1d_data(interp_avg_dynspecmean,smooth_type='wiener',smooth_kernel=101)
                    #plt.plot(concat_freq[m][get_outer_bondaries[0]:get_outer_bondaries[1]],interp_avg_dynspecmean,'black')
                    #plt.plot(concat_freq[m][get_outer_bondaries[0]:get_outer_bondaries[1]],sm_data,'-r')
                    #plt.plot(np.ma.masked_array(concat_freq[m],nf_mask),np.ma.masked_array(avg_dynspec,nf_mask),'-r')

                    plt_filename    =  pltf_marker +showparameter+'_SPECTRUM_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                    WZKPL.spectrum_average(spectrum_masked,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type,showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)

                    if doplotstddata:

                        plt_filename    =  pltf_marker +showparameter+'_SPECTRUM_STD_'+'SPWD_'+str(select_spwd)+'_SCAN_'+str(set_scan)+'_'+sto+'_'+str(m)
                        WZKPL.spectrum_average(spectrum_masked_std,np.array(concat_time[m]),np.array(concat_freq[m]),select_spwd,data_type+'STD',showparameter,sto,source_name,plt_filename,cwd,dofigureswap,dopltfig)




    # save the masked file
    #
    if len(dosaveflagmask) > 0:

        # The averaged flagging is merged here
        #
            
        concat_index          = 0   # just in case if data structure changes in the future

        if prtinfo:
            print('\nCAUTION HARDCODED INDEX concat_index, may want to change this in the future')

        merge_spwd_stokes     = []
        timerange_spwd_stokes = []
        for st in stokes:
            merge_spwd_stokes.append(flag_data[str(concat_index)][st]['mask_spwd'])
            timerange_spwd_stokes.append(flag_data[str(concat_index)][st]['fg_concat_time'])

        # rudimental check if timerange match
        #
        if len(timerange_spwd_stokes) > 1:
            check_time = np.nonzero(np.array(timerange_spwd_stokes[0]) - np.array(timerange_spwd_stokes[1]))
            if check_time[0].size > 1:
                print('Strange timeanges, data does not fit')
                sys.exit(-1)

            timerange = timerange_spwd_stokes[0]

        # Data structure is changed from 
        # 
        # if only 1 spwd
        # [stokes, spwd, time, channel] --> [time,channel,stokes]
        #
        # if multiple spwd
        # [stokes, spwd, time,channel] --> [spwd,time,channel,stokes]
        #
        # this is the natural MS and used in the dask-ms
        #
        flag_mask = np.moveaxis(merge_spwd_stokes,0,-1)

        if prtinfo:
                print('\n\t -Change structure of FG mask from ',np.array(merge_spwd_stokes).shape,' to ',flag_mask.shape)

        now = datetime.today().isoformat()

        # save the flagging mask
        #
        pickle_data = {}
        pickle_data['type']            = 'FGMASK'
        pickle_data['data_type']       = 'FLAG'
        pickle_data['dousestddata']    = dousestddata
        pickle_data['fg_wf_mask']      = flag_data
        pickle_data['flag_mask']       = flag_mask
        pickle_data['timerange']       = timerange
        pickle_data['flagonbsl']       = doflagonbsls
        pickle_data['baselines_id']    = bsls_ant_id
        pickle_data['baselines_name']  = bsls_ant_name
        pickle_data['scan_num']        = set_scan
        pickle_data['showparameter']   = showparameter
        pickle_data['corr']            = stokes
        pickle_data['source_name']     = source_name
        pickle_data['field']           = field
        pickle_data['MSFN']            = usedmsfile
        pickle_data['produced']        = str(now)


        picklename = cwd + pltf_marker +dodatainfoutput+'_pickle'
        #picklename = cwd + dosaveflagmask+'_pickle'

        saveparameter(picklename,'FGDATA',pickle_data)



if __name__ == "__main__":
    main()
