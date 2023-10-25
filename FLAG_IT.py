# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
#
#
# This program use DASK-MS to apply flags produced by DYNAMIC_SPECTRUM_PICKLEPLOTANDATAMARKER 
# and can save or apply a CASA flag table 
# 
# Hope you enjoy it
# 
# --------------------------------------------------------------------
#
# == Procedure to flag the dataset ==
#
# 1) produce an average spectrum pickle file with DYNAMIC_SPECTRUM_PLOTTER.py using the setting 
#    --DO_SAVE_AVERAGE_DATA=
#
# 2) based on the averages a new fg mask can be produced by BASIC_SPEC_PICKLE_PLOT.py
#    --DOFLAGDATA --DO_SAVE_FLAG_MASK=
#
# 3) use that output to load into FLAG_IT.py as --FGMASK_FILE=
#
#
# In a singularity container you can run it like
#
# singularity exec --bind ${PWD}:/data HRK_CASA_6.5_DASK.simg python FLAG_IT.py --MS_FILE= --FGMASK_FILE= --WORK_DIR='/data/'
#
#
# --------------------------------------------------------------------
#
def main():

    import sys
    import shutil

    from optparse import OptionParser
    import numpy as np

    from daskms import xds_from_ms,xds_from_table,xds_to_table
    import dask
    import dask.array as da

    import DASK_MS_WERKZEUGKASTEN as INFMS


    # argument parsing
    #
    usage = "usage: %prog [options]\
            This programme can be used to plot and store an average (over baseline or antenna) waterfall spectrum"

    parser = OptionParser(usage=usage)


    parser.add_option('--MS_FILE', dest='msfile', type=str,
                      help='MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms')

    parser.add_option('--FGMASK_FILE', dest='fgfile', type=str,
                      help='pickle - FG mask file')

    parser.add_option('--FGMASKALLDATA', dest='doflagonbsls', action='store_false',default=True,
                      help='apply the FG mask to all baselines (ignoring selection)')

    parser.add_option('--ERASEALLFG', dest='erase_flag', action='store_true',default=False,
                      help='erase all the FG information')

    parser.add_option('--CASAFGTABNAME', dest='casafgtabfile', type=str,default='',
                      help='CASA FG table name')

    parser.add_option('--CASAFGSAVE', dest='docasasafefg', action='store_true',default=False,
                      help='save the FG CASA table (using casa flagmanager)')

    parser.add_option('--CASAFGRESTORE', dest='docasarestorefg', action='store_true',default=False,
                      help='restore FG table in MS file (using casa flagmanager)')

    parser.add_option('--WORK_DIR', dest='cwd', default='',type=str,
                      help='Points to the working directory if output is produced (e.g. usefull for containers)')

    parser.add_option('--PRTINFO', dest='prtinfo', action='store_true',default=False,
                      help='Print out information during processing')

   
    # ----

    (opts, args)         = parser.parse_args()

     
    if opts.msfile == None:        
        parser.print_help()
        sys.exit()


    # set the parmaters
    #
    cwd                 = opts.cwd        # used to write out information us only for container
    MSFN                = opts.msfile
    fgfile              = opts.fgfile
    #
    doflagonbsls        = opts.doflagonbsls
    #
    casafgtabfile       = opts.casafgtabfile
    docasasafefg        = opts.docasasafefg
    docasarestorefg     = opts.docasarestorefg
    #
    erase_flag          = opts.erase_flag
    #
    prtinfo             = opts.prtinfo  
    # ------------------------------------------------------------------------------
    

    if len(casafgtabfile) > 0:
        # save the current flag table
        #
        import casatasks
        import CAL2GC_lib as CLIB
        #
        msfile        = MSFN
        casa_fg_table = casafgtabfile

        havedone = ''
        if docasasafefg:
            casatasks.flagmanager(vis=msfile, mode='save', versionname=casa_fg_table)
            havedone = 'save'

        if docasarestorefg:
            casatasks.flagmanager(vis=msfile, mode='restore', versionname=casa_fg_table)
            havedone = 'restore'

        # store casa log file to current directory 
        #
        current_casa_log = CLIB.find_CASA_logfile(checkdir='HOME',homdir='')
        shutil.move(current_casa_log,cwd) 

        print('Used CASA flag manager to ',havedone,' ',casa_fg_table)
        sys.exit(-1)


    # ------------------------------------------------------------------------------
    # load the FG MASK
    #
    fgfilename = fgfile

    # load the flag mask 
    #
    pickle_data = INFMS.getparameter(fgfilename)

    pickle_keys = pickle_data.keys()

    for pk in pickle_keys:
        if pk == 'WFDATA':
            print('\n === You are using the wrong pickle file ===\n')
            print('\n === use DYNAMIC_SPECTRUM_PICKLE_PLTFLG.py  ===\n')
            sys.exit(-1)

    data_type      = pickle_data['FGDATA']['data_type']       
    flag_mask      = np.array(pickle_data['FGDATA']['flag_mask'])      
    timerange      = np.array(pickle_data['FGDATA']['timerange'])
    if doflagonbsls == True:
        doflagonbsls   = pickle_data['FGDATA']['flagonbsl']   
    bsls_ant_id    = pickle_data['FGDATA']['baselines_id'] 
    bsls_ant_name  = pickle_data['FGDATA']['baselines_name']
    set_scan       = pickle_data['FGDATA']['scan_num']
    showparameter  = pickle_data['FGDATA']['showparameter']
    stokes         = pickle_data['FGDATA']['corr']
    source_name    = pickle_data['FGDATA']['source_name']
    field          = pickle_data['FGDATA']['field']
    FGMSFILE       = pickle_data['FGDATA']['MSFN']
    produced       = pickle_data['FGDATA']['produced']
    # 
    # ------------------------------------------------------------------------------

    if prtinfo:
        print('\n === FLAGER INPUT === ')
        print('- use ',fgfile,' for flagging ')
        print('- produced     ',produced)
        print('- used MS File ',FGMSFILE)
        print('- flag field   ',field)
        print('- flag source  ',source_name)
        print('- Stokes       ',stokes)
        print('- flag on baseline selection ',doflagonbsls)

        if (MSFN.replace(cwd,'') == FGMSFILE.replace(cwd,'')) == False:
            print('\n Caution the input MS-file ',MSFN,' does not match FG mask file origin.')
            sys.exit(-1)


    # ------------------------------------------------------------------------------
    #
    #  APPLY THE FLAG TABLE
    #
    # get the overall information
    #
    msstab,msstab_kw,msstab_col =  xds_from_table(MSFN+'::DATA_DESCRIPTION',table_keywords=True,column_keywords=True)

    # get the frequency info
    #
    spwd_info   = xds_from_table(MSFN+'::SPECTRAL_WINDOW')
    spwds       = msstab[0].SPECTRAL_WINDOW_ID.data.compute()

    # optain the antenna information
    #
    ant_index   = INFMS.ms_unique_antenna(MSFN,tabs='FEED')
    ant_name    = INFMS.ms_unique_antenna(MSFN,tabs='ANTENNA')


    # Produce a list of xarray datasets backed by reads from columns on disk.
    # Here I have chosen to produce an independent dataset for each unique
    # combination of SCAN_NUMBER, FIELD_ID and DATA_DESC_ID.
    chunksize = 100000
    #
    xdsl = xds_from_ms(
        MSFN,chunks={'row':chunksize},
        columns=("FLAG", "DATA","ANTENNA1","ANTENNA2"),
        index_cols= (),
        group_cols=("SCAN_NUMBER","FIELD_ID","TIME", "DATA_DESC_ID")
    )


    # used for bookeeping
    applied_new_flag     = []
    not_applied_new_flag = []
    output_xdsl          = []
    #
    for xds in xdsl:

        data       = xds.DATA.data
        flags      = xds.FLAG.data

        fie_id     = xds.attrs['FIELD_ID']
        scan_no    = xds.attrs['SCAN_NUMBER']
        ddid       = xds.attrs['DATA_DESC_ID']                
        time       = xds.attrs['TIME']


        ant1       = xds.ANTENNA1.data
        ant2       = xds.ANTENNA2.data


        if fie_id == field:

            spwd_id    = msstab[0].SPECTRAL_WINDOW_ID.data[ddid].compute()

            if prtinfo:
                print('field:     ',fie_id)
                print('scan_no:   ',scan_no)
                print('ddid:      ',ddid)
                print('spwd:      ',spwd_id)
                print('time:      ',time)
                print('flagonbsl: ',doflagonbsls)

            # define baseline selection per time 
            #
            if doflagonbsls:
                sel_bsls  = np.zeros(xds.ANTENNA1.shape,dtype=bool)                        
                for bl in bsls_ant_id:

                    sel_ant1 = xds.ANTENNA1.data == bl[0]
                    sel_ant2 = xds.ANTENNA2.data == bl[1]

                    sel_bsl_single = da.logical_and(sel_ant1,sel_ant2).compute()                    
                    sel_bsls       = da.logical_or(sel_bsl_single,sel_bsls)
            else:            
                sel_bsls  = np.ones(xds.ANTENNA1.shape,dtype=bool)


            
            # this may need to be somewhere else
            #
            if len(flag_mask.shape) == 3:
                spwd_flag_mask = flag_mask
            else:

                spwd_flag_mask = flag_mask[spwd_id]

                if prtinfo:
                    print('Use mask for spwd: ',spwd_id)
                    print('mask shape       : ',spwd_flag_mask.shape)
                    print('org flag shape   : ',xds.FLAG.dims)
                    print('org flag shape   : ',xds.FLAG.shape)
                    print('availible bsl    : ',sel_bsls.shape)
                    #print('select bsl       : ',np.cumsum(sel_bsls.as_dtype(int))[-1])

            # search for the time in the flag mask
            #
            time_idx = np.where(timerange == time)

            if time_idx[0].size > 0:
                if prtinfo:
                    print('Found time stamp in FG MASK\n')

                new_flags           = np.zeros(xds.FLAG.shape,dtype=bool)                        

                if erase_flag == False:
                    new_flags[sel_bsls] = spwd_flag_mask[time_idx]
                    
                # bookkeeping 
                #
                applied_new_flag.append(time)

            else:
                if prtinfo:
                    print('NO time stamp found in FG MASK\n')

                new_flags = flags

                # bookkeeping 
                #
                not_applied_new_flag.append(time)

            # combine new flags and the pre-exsisting flags
            #
            if erase_flag:
                updated_flags = da.logical_or(new_flags,new_flags)
            else:
                if prtinfo:
                    print('Combine org and new flags\n')
                updated_flags = da.logical_or(flags, new_flags)

            # Replace the existing FLAG data variable with the updated version.
            # Note that we specify the dimensions of the data variable in
            # addition to the new values.
            updated_xds = xds.assign({"FLAG": (xds.FLAG.dims, updated_flags)})

            # Append the modified dataset to the list of outputs.
            output_xdsl.append(updated_xds)


    # ouptut_xdsl now contains a list of datasets which include the updated
    # flag values. We can now write these to disk.
    writes = xds_to_table(output_xdsl, MSFN, columns="FLAG")

    # As the above operations are lazy, nothing has actually been computed.
    # Finally, trigger the computation.
    da.compute(writes)


    if prtinfo:
        print('Availible time ',len(timerange),' applied FG ',len(applied_new_flag),' missed flags ',len(not_applied_new_flag),'hhh',len(output_xdsl))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

if __name__ == "__main__":
    main()


