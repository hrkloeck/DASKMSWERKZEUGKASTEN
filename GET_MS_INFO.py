# HRK 2021
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
#
# This program extracts information of a measurement set (MS) using 
# DASK-MS
#
# --------------------------------------------------------------------

import sys
from optparse import OptionParser
import json

import numpy as np
from astropy.time import Time

import DASK_MS_WERKZEUGKASTEN as INFMS


def main():

    # argument parsing
    #
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)



    parser.add_option('--MS_FILE', dest='msfile', type=str,
                      help='MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms')

    parser.add_option('--WORK_DIR', dest='cwd', default='',type=str,
                      help='Points to a working directory (e.g. usefull for containers)')


    parser.add_option('--DOINFO_TAB', dest='getinfotab', default='',type=str,
                      help='Show table info [default ALL tables, else use table name]')


    parser.add_option('--DO_MS_INFO_JSON', dest='dodatainfoutput',default='',type=str,
                      help='Output file name in JSON format.')


    parser.add_option('--NOT_PRINT_MS_INFO', dest='doprtdatainfo', action='store_false', default=True,
                      help='Stop printing MS info. Useful in pipelines')



    # ----

    (opts, args)         = parser.parse_args()

    if opts.msfile == None:
    
        parser.print_help()
        sys.exit()


    # set the parmaters
    #
    cwd             = opts.cwd
    MSFN            = cwd + opts.msfile
    getinfotabs     = opts.getinfotab
    doprtdatainfo   = opts.doprtdatainfo
    dodatainfoutput = opts.dodatainfoutput



    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    dodatainfo = 1

    # ----------------------------------------
    # print out table information of the MS
    # ----------------------------------------
    if len(getinfotabs) > 0: 

        table_info = INFMS.ms_tables(MSFN,doinfo='')

        if list(table_info.keys()).count(getinfotabs) > 0:
                INFMS.ms_tables(MSFN,doinfo=getinfotabs)
        else:
 
            print('\nMS - file: ',MSFN, '\n\nAvailible tables: ',list(table_info.keys()))

            for k in table_info:
                if table_info[k] > 0:
                    INFMS.ms_tables(MSFN,doinfo=k)
        
        dodatainfo = -1


    # ----------------------------------------
    # print out information of observation 
    # ----------------------------------------

    if dodatainfo > 0:

        # get general information
        msinfo = INFMS.ms_obs_info(MSFN)

        array_type = 'HOMOGENIOUS'
        if len(np.unique(msinfo['DISH_DIAMETER'])) > 1:
            array_type = 'INHOMOGENIOUS'

        number_of_antennas = len(msinfo['ANTS'])

        # get baseline length 
        bsl_length = INFMS.ms_baselines_length(MSFN)

        # Field Information
        field_info = INFMS.ms_field_info(MSFN)
        
        # Source Information
        msource_info = INFMS.ms_source_info(MSFN)

        # Polarisation Information
        mspol_info = INFMS.ms_pol_info(MSFN)

        if len(mspol_info['STOKES']) > 1:
            n_pol = 2

        # Frequency Information
        msfreq_info = INFMS.ms_freq_info(MSFN)


        # Determine addition frequency information
        msfreq_key = msfreq_info.keys()
        freq_range = []
        chan_freq_width = []
        total_number_of_channels = 0
        for fky in msfreq_key:
            freq_range.append(msfreq_info[fky]['CHAN_FREQ'][0])
            freq_range.append(msfreq_info[fky]['CHAN_FREQ'][-1])
            total_number_of_channels += msfreq_info[fky]['NUM_CHAN']
            chan_freq_width.append(min(msfreq_info[fky]['CHANWIDTH']))
            chan_freq_width.append(max(msfreq_info[fky]['CHANWIDTH']))


        bandwidth   = max(freq_range) - min(freq_range)
        center_freq = (max(freq_range) + min(freq_range))/2.

        # Determine addition time information
        sinfo_keys        = msource_info.keys()
        int_time_so       = {}
        exposure_time_so  = {}
        delta_int_time_so = {}
        time_range        = []
        noscans           = 0

        for so in sinfo_keys:
            int_time_so[so] = []
            delta_int_time_so[so] = []
            exposure_time_so[so] = []

            for sp in range(len(msource_info[so]['SCANTIMES'])):
                tmin = msource_info[so]['SCANTIMES'][sp][0] - msource_info[so]['SCANEXPOSURE'][sp][0]/2.0
                tmax = msource_info[so]['SCANTIMES'][sp][-1] + msource_info[so]['SCANEXPOSURE'][sp][-1]/2.0
                int_time_so[so].append([tmin,tmax])
                delta_int_time_so[so].append(tmax-tmin)
                exposure_time_so[so].append(min(msource_info[so]['SCANEXPOSURE'][sp]))
                exposure_time_so[so].append(max(msource_info[so]['SCANEXPOSURE'][sp]))
                time_range.append([tmin,tmax])
                noscans += 1


        inttimepersource = {}
        inttimes         = []
        exptimes         = []
        for so in sinfo_keys:
            inttimepersource[so] = np.cumsum(delta_int_time_so[so])[-1]
            inttimes.append(int(np.cumsum(delta_int_time_so[so])[-1]))
            exptimes += exposure_time_so[so]

        # Determine FoV
        FOV_calc = []
        FOV_calc.append(min(INFMS.beams(msinfo['DISH_DIAMETER'],min(freq_range),type='FoV')))
        FOV_calc.append(min(INFMS.beams(msinfo['DISH_DIAMETER'],max(freq_range),type='FoV')))
        FOV_calc.append(max(INFMS.beams(msinfo['DISH_DIAMETER'],min(freq_range),type='FoV')))
        FOV_calc.append(max(INFMS.beams(msinfo['DISH_DIAMETER'],max(freq_range),type='FoV')))


        # Determine angular resolution
        ang_res_calc = []
        ang_res_calc.append(min(INFMS.beams(bsl_length,max(freq_range),type='res')))
        ang_res_calc.append(min(INFMS.beams(bsl_length,min(freq_range),type='res')))

        # Determine image size
        cell_size  = min(ang_res_calc) / 3.
        image_size = max(FOV_calc)/cell_size


        # Determine baseline sensitivity

        # ------ THIS NEEDS TO BE DONE BETTER e.g. YAML file !!!!
        teleskopinfo = INFMS.telescope_array_info()

        T_sys  = teleskopinfo[0]
        eta_a  = teleskopinfo[1]
        # ---- 


        if array_type == 'HOMOGENIOUS':

            SEFD      = INFMS.SEFD(np.unique(msinfo['DISH_DIAMETER']),T_sys,eta_a)

            bsl_sens = []
            bsl_sens.append(INFMS.baseline_sensitivity(SEFD,SEFD,bandwidth,min(np.unique(exptimes)),eta_s=1))
            bsl_sens.append(INFMS.baseline_sensitivity(SEFD,SEFD,bandwidth,max(np.unique(exptimes)),eta_s=1))

            bsl_sens  = np.unique(bsl_sens)

            bsl_sens_jy = np.array(bsl_sens) * 1E26

            image_sens = []
            for t_obs in inttimes:
                image_sens.append(INFMS.image_sensitivity(SEFD,number_of_antennas,t_obs,bandwidth,n_pol,eta_s=1))

            image_sens_jy = np.array(image_sens) * 1E26

        else:
            print('Array type is: ',array_type)
            print('Can not determine image sensitivities please add telescope information')
            image_sens_jy = -1


        MS_FULL_INFO = {}
        MS_FULL_INFO.update({'TELESCOPE_NAME':msinfo['TELESCOPE_NAME']})
        MS_FULL_INFO.update({'PROJECT_ID':msinfo['PROJECT']})
        MS_FULL_INFO.update({'TIMERANGE_START_utc':Time(np.amin(np.array(time_range))/(24. * 3600.),scale='utc',format='mjd').iso})
        MS_FULL_INFO.update({'TIMERANGE_STOP_utc':Time(np.amax(np.array(time_range))/(24. * 3600.),scale='utc',format='mjd').iso})

        MS_FULL_INFO.update({'NUM_SCANS':noscans})
        MS_FULL_INFO.update({'NUM_ANT':len(msinfo['ANTS'])})
        MS_FULL_INFO.update({'ANT_DIAMETER_m':list(np.unique(msinfo['DISH_DIAMETER']))})

        MS_FULL_INFO.update({'ARRAY_TYPE':array_type})
        MS_FULL_INFO.update({'FoV_deg':[min(FOV_calc),max(FOV_calc)]})

        MS_FULL_INFO.update({'BSL_LENGTH_m':[min(bsl_length),max(bsl_length)]})

        MS_FULL_INFO.update({'ANG_RESOLUTION_deg':[min(ang_res_calc),max(ang_res_calc)]})

        MS_FULL_INFO.update({'IMAGE_SIZE_pix':image_size})

        MS_FULL_INFO.update({'CELLSIZE_deg_per_pix':cell_size})

        MS_FULL_INFO.update({'STOKES':mspol_info['STOKES']})

        MS_FULL_INFO.update({'NUM_SPWD':len(msfreq_key)})
        
        MS_FULL_INFO.update({'FREQ_RANGE_hz':[min(freq_range),max(freq_range)]})
        MS_FULL_INFO.update({'CENTER_FREQ_hz':center_freq})
        MS_FULL_INFO.update({'BAND_WIDTH_hz':bandwidth})

        MS_FULL_INFO.update({'NUM_CHANNEL':int(total_number_of_channels)})

        MS_FULL_INFO.update({'CHANNEL_WIDTH_hz':[min(chan_freq_width),max(chan_freq_width)]})

        MS_FULL_INFO.update({'SOURCES':list(msource_info.keys())})

        MS_FULL_INFO.update({'FIELD_ID':list(field_info.keys())})

        MS_FULL_INFO.update({'TIME_PER_SOURCE_s':list(inttimes)})

        MS_FULL_INFO.update({'INTEGRATION_TIME_s':[min(np.unique(exptimes)),max(np.unique(exptimes))]})

        MS_FULL_INFO.update({'IMAGE_SENSITIVITY_jy':image_sens_jy.flatten().tolist()})

        MS_FULL_INFO.update({'BSL_SENSITIVITY_jy':bsl_sens_jy.flatten().tolist()})


        if doprtdatainfo:
        
            print('\n')
            print('telescope                   :  ',msinfo['TELESCOPE_NAME'])        
            print('project ID                  :  ',msinfo['PROJECT'])

            print('observation timerange (UTC) :  ', Time(np.amin(np.array(time_range))/(24. * 3600.),scale='utc',format='mjd').iso,\
                      ' --- ',Time(np.amax(np.array(time_range))/(24. * 3600.),scale='utc',format='mjd').iso)

            print('number of individual scans  :  ',noscans)

            print('number of antennas          :  ',len(msinfo['ANTS']))
            print('antenna diameter        [m] :  ',np.unique(msinfo['DISH_DIAMETER']))
            print('array type                  :  ',array_type)
            print('field of view  (FoV)  [deg] :   [',np.round(min(FOV_calc),2),',',np.round(max(FOV_calc),2),']')
            print('baseline length         [m] :   [',np.round(min(bsl_length),2),',',np.round(max(bsl_length),2),']')  
            print('angular resolution [arcsec] :   [',np.round(min(ang_res_calc)*3600,2),',',np.round(max(ang_res_calc)*3600,2),']')
            print('imagesize           [pixel] :  ',np.round(image_size,2))
            print('cellsize     [arcsec/pixel] :  ',np.round(cell_size*3600,6))
            print('polarisation property       :  ',mspol_info['STOKES'])
            print('spectral windows     [SPWD] :  ',len(msfreq_key))

            print('total frequency range  [Hz] :   %e '%min(freq_range),' --   %e'%max(freq_range))

            print('center frequency       [Hz] :   %e '%center_freq)

            print('total bandwidth        [Hz] :   %e '%bandwidth)

            print('total number of channels    :  ',total_number_of_channels)

            print('channels width         [Hz] :   [%e'%min(chan_freq_width),', %e'%max(chan_freq_width),']')

            print('observed sources            :  ',list(msource_info.keys()))

            print('field id                    :  ',list(field_info.keys()))

            print('time per source         [s] :  ',list(inttimes))

            print('integration time        [s] :   [',np.round(min(np.unique(exptimes)),2),',',np.round(max(np.unique(exptimes)),2),']')  

            print('image sensitivity      [Jy] :  ',image_sens_jy.flatten().tolist())

            print('baseline sensitivity   [Jy] :  ',bsl_sens_jy.flatten().tolist())



        fullspecinfo = 1
        if len(msfreq_key) == 1:
                    fullspecinfo = -1
        if fullspecinfo > 0:
            SPWD_INFO = {}
            
            if doprtdatainfo:
                print('\ndetailed frequency information')

            for fky in msfreq_key:
                SPWD_INFO['SPWD_'+fky] = {}
                SPWD_INFO['SPWD_'+fky].update({'ID':fky})
                SPWD_INFO['SPWD_'+fky].update({'SPWD_FREQ_RANGE_hz':[min(msfreq_info[fky]['CHAN_FREQ']),max(msfreq_info[fky]['CHAN_FREQ'])]})
                SPWD_INFO['SPWD_'+fky].update({'SPWD_BANDWIDTH_hz':msfreq_info[fky]['SPW_BW']})
                SPWD_INFO['SPWD_'+fky].update({'SPWD_NUM_CHANNEL':msfreq_info[fky]['SPW_BW']})
                SPWD_INFO['SPWD_'+fky].update({'SPWD_CHANNEL_WIDTH_hz':[min(msfreq_info[fky]['CHANWIDTH']),max(msfreq_info[fky]['CHANWIDTH'])]})


                if doprtdatainfo:
                    print('\t--------------')
                    print('\tSPWD_ID         :',fky)
                    print('\tfrequencies     : %e'%(min(msfreq_info[fky]['CHAN_FREQ'])),' --  %e'%(max(msfreq_info[fky]['CHAN_FREQ'])))
                    print('\tbandwidth       : %e'%(msfreq_info[fky]['SPW_BW']))
                    print('\tchannels        :',msfreq_info[fky]['NUM_CHAN'])
                    print('\tchannel width   : %e'%min(msfreq_info[fky]['CHANWIDTH']),', %e'%max(msfreq_info[fky]['CHANWIDTH']))

            if doprtdatainfo:
                print('\n')

            MS_FULL_INFO.update(SPWD_INFO)


        fullsourceinfo = 1
        if fullsourceinfo > 0:
            SCAN_INFO = {}

            if doprtdatainfo:
                print('\ndetailed source information')
                print('\t--------------')

            for so in sinfo_keys:
                scan_ids   = []
                scan_times = []
                SCAN_INFO[so] = {}
                for sp in range(len(msource_info[so]['SCANTIMES'])):
                    time_low_str = Time(int_time_so[so][sp][0]/(24. * 3600.),scale='utc',format='mjd').iso
                    time_high_str = Time(int_time_so[so][sp][1]/(24. * 3600.),scale='utc',format='mjd').iso
                    if doprtdatainfo:
                        print('\t',so,'| SCAN_ID ',msource_info[so]['SCAN_ID'][sp],' | ',time_low_str,'---',time_high_str)
                    scan_ids.append(int(msource_info[so]['SCAN_ID'][sp]))
                    scan_times.append([time_low_str,time_high_str])
                if doprtdatainfo:
                    print('\t--------------')
                SCAN_INFO[so].update({'SCN_IDS':scan_ids})
                SCAN_INFO[so].update({'SCN_TIMES':scan_times})
                print('\n')
            if doprtdatainfo:
                print('\n')

            MS_FULL_INFO.update(SCAN_INFO)

        if len(dodatainfoutput) > 0:
            with open(cwd + dodatainfoutput, 'w') as fout:
                json_dumps_str = json.dumps(MS_FULL_INFO,indent=4,sort_keys=False,separators=(',', ': '))
                print(json_dumps_str, file=fout)

if __name__ == "__main__":
    main()


