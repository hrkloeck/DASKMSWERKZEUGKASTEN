# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
# this is a first attemp to use dask to extract 
# and handle data from MS files
#
# Hope you enjoy it
# 
# --------------------------------------------------------------------


from daskms import xds_from_ms,xds_from_table
import dask
import dask.array as da

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
import numpy as np
import sys
import json


def ms_data_info(msdata):

    msstab,msstab_kw,msstab_col =  xds_from_table(msdata,table_keywords=True,column_keywords=True)


    ms_tables = []
    for k in msstab_kw:
        if type(msstab_kw[k]) is str:
            if msstab_kw[k].count('Table') > 0:
                ms_tables.append(k)

    ms_data_info = list(msstab_col.keys())

    return ms_tables, ms_data_info


def ms_obs_info(msdata):

    ob_info = xds_from_table(msdata+'::OBSERVATION',group_cols='__row__')

    obs_info = {}
    obs_info['TELESCOPE_NAME'] = list(ob_info[0].TELESCOPE_NAME.values)[0]
    obs_info['PROJECT']        = list(ob_info[0].PROJECT.values)[0]

    ob_infof                   = xds_from_table(msdata+'::FEED')
    obs_info['ANTS']           = list(ms_unique_antenna(msdata))  

    ob_infoa                   = xds_from_table(msdata+'::ANTENNA')
    obs_info['DISH_DIAMETER']  = list(ob_infoa[0].DISH_DIAMETER.values)  

    return obs_info


def convert_jd(time):
    """
    convert the Julian time into UTC
    """
    # convert the Julian time 
    #
    time_utc = Time(time/(24. * 3600.),scale='utc',format='mjd').ymdhms

    return time_utc


def ms_field_info(msdata):
    """
    extract basic source field information
    out of a Measurementset
    """

    # get the relation of the fields and source name
    field_info = xds_from_table(msdata+'::FIELD',group_cols='__row__',columns=["SOURCE_ID","NAME"])

    sour_idx     = {}
    for fi in field_info:
        sour_idx[str(fi.SOURCE_ID.values[0])]  = fi.NAME.values[0]

    return sour_idx


def ms_source_info(msdata):
    """
    extract basic source information
    out of a Measurementset
    """

    # get the relation of the fields and source name
    field_info = xds_from_table(msdata+'::FIELD',group_cols='__row__',columns=["SOURCE_ID","NAME"])

    # get field source index
    sour_idx     = ms_field_info(msdata)

    # get source information
    msinfo = xds_from_table(msdata+'::SOURCE',group_cols='__row__',columns=["TIME", "INTERVAL", "NAME", "CODE","DIRECTION","SOURCE_ID","CALIBRATION_GROUP"])

    sour_info    = {}
    for si in msinfo:
        s_name  = si.NAME.data.compute().flatten()[0]
        ra, dec = si.DIRECTION.data.compute().flatten()
        s_coor  = SkyCoord(ra,dec, frame='icrs', unit='rad')
        #
        sour_info[s_name]= {}
        sour_info[s_name]['SOURCE_ID']    = [si.SOURCE_ID.values[0]]
        sour_info[s_name]['RADEC']        = [s_coor.ra.degree, s_coor.dec.degree]
        sour_info[s_name]['glgb']         = [s_coor.galactic.l.degree, s_coor.galactic.b.degree]
        sour_info[s_name]['HMSDMS']       = [s_coor.to_string('hmsdms')]
        sour_info[s_name]['MIDPOINTTIME'] = [si.TIME.values[0]]
        #
        #print(s_name)
        #print(s_coor.to_string('hmsdms'))
        #print('TIMEVALUES',si.TIME.values)
        #print(si.INTERVAL.values)
        #print(si.CODE.values)
        #print(si.SOURCE_ID.values)
        #print(si.CALIBRATION_GROUP.values)
        

        sour_info[s_name]['SCANTIMES']    = [] 
        sour_info[s_name]['SCANEXPOSURE'] = [] 
        sour_info[s_name]['SCAN_ID']       = [] 


    #no_of_bsls = len(ms_baselines(msdata,tabs='FEED'))

    # Here actually need the time info out of the dataset itself
    #
    #msdata = xds_from_ms(msdata,group_cols=['FIELD_ID'],table_keywords=False,column_keywords=False)
    msdata = xds_from_ms(msdata,group_cols=['SCAN_NUMBER'],table_keywords=False,column_keywords=False)

    # loop over scans
    for scnid in msdata:

        field_obs_trange = scnid.TIME.values            # note the msfid.TIME.values = msfid.TIME_CENTROID.values
        expval_range     = scnid.EXPOSURE.values        # EXPOSURE effective integration time
 
        #print(scnid.ANTENNA1.values)
        #print(scnid.ANTENNA2.values)

        #print(len(scnid.ANTENNA2))

        #print(len(scnid.FIELD_ID.values))
        
        
        #print(scnid.SCAN_NUMBER,scnid.FIELD_ID.values[0],sour_idx[str(scnid.FIELD_ID.values[0])],len(scnid.FIELD_ID.values),len(scnid.TIME.values),np.sum(expval_range)/no_of_bsls,np.mean(expval_range))
        #print('lower t range',Time(((field_obs_trange[0] - (expval_range[0]/2.0))/(24. * 3600.)),scale='utc',format='mjd').iso)
        #print('higher t range',Time(((field_obs_trange[-1] + (expval_range[-1]/2.0))/(24. * 3600.)),scale='utc',format='mjd').iso)

        sour_info[sour_idx[str(scnid.FIELD_ID.values[0])]]['SCANTIMES'].append(scnid.TIME.values)
        sour_info[sour_idx[str(scnid.FIELD_ID.values[0])]]['SCANEXPOSURE'].append(scnid.EXPOSURE.values)
        sour_info[sour_idx[str(scnid.FIELD_ID.values[0])]]['SCAN_ID'].append(scnid.SCAN_NUMBER)

        #sys.exit(-1)

        #for s in sour_info:
            #print(msfid.FIELD_ID,sour_info[s]['SOURCE_ID'],t_low, sour_info[s]['MIDPOINTTIME'],t_up)
        #    if sour_info[s]['MIDPOINTTIME'][0] >= t_low and sour_info[s]['MIDPOINTTIME'][0] <= t_up:
        #        sour_info[s]['FIELD_ID']       = msfid.FIELD_ID
        #        sour_info[s]['TIMERANGEJDSEC'] = [t_low,t_up]
        #        sour_info[s]['TIMERANGEUTC']   = \
        #          [Time(((field_obs_trange[0] - (expval_range[0]/2.0))/(24. * 3600.)),scale='utc',format='mjd').iso,\
        #               Time(((field_obs_trange[-1] + (expval_range[-1]/2.0))/(24. * 3600.)),scale='utc',format='mjd').iso]
        #        print('--------------------------')
        #        print(s)
                #print(msfid.FIELD_ID,sour_info[s]['SOURCE_ID'],t_low, sour_info[s]['MIDPOINTTIME'][0],t_up)
                #print(t_low,t_up)
                # somehow the lower t- range is inconsistemt with CASA listobs (rounding error?)
                #print('lower t range',Time(((field_obs_trange[0] - (expval_range[0]/2.0))/(24. * 3600.)),scale='utc',format='mjd').iso)
                #print('upper t range',Time(((field_obs_trange[-1] + (expval_range[-1]/2.0))/(24. * 3600.)),scale='utc',format='mjd').iso)
        #        print('--------------------------')
        #        print('\n')

    return sour_info


def source_separation(ms_info):
    """
    calculates the source separation (astropy great circle)
    """
    separation = []
    keys       = list(ms_info.keys())
    if len(ms_info.keys()) > 1:
        for so in range(len(keys)):
            for soff in range(so+1,len(keys)):
                so_coord    = SkyCoord(ms_info[keys[so]]['RADEC'][0],ms_info[keys[so]]['RADEC'][1], unit='deg',frame='icrs')
                offse_coord = SkyCoord(ms_info[keys[soff]]['RADEC'][0],ms_info[keys[soff]]['RADEC'][1], unit='deg',frame='icrs')
                separation.append([keys[so],keys[soff],so_coord.separation(offse_coord).deg])
                
    return separation
   


def exposure_per_source(msource_info):
    """
    provides the exosure time per source
    """


    # Determine addition time information
    # copied form the GET_MS_INFO.py
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

    return inttimepersource
 

def ms_freq_info(msdata):
    """
    extract basic frequencyinformation
    out of a Measurementset
    """
    f_info = xds_from_table(msdata+'::SPECTRAL_WINDOW',group_cols='__row__')

    freq_info    = {}
    for fr in f_info:
        freq_info[str(fr.ROWID.values[0])] = {}    
        freq_info[str(fr.ROWID.values[0])]['NUM_CHAN']        = fr.NUM_CHAN.values[0]
        freq_info[str(fr.ROWID.values[0])]['CHAN_FREQ']       = fr.CHAN_FREQ.values[0] 
        freq_info[str(fr.ROWID.values[0])]['CHANWIDTH']       = fr.CHAN_WIDTH.values[0] 
        freq_info[str(fr.ROWID.values[0])]['SPW_BW']          = fr.CHAN_FREQ.values[0].max() - fr.CHAN_FREQ.values[0].min()

        #freq_info[str(fr.ROWID.values[0])]['CHANWIDTH']       = {'MIN':fr.CHAN_WIDTH.min().data.compute(),'MAX':fr.CHAN_WIDTH.max().data.compute()}
        #freq_info[str(fr.ROWID.values[0])]['TOTAL_BANDWIDTH'] = list(fr.TOTAL_BANDWIDTH.values)    
        #freq_info[str(fr.ROWID.values[0])]['EFFECTIVE_BW']    = list(fr.EFFECTIVE_BW.values)
        #freq_info[str(fr.ROWID.values[0])]['NET_SIDEBAND']    = list(fr.NET_SIDEBAND.values)


    return freq_info

def ms_pol_info(msdata):
    """
    extract basic polarisation information
    out of a Measurementset
    """
    from STOKES_ID import STOKES_TYPES

    p_info = xds_from_table(msdata+'::POLARIZATION',group_cols='__row__')

    pol_info = {}    
    pol_info['STOKES']       = [STOKES_TYPES[icorr] for icorr in p_info[0].CORR_TYPE.values[0]]
    pol_info['NUMBOFSTOKES'] =  p_info[0].NUM_CORR.values
    pol_info['CORR_PRODUCT'] =  p_info[0].CORR_PRODUCT.values
    pol_info['CORR_TYPE']    =  p_info[0].CORR_TYPE.values
    
    return pol_info

def get_stokes_para(stokesp):
    """
    input: integer or string
    return the Stokes Parameter or the Stokes ID
    """
    from STOKES_ID import STOKES_TYPES
    
    if str(stokesp).isdigit():
        stokes = STOKES_TYPES[stokesp]
        return stokes
    else:
        stkey = STOKES_TYPES.keys()
        st_id = []
        st_p  = []
        for k in stkey: 
            st_id.append(k)
            st_p.append(STOKES_TYPES[k])

        if st_p.count(stokesp) == 0:
            return st_id[st_p.index('Undefined')]
        else:
            return st_id[st_p.index(stokesp)]


def get_array_center(msdata):
    """
    determine array center as 3-dim point
    """

    import numpy as np


    ant_pos           = list(xds_from_table(msdata+'::'+'ANTENNA')[0].POSITION.values)
    ant_pos_transpose = np.transpose(ant_pos)
    array_centre      = [np.mean(ant_pos_transpose[0]),np.mean(ant_pos_transpose[1]),np.mean(ant_pos_transpose[2])]

    return array_centre

def order_antenna_wrst_A_center(msdata):
    """
    provide antenna name and idex sorted by distance
    to the array center
    """

    ant_idx       = ms_unique_antenna(msdata,tabs='FEED')
    ant_name      = ms_unique_antenna(msdata,tabs='ANTENNA')
    ant_pos       = list(xds_from_table(msdata+'::'+'ANTENNA')[0].POSITION.values)

    array_centre  = get_array_center(msdata)

    dist       = []
    dist_a_idx = []
    for i in range(len(ant_pos)):

        dist.append(np.linalg.norm(ant_pos[i]-array_centre))
        dist_a_idx.append(ant_idx[i])

    sort_idx = np.argsort(dist)

    return np.array(dist)[sort_idx], np.array(dist_a_idx)[sort_idx],np.array(ant_name)[sort_idx]


def ms_unique_antenna(msdata,tabs='FEED'):
    """
    """
    ob_info = xds_from_table(msdata+'::'+tabs)

    if tabs == 'FEED':
        ants   = list(np.unique(ob_info[0].ANTENNA_ID.values)) 
    elif tabs == 'ANTENNA':
        ants   = list(np.unique(ob_info[0].NAME.values))
    return(ants)


def ms_baselines(msdata,tabs='FEED'):
    """
    return the baseline pairs
    if typ is FEED than antenna ID are provided
    if typ is ANTENNA than antenna NAME are provided

    """
    ob_info  = xds_from_table(msdata+'::'+tabs)
    if tabs == 'FEED':
        ants   = list(np.unique(ob_info[0].ANTENNA_ID.values)) 
    elif tabs == 'ANTENNA':
        ants   = list(np.unique(ob_info[0].NAME.values))
    else:
        ants = []

    baselines      = [ [i,j] for i in ants for j in ants if i<j ]

    return baselines


def ms_baselines_length(msdata,bsls=[[]]):
    """
    return the baseline pairs
    if typ is FEED than antenna ID are provided
    if typ is ANTENNA than antenna NAME are provided
    unit is meter [m]
    """

    if np.cumprod(np.array(bsls).shape)[-1] == 0:
        bsls     = ms_baselines(msdata,tabs='FEED')

    ob_info  = xds_from_table(msdata+'::ANTENNA')
    pos      = list(ob_info[0].POSITION.data.compute())  
    
    bsl_norm = []
    for bl in bsls:
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        bsl_norm.append(np.linalg.norm(pos[bl[0]]-pos[bl[1]]))

    return bsl_norm


def get_nearest_bin_value(data,value,type=1):
    """
    provide index of the nearest value
    this is to ensure to get the correct bin in freq or time
    type = 1 : return the value of the data
    type != 1: return the index of the data 
    """
    idx  = np.abs(np.array(data) - value).argmin()

    if type == 1:
        return data[idx]
    else:
        return idx

def ms_tab(msdata,tabs='SOURCE',col=''):
    
    if tabs != 'SOURCE':
        msstab,msstab_kw,msstab_col =  xds_from_table(msdata+'::'+tabs,table_keywords=True,column_keywords=True)
    else:
        msstab,msstab_kw,msstab_col = xds_from_table(msdata+'::SOURCE',group_cols='__row__',columns=['DIRECTION', 'PROPER_MOTION', 'CALIBRATION_GROUP', 'CODE', 'INTERVAL', 'NAME', 'NUM_LINES', 'SOURCE_ID', 'SPECTRAL_WINDOW_ID', 'TIME'],table_keywords=True,column_keywords=True)

    return list(msstab[0][col].values)


def ms_tables(msdata,doinfo=''):
    """
    Dertermnine the availible tables of a MS set and
    provides information table is full
    """
    
    # Determine the attached tables to the MS file
    ms,mstab  = xds_from_ms(msdata,table_keywords=True,column_keywords=False)
    mstab_kys = list(mstab.keys())


    ms_tables = []
    for i in range(len(mstab_kys)):
        if type(mstab[mstab_kys[i]]) is str:
            ms_tables.append(mstab[mstab_kys[i]].replace('Table: '+msdata+'/',''))

    tab_info = {}
    for tabs in ms_tables:
        if tabs != 'SOURCE':
            msstab,msstab_kw,msstab_col =  xds_from_table(msdata+'::'+tabs,table_keywords=True,column_keywords=True)
        else:
            msstab,msstab_kw,msstab_col = xds_from_table(msdata+'::SOURCE',group_cols='__row__',columns=['DIRECTION', 'PROPER_MOTION', 'CALIBRATION_GROUP', 'CODE', 'INTERVAL', 'NAME', 'NUM_LINES', 'SOURCE_ID', 'SPECTRAL_WINDOW_ID', 'TIME'],table_keywords=True,column_keywords=True)
            if doinfo == tabs:                
                print('\n CAUTION TABLE SOURCE CRASH DUE TO SOURCE_MODEL COLUMN -- IS EXCULDED')
                print(list(msstab_col.keys()),'\n')


        # This entire thing is not super clean yet, because I'm not checking each row 
        tab_info[tabs] = len(msstab[0].row.values)

        if doinfo == tabs:
            print('\n\nGet Info from table: ',tabs)
            for row in msstab:
                kywds = list(row.keys())
                for co in kywds:
                    print('\tcolumn : ',co)

                    #   ---- ERROR TESING ----- 
                    if co == 'APP_PARAMS99':
                        #print(msstab_col[co])

                        #print(len(msstab_col[co]))
                        
                        for un in msstab_col[co].keys():
                            print('\tunit   : ',msstab_col[co][un])

                        #print(row[co].dims)
                        #print(row[co].attrs)
                        #print(row[co]['APP_PARAMS-1'])
                        #print(row[co]['row'])

                        #if len(row[co].attrs)
                        print('\tdata   : ',row[co])
                        #print('data   : ',row[co].values)
                        print('\n')
                        #
                        sys.exit(-1)
                       

                    for un in msstab_col[co].keys():
                        print('\tunit   : ',msstab_col[co][un])

                    if co == 'APP_PARAMS' or co == 'CLI_COMMAND':
                        print('\tdata   : ',row[co])
                        #if co == 'CLI_COMMAND':
                        #    print('\tlllllllldata   : ',row[co].data.compute)
                        #    sys.exit(-1)
                    else:
                        print('\tdata   : ',row[co].values)

                    print('\n')


    return tab_info

def ms_check_col(msdata,col_name):
    """
    checks if MS set has e.g. MODEL_DATA
    """

    ms,tab,col = xds_from_ms(msdata,chunks={'row':1000},group_cols=['FIELD_ID'],table_keywords=True,column_keywords=True)
    col_kys    = list(col.keys())

    table_isthere = -1
    for ky in col_kys:
        if col_name == ky:
            table_isthere = 1

    return table_isthere




def ms_average_data(MSFN,input_field_id,input_scan_no,bsl,data_type,showparameter,print_info):
    """
    computes an average spectrum
    
    this function extracts the data from the original structure of
    visibilities (per time correlation products of all baselines)
    here we select on baselines and average the measurements over 
    all baselines
    """

    if print_info:
        # get some info to 
        msource_info     = ms_source_info(MSFN)
        fie_sour         = ms_field_info(MSFN)
        inttimepersource = exposure_per_source(msource_info)


    chunksize = 10000

    # get the measurmment set 
    # 
    # define the data sorting IMPORTANT
    group_cols = ['FIELD_ID', 'SCAN_NUMBER','TIME','DATA_DESC_ID']
    index_cols = []
    #
    ms         = xds_from_ms(MSFN,chunks={'row':chunksize}, group_cols=group_cols,index_cols=index_cols)

    # get the overall information
    #
    msstab,msstab_kw,msstab_col =  xds_from_table(MSFN+'::DATA_DESCRIPTION',table_keywords=True,column_keywords=True)

    # get the frequency info
    #
    spwd_info   = xds_from_table(MSFN+'::SPECTRAL_WINDOW')
    spwds       = msstab[0].SPECTRAL_WINDOW_ID.data.compute()

    # get the polarisation info
    #
    #pol_info       = xds_from_table(MSFN+'::POLARIZATION')
    dapolinfo   = ms_pol_info(MSFN)
    corr_type   = dapolinfo['STOKES']


    # here the data averaging is happening
    #
    dyn_specs    = {}
    #
    previous_scan_no  = 0
    tcounter          = 0
    previous_ddid     = 0
    #
    for msds in ms:

        # optain sorting info
        #
        fie_id     = msds.attrs['FIELD_ID']
        scan_no    = msds.attrs['SCAN_NUMBER']
        ddid       = msds.attrs['DATA_DESC_ID']                


        #
        if input_field_id == -1:
                select_field_id = [fie_id]
        else:
                select_field_id = [input_field_id]

        if input_scan_no == -1:
                select_scan_no = [scan_no]
        else:
                select_scan_no = [input_scan_no]


        #
        if (fie_id in select_field_id) and (scan_no in select_scan_no):


                # get some progress info to see if its still working
                #
                if print_info:
                    
                    if ddid - previous_ddid != 0:
                        tcounter = 0
                        print('data descent id', ddid)

                    if scan_no - previous_scan_no == 0:
                        tcounter += (msds.attrs['TIME'] - previous_time)

                    previous_scan_no = scan_no
                    previous_time    = msds.attrs['TIME']
                    previous_ddid    = ddid

                    # show progress bar
                    #
                    progressBar(int(100*tcounter/inttimepersource[fie_sour[str(fie_id)]]), 100, bar_length=20)


                # define baseline selection
                #
                if len(bsl) == 0:
                        sel_bsls  = np.ones(msds.ANTENNA1.shape,dtype=bool)
                else:
                        sel_bsls  = np.zeros(msds.ANTENNA1.shape,dtype=bool)                        
                        for bl in bsl:

                                sel_ant1 = msds.ANTENNA1.data == bl[0]

                                if type(bl[1]) is str and bl[1] == '*':
                                    sel_ant2 = np.ones(msds.ANTENNA1.shape,dtype=bool)
                                else:
                                   sel_ant2 = msds.ANTENNA2.data == bl[1]

                                sel_bsl_single = da.logical_and(sel_ant1,sel_ant2).compute()                    
                                sel_bsls       = da.logical_or(sel_bsl_single,sel_bsls)


                # determine number of baseline
                # that are selected
                #
                sel_bsl_ant1 = msds.ANTENNA1.data.compute()[sel_bsls]
                sel_bsl_ant2 = msds.ANTENNA2.data.compute()[sel_bsls]
                #
                sel_bsls_ant = []
                for aa in range(len(sel_bsl_ant1)):
                        sel_bsls_ant.append([sel_bsl_ant1[aa],sel_bsl_ant2[aa]])


                # optain info
                #
                spwd_id    = msstab[0].SPECTRAL_WINDOW_ID.data[ddid].compute()
                #pol_id     = msstab[0].POLARIZATION_ID.data[ddid].compute()        
                #
                chan_freq  = spwd_info[0].CHAN_FREQ.data[spwd_id].compute()
                #corr_type  = pol_info[0].CORR_TYPE.data[pol_id].compute()
                #
                if ddid != spwd_id:
                        print('The dask file structuring changed, stopped here')
                        print('Check ', didd,' and ',spwd_id)
                        sys.exit(-1)

                # init data selection 
                #
                if str(scan_no) not in dyn_specs:
                        dyn_specs[str(scan_no)]                   = {}
                        dyn_specs[str(scan_no)]['INFO_SPWD']      = spwds
                        dyn_specs[str(scan_no)]['INFO_CORR']      = corr_type
                        dyn_specs[str(scan_no)]['INFO_BSLS']      = sel_bsls_ant
                        dyn_specs[str(scan_no)]['INFO_DATATYPE']  = data_type

                if str(ddid) not in dyn_specs[str(scan_no)]:
                        dyn_specs[str(scan_no)][str(spwd_id)] = {}
                        dyn_specs[str(scan_no)][str(spwd_id)]['chan_freq']        = chan_freq
                        dyn_specs[str(scan_no)][str(spwd_id)]['bsl_sel']          = bsl
                        #
                        dyn_specs[str(scan_no)][str(spwd_id)]['time_range']       = []
                        dyn_specs[str(scan_no)][str(spwd_id)][data_type]          = []
                        dyn_specs[str(scan_no)][str(spwd_id)][data_type+'STD']    = []
                        dyn_specs[str(scan_no)][str(spwd_id)]['flag']             = []   # CAUTION NEEDS TO BE lower case characters



                if len(sel_bsls_ant) >= 1:

                    # get data
                    #
                    time       = msds.attrs['TIME']

                    # include time  
                    #
                    dyn_specs[str(scan_no)][str(spwd_id)]['time_range'].append(time)

                    # do the data averaging
                    #
                    if data_type == 'DATA' or data_type == 'CORRECTED_DATA' or data_type == 'MODEL':
                            if showparameter == 'AMP':
                                    data_dt   = np.abs(msds[data_type].data[sel_bsls].compute())
                            else:
                                    data_dt   = np.phase(msds[data_type].data[sel_bsls].compute(),deg=True)

                    else:
                            data_dt        = msds[data_type].data[sel_bsls].compute()

                    # Generates a dynamic spectrum
                    #
                    # Note CASA flag data, boolean value is True
                    flag_dt        = msds.FLAG.data[sel_bsls].compute()

                    #
                    #
                    if len(sel_bsls_ant) > 1:
                            dyn_specs[str(scan_no)][str(spwd_id)]['flag'].append(flag_dt.astype(dtype=int).mean(axis=0))                        
                            #
                            # Note in NUMPY MASKED arrays a bolean value of True is considered invalid  
                            data_dt_masked = np.ma.masked_array(data_dt,mask=flag_dt)
                            #
                            dyn_specs[str(scan_no)][str(spwd_id)][data_type].append(data_dt_masked.mean(axis=0))
                            dyn_specs[str(scan_no)][str(spwd_id)][data_type+'STD'].append(data_dt_masked.std(axis=0))

                    else:
                            dyn_specs[str(scan_no)][str(spwd_id)]['flag'].append(flag_dt.astype(dtype=int).mean(axis=0))                        
                            #
                            # Note in NUMPY MASKED arrays a bolean value of True is considered invalid  
                            data_dt_masked = np.ma.masked_array(data_dt,mask=flag_dt)
                            #
                            dyn_specs[str(scan_no)][str(spwd_id)][data_type].append(data_dt_masked[0,:,:])
                            dyn_specs[str(scan_no)][str(spwd_id)][data_type+'STD'].append(data_dt_masked[0,:,:])


    return dyn_specs



def combine_averaged_data(dyn_specs,select_freq,select_time,select_spwd,print_info=True):
    """
    will combine the averaged data
    """

    # get the overall accessing
    #
    scans = dyn_specs.keys()
    scan_key_info = []
    for sc in scans:
        sscan_keys = dyn_specs[sc].keys()
        info_keys = []
        spwd_id   = []
        for ssc in sscan_keys:
        
            if ssc.count('INFO') > 0:
                info_keys.append(ssc)
            else:
                spwd_id.append(ssc)
        scan_key_info.append([sc,info_keys,spwd_id])


    # build a merged waterfall spectrum
    #
    concat_freq       = []
    concat_time       = []
    concat_data       = []
    concat_datastd    = []
    concat_data_flag  = []
    #
    
    for i, sc in enumerate(scan_key_info):
        info_corr    = dyn_specs[sc[0]][sc[1][1]]
        info_bsls    = dyn_specs[sc[0]][sc[1][2]]
        info_bsls    = dyn_specs[sc[0]][sc[1][2]]
        data_type    = dyn_specs[sc[0]][sc[1][3]]
        spwd         = dyn_specs[sc[0]][sc[1][0]]


        if select_spwd == -1:
            for s,sw in enumerate(spwd):

                if print_info:
                    print('sc',sc)
                    print('swpd',sw)
                    print('datatype',data_type)
                    print('info_corr',info_corr)
                    print('time',np.array(dyn_specs[sc[0]][str(sw)]['time_range']).shape)
                    print('freq',np.array(dyn_specs[sc[0]][str(sw)]['chan_freq']).shape)
                    print('flag',np.array(dyn_specs[sc[0]][str(sw)]['FLAG']).shape)
                    print('data',np.array(dyn_specs[sc[0]][str(sw)][data_type]).shape)


                if s == 0:
                    concat_sw_data      = dyn_specs[sc[0]][str(sw)][data_type]
                    concat_sw_datastd   = dyn_specs[sc[0]][str(sw)][data_type+'STD']
                    concat_sw_flag      = dyn_specs[sc[0]][str(sw)]['FLAG']
                    concat_freq         = dyn_specs[sc[0]][str(sw)]['chan_freq']
                else:
                    concat_sw_data      = np.concatenate((concat_sw_data,dyn_specs[sc[0]][str(sw)][data_type]),axis=1)
                    concat_sw_datastd   = np.concatenate((concat_sw_datastd,dyn_specs[sc[0]][str(sw)][data_type+'STD']),axis=1)
                    concat_sw_flag      = np.concatenate((concat_sw_flag,dyn_specs[sc[0]][str(sw)]['FLAG']),axis=1)
                    concat_freq         = np.concatenate((concat_freq,dyn_specs[sc[0]][str(sw)]['chan_freq']))

                time_sc_sw = dyn_specs[sc[0]][str(sw)]['time_range']
 

        else:
            concat_sw_data      = dyn_specs[sc[0]][str(select_spwd)][data_type]
            concat_sw_datastd   = dyn_specs[sc[0]][str(select_spwd)][data_type+'STD']
            concat_sw_flag      = dyn_specs[sc[0]][str(select_spwd)]['FLAG']
            concat_freq         = dyn_specs[sc[0]][str(select_spwd)]['chan_freq']

            time_sc_sw          = dyn_specs[sc[0]][str(select_spwd)]['time_range']


        if i == 0:
            concat_data         = concat_sw_data
            concat_datastd      = concat_sw_datastd
            concat_data_flag    = concat_sw_flag
            concat_time         = time_sc_sw
        else:
            concat_data         = np.concatenate((concat_data,concat_sw_data),axis=0)
            concat_datastd      = np.concatenate((concat_datastd,concat_sw_datastd),axis=0)
            concat_data_flag    = np.concatenate((concat_data_flag,concat_sw_flag),axis=0)
            concat_time         = np.concatenate((concat_time,time_sc_sw))

    return concat_data,concat_datastd,concat_data_flag,concat_freq,concat_time





def ms_get_bsl_data_old_backup(msdata,field_idx=0,setspwd=0,bsls=[[0,1],[1,2]],bsl_idx=[0,2]):
    """
    return bsl selected dataset for a single field/source
    in a nested dictionary

    Dictionary keywords are: 
    [baseline index][DATA]
    [baseline index][FLAG]
    [baseline index][MODEL_DATA]    (only if data is present)
    [baseline index][CORRECTED_DATA] (only if data is present)
    [baseline index][TIME_CENTROID]
    [baseline index][CHAN_FREQ]
    """
    
    # Load the real data
    ms = xds_from_ms(msdata, group_cols=['DATA_DESC_ID','FIELD_ID'])

    # Get data description (e.g. spectral windows)
    #
    dades          = xds_from_table(msdata+'::DATA_DESCRIPTION')
    didesinfo      = dades[0].compute()
    spwd_idx       = didesinfo.SPECTRAL_WINDOW_ID.data

    # Check if data has MODEL data 
    get_model_data = ms_check_col(msdata,'MODEL_DATA')


    # Check if data has CORRECTED_DATA data 
    get_corrected_data = ms_check_col(msdata,'CORRECTED_DATA')


    # get the frequency info
    daspc     = xds_from_table(msdata+'::SPECTRAL_WINDOW')
    daspcinfo = daspc[0].compute()
    spwd_freq = daspcinfo.CHAN_FREQ.data


    sub_data_bsl ={}
    for msds in ms:
        # seting to allow to fish a spectral window out
        # otherwise will concatenate all spectral windows 
        #
        if setspwd == -1:
            spwd  = msds.attrs['DATA_DESC_ID']
        else:
            spwd = setspwd


        # Selects only Data from selected field
        #
        if msds.attrs['FIELD_ID'] == field_idx and msds.attrs['DATA_DESC_ID'] == spwd:

            #for bl,blidx in zip(bsls,bsl_idx):
            for blidx in bsl_idx:
                if setspwd == -1:
                    if msds.attrs['DATA_DESC_ID'] == 0:
                        sub_data_bsl[blidx] = {}
                        sub_data_bsl[blidx]['DATA']      = []
                        sub_data_bsl[blidx]['FLAG']      = []

                        if get_corrected_data != -1:
                            sub_data_bsl[blidx]['CORRECTED_DATA'] = []

                        if get_model_data != -1:
                            sub_data_bsl[blidx]['MODEL'] = []

                else:

                    if msds.attrs['DATA_DESC_ID'] == spwd:
                        sub_data_bsl[blidx] = {}
                        sub_data_bsl[blidx]['DATA']      = []
                        sub_data_bsl[blidx]['FLAG']      = []

                        if get_corrected_data != -1:
                            sub_data_bsl[blidx]['CORRECTED_DATA'] = []

                        if get_model_data != -1:
                            sub_data_bsl[blidx]['MODEL'] = []


                # Selecting baseline data
                #
                sel_bsl    = da.logical_and(msds.ANTENNA1.data == bsls[blidx][0],msds.ANTENNA2.data == bsls[blidx][1])    

                #bsl_data   = dask.compute(msds.DATA.data[sel_bsl])

                # store DATA and FLAG's 
                sub_data_bsl[blidx]['DATA'].append(msds.DATA.data[sel_bsl])
                sub_data_bsl[blidx]['FLAG'].append(msds.FLAG.data[sel_bsl])


                # if CORRECTED data is present inculde 
                if get_corrected_data != -1:
                    sub_data_bsl[blidx]['CORRECTED_DATA'].append(msds.CORRECTED_DATA.data[sel_bsl])


                # if MODEL data is present inculde
                if get_model_data != -1:
                    sub_data_bsl[blidx]['MODEL'].append(msds.MODEL_DATA.data[sel_bsl])

                
                # get the time
                # CAUTION THIS COULD BE DANGEROUS IF SPWD HAVE DIFFERENT TIME-RANGES
                sub_data_bsl[blidx]['TIME_CENTROID'] = msds.TIME_CENTROID.data[sel_bsl]

                # get the frequency
                sub_data_bsl[blidx]['CHAN_FREQ'] = spwd_freq[spwd]


    return sub_data_bsl


def ms_get_bsl_data_scan(msdata,field_idx=-1,scan_num=-1,spwd=-1,bsls=[[0,1],[1,2]],bsl_idx=[0,2]):
    """
    return bsl selected dataset for a single field/source
    in a nested dictionary

    Dictionary keywords are: 
    [baseline index][DATA]
    [baseline index][FLAG]
    [baseline index][MODEL_DATA]    (only if data is present)
    [baseline index][CORRECTED_DATA] (only if data is present)
    [baseline index][TIME_CENTROID]
    [baseline index][CHAN_FREQ]
    """
    
    ms = xds_from_ms(msdata, group_cols=['DATA_DESC_ID','FIELD_ID','SCAN_NUMBER'])

    # Get data description (e.g. spectral windows)
    #
    dades          = xds_from_table(msdata+'::DATA_DESCRIPTION')
    didesinfo      = dades[0].compute()
    spwd_idx       = didesinfo.SPECTRAL_WINDOW_ID.data

    # Check if data has MODEL data 
    get_model_data = ms_check_col(msdata,'MODEL_DATA')


    # Check if data has CORRECTED_DATA data 
    get_corrected_data = ms_check_col(msdata,'CORRECTED_DATA')


    # get the frequency info
    daspc     = xds_from_table(msdata+'::SPECTRAL_WINDOW')
    daspcinfo = daspc[0].compute()
    spwd_freq = daspcinfo.CHAN_FREQ.data

    # make a copy of the input
    set_spwd      = spwd
    set_scan_num  = scan_num
    set_field_idx = field_idx   

    sub_data_bsl ={}
    for msds in ms:
        #
        # seting to allow to fish a spectral window out
        # otherwise will concatenate all spectral windows 
        #
        if set_spwd == -1:
            spwd  = msds.attrs['DATA_DESC_ID']
        else:
            spwd  = set_spwd

        if set_scan_num == -1:
            scan_num  = msds.attrs['SCAN_NUMBER']
        else:
            scan_num  = set_scan_num

        if set_field_idx == -1:
            field_idx  = msds.attrs['FIELD_ID']
        else:
            field_idx = set_field_idx


        # Selects only Data from selected field
        #
        if msds.attrs['FIELD_ID'] == field_idx and msds.attrs['DATA_DESC_ID'] == spwd and msds.attrs['SCAN_NUMBER'] == scan_num:

            #for bl,blidx in zip(bsls,bsl_idx):
            for blidx in bsl_idx:

                if (blidx in sub_data_bsl) == False:
                        sub_data_bsl[blidx] = {}
                        sub_data_bsl[blidx]['CHAN_FREQ'] = []
                        sub_data_bsl[blidx]['DATA']      = []
                        sub_data_bsl[blidx]['FLAG']      = []

                        if get_corrected_data != -1:
                            sub_data_bsl[blidx]['CORRECTED_DATA'] = []

                        if get_model_data != -1:
                            sub_data_bsl[blidx]['MODEL_DATA'] = []

                #print('SPWD',spwd)

                # Selecting baseline data
                #
                sel_bsl    = da.logical_and(msds.ANTENNA1.data == bsls[blidx][0],msds.ANTENNA2.data == bsls[blidx][1])    

                #bsl_data   = dask.compute(msds.DATA.data[sel_bsl])

                # store DATA and FLAG's 
                sub_data_bsl[blidx]['DATA'].append(msds.DATA.data[sel_bsl])
                sub_data_bsl[blidx]['FLAG'].append(msds.FLAG.data[sel_bsl])

                #print(sub_data_bsl[blidx]['DATA'].compute().shape)
                #print(msds.TIME_CENTROID.data[sel_bsl].compute().shape,spwd_freq[spwd].shape)

                #print(msds.TIME_CENTROID.data[sel_bsl].compute())

                # if CORRECTED data is present inculde 
                if get_corrected_data != -1:
                    sub_data_bsl[blidx]['CORRECTED_DATA'].append(msds.CORRECTED_DATA.data[sel_bsl])


                # if MODEL data is present inculde
                if get_model_data != -1:
                    sub_data_bsl[blidx]['MODEL_DATA'].append(msds.MODEL_DATA.data[sel_bsl])
                

                # get the frequency
                #
                sub_data_bsl[blidx]['CHAN_FREQ'].append(spwd_freq[spwd])

                # get the time
                #
                sub_data_bsl[blidx]['TIME_CENTROID'] = msds.TIME_CENTROID.data[sel_bsl]


    
    return sub_data_bsl


def ms_get_bsl_data(msdata,field_idx=-1,spwd=-1,bsls=[[0,1],[1,2]],bsl_idx=[0,2]):
    """
    return bsl selected dataset for a single field/source
    in a nested dictionary

    Dictionary keywords are: 
    [baseline index][DATA]
    [baseline index][FLAG]
    [baseline index][MODEL_DATA]    (only if data is present)
    [baseline index][CORRECTED_DATA] (only if data is present)
    [baseline index][TIME_CENTROID]
    [baseline index][CHAN_FREQ]
    """

    
    # Load the data
    ms = xds_from_ms(msdata, group_cols=['DATA_DESC_ID','FIELD_ID'])

    # Get data description (e.g. spectral windows)
    #
    dades          = xds_from_table(msdata+'::DATA_DESCRIPTION')
    didesinfo      = dades[0].compute()
    spwd_idx       = didesinfo.SPECTRAL_WINDOW_ID.data

    # Check if data has MODEL data 
    get_model_data = ms_check_col(msdata,'MODEL_DATA')


    # Check if data has CORRECTED_DATA data 
    get_corrected_data = ms_check_col(msdata,'CORRECTED_DATA')


    # get the frequency info
    daspc     = xds_from_table(msdata+'::SPECTRAL_WINDOW')
    daspcinfo = daspc[0].compute()
    spwd_freq = daspcinfo.CHAN_FREQ.data

    # make a copy of the input
    set_spwd      = spwd
    set_field_idx = field_idx   

    sub_data_bsl ={}
    for msds in ms:
        #
        # seting to allow to fish a spectral window out
        # otherwise will concatenate all spectral windows 
        #
        if set_spwd == -1:
            spwd  = msds.attrs['DATA_DESC_ID']
        else:
            spwd  = set_spwd

        if set_field_idx == -1:
            field_idx  = msds.attrs['FIELD_ID']
        else:
            field_idx = set_field_idx


        # Selects only Data from selected field
        #
        if msds.attrs['FIELD_ID'] == field_idx and msds.attrs['DATA_DESC_ID'] == spwd:

            #for bl,blidx in zip(bsls,bsl_idx):
            for blidx in bsl_idx:

                if (blidx in sub_data_bsl) == False:
                        sub_data_bsl[blidx] = {}
                        sub_data_bsl[blidx]['CHAN_FREQ'] = []
                        sub_data_bsl[blidx]['DATA']      = []
                        sub_data_bsl[blidx]['FLAG']      = []

                        if get_corrected_data != -1:
                            sub_data_bsl[blidx]['CORRECTED_DATA'] = []

                        if get_model_data != -1:
                            sub_data_bsl[blidx]['MODEL_DATA'] = []

                #print('SPWD',spwd)

                # Selecting baseline data
                #
                sel_bsl    = da.logical_and(msds.ANTENNA1.data == bsls[blidx][0],msds.ANTENNA2.data == bsls[blidx][1])    


                #bsl_data   = dask.compute(msds.DATA.data[sel_bsl])

                # store DATA and FLAG's 
                sub_data_bsl[blidx]['DATA'].append(msds.DATA.data[sel_bsl])
                sub_data_bsl[blidx]['FLAG'].append(msds.FLAG.data[sel_bsl])

                #print(sub_data_bsl[blidx]['DATA'].compute().shape)
                #print(msds.TIME_CENTROID.data[sel_bsl].compute().shape,spwd_freq[spwd].shape)

                #print(msds.TIME_CENTROID.data[sel_bsl].compute())

                # if CORRECTED data is present inculde 
                if get_corrected_data != -1:
                    sub_data_bsl[blidx]['CORRECTED_DATA'].append(msds.CORRECTED_DATA.data[sel_bsl])


                # if MODEL data is present inculde
                if get_model_data != -1:
                    sub_data_bsl[blidx]['MODEL_DATA'].append(msds.MODEL_DATA.data[sel_bsl])

                # get the frequency
                #
                sub_data_bsl[blidx]['CHAN_FREQ'].append(spwd_freq[spwd])
                
                # get the time
                #
                sub_data_bsl[blidx]['TIME_CENTROID'] = msds.TIME_CENTROID.data[sel_bsl]



    
    return sub_data_bsl



def nested_dict(existing=None, **kwargs):
    """
    useful thing to create nested dictionaries
    https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict
    """
    from collections import defaultdict

    if existing is None:
        existing = defaultdict()
    if isinstance(existing, list):
        existing = [nested_dict(val) for val in existing]
    if not isinstance(existing, dict):
        return existing
    existing = {key: nested_dict(val) for key, val in existing.items()}

    return defaultdict(nested_dict, existing, **kwargs)



def getitem_recursive(d, key):
    """
    # https://stackoverflow.com/questions/71460721/best-way-to-get-nested-dictionary-items
    """
    if len(key) !=  1:
        return getitem_recursive(d[key[0]], key[1:])
    else:
        return d[key[0]]


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


def merge_spwds(bsl_data):
    """
    Reshape the bsl data from 
    [spwd, time, channels,polarisation]
    [time, spwd * channels,polarisation]

    """

    org_shape = bsl_data.shape

    # assume 0 axis is spwds and axis 2 is channels
    swap_bsldata    = np.swapaxes(bsl_data,0,1)
    dataswapshape   = swap_bsldata.shape
    bsldata_t_c_p   = swap_bsldata.reshape(dataswapshape[0],dataswapshape[1]*dataswapshape[2],dataswapshape[3])

    return(bsldata_t_c_p)

def merge_spwds_freqs(freqs):
    """
    Reshape the frequency info 
    [spwd, channels]
    [spwd * channels]
    """

    org_shape  = freqs.shape
    freqs_full = freqs.reshape(org_shape[0]*org_shape[1])

    return(freqs_full)




def average_cdata(cdata,axis=1):
    """
    This averages the complex data by averaging
    the real and imag parts separately  and 
    combines them back into a complex array
    
    cdata structure: axis = 0 is time; axis = 1 frequency
    
    """

    rdata           = np.nanmean(cdata.real,axis=axis)
    data_time_shape = rdata.shape
    #
    ccdata = np.empty(data_time_shape,dtype=np.complex128)
    ccdata.real = rdata
    ccdata.imag = np.nanmean(cdata.imag,axis=axis)
    
    return(ccdata)



def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()



def beams(ant_dia,obs_freq,type='FoV'):
    """
    input diameter in [m] and frequency in [Hz]
    provide the primary beam or synthesised beam 
    sizde in [deg]
    """
    from astropy import constants as const
    from math import pi

    if type == 'FoV':
        # position of first null is 1.22 
        # the fov of the primary bean is twice this value
        return((2 * 1.22 * (const.c/obs_freq)/np.array(ant_dia)  * 180./pi).value)
    else:
        # half power beam width = 1.02
        return((1.02 *  (const.c/obs_freq)/np.array(ant_dia) * 180./pi).value)


def SEFD_theo(diameter,T_sys,eta_a):
    """
    SEFD  stands for system equivalent flux density

    Based on Wrobel & Walker 1999 Synthesis Imaging -- page 172

    equation: 9-5

    diameter   = antenna diameter     [m]
    eta_a      = antenna efficiency
    T_sys      = system temperature   [K]
    """
    import numpy as np
    from astropy import constants as const
    from math import pi

    K_B   = const.k_B.value          # J/(K)

    A     =  pi * (np.array(diameter)/2.)**2   # antenna area         [m^2]

    return(T_sys / ((eta_a * A)/(2*K_B)))


def SEFD_MK_SYSTEM(obsband):
    """
    https://www.meerkatplus.tel/mk-technical-details/

    The SEFD's are based on the MK+ page
    """

    SEFD = 0

    if obsband == 'SBAND':
        SEFD = 495   # Jy
    if obsband =='LBAND':
        SEFD = 426   # Jy

    return SEFD

def SEFD_SKAMID_SYSTEM(obsband):
    """
    https://www.meerkatplus.tel/mk-technical-details/

    The SEFD's are based on the MK+ page
    """

    SEFD = 0

    if obsband == 'SBAND':
        SEFD = 0.7332 * 495   # Jy
    if obsband =='LBAND':
        SEFD = 0.7332 * 426   # Jy

    return SEFD

def image_sensitivity(SEFD,n_ant,t_obs,bw,n_pol,eta_s=1):
    """
    Based on Wrobel & Walker 1999 Synthesis Imaging -- page 

    equation: 9-23, 9-24

    SEFD        [J/m^2]
    n_ant      = number of antennas  (array of equal antennas)
    t_obs      = total observation time [s]
    bw         = band width       [Hz]
    n_pol      = number of polarisation (XX YY or RR LL --> maximum 2)
    eta_s      = overall system efficiency (electronic and digital losses)
    """
    from math import sqrt

    delta_image = (1/eta_s) * SEFD / sqrt( n_ant * (n_ant - 1) * t_obs * bw * n_pol ) 

    return(delta_image)

def baseline_sensitivity(SEFD1,SEFD2,bw,t_int,eta_s=1):
    """
    Based on Wrobel & Walker 1999 Synthesis Imaging -- page 

    equation: 9-14

    SEFD1      for antenna 1   [J/m^2]
    SEFD2      for antenna 2   [J/m^2]

    t_int     = integration time [s]
    bw        = band width       [Hz]
    eta_s     = overall system efficiency (electronic and digital losses)
    """
    from math import sqrt

    rms_bsl = 1/eta_s * sqrt( SEFD1 * SEFD2 / ( 2 * bw * t_int))

    return(rms_bsl)


def image_sensitivity_inhomogenious_array(N_MK,SEFD_MK,N_MKplus,SEFD_SKA,t_int,bw,n_pol,array_eff_mkplus=1):
    """
    Based on Wrobel & Walker 1999 Synthesis Imaging -- page 

    equation: 9-14

    SEFD        [J/m^2]                           # array per antenna 
    t_int      = integration time [s]
    bw         = band width       [Hz]
    n_pol      = number of polarisation (XX YY or RR LL --> maximum 2)
    eta_s      = overall system efficiency (electronic and digital losses)

    I have placed this formula also at the MK+ page
    https://www.meerkatplus.tel/image-sensitivity-for-a-heterogenous-array/
    """
    import numpy as np
    from math import sqrt


    # get the number of baselines
    #
    N_tot           =  N_MK + N_MKplus
    N_bsl_tot       =  N_tot * (N_tot -1) / 2                   # total number of baselines
    N_bsl_MK        =  N_MK * (N_MK -1) / 2                     # pure MK Antenna baselines
    N_bsl_MKplus    =  N_MKplus * (N_MKplus -1) / 2             # pure SKAMID antenna baselines
    N_bsl_MK_MKplus =  N_bsl_tot -  N_bsl_MK - N_bsl_MKplus     # intermixed baselines
    # ############################

    image_sensitivity_MKplus =  1/array_eff_mkplus * sqrt( 1 / (n_pol * N_tot * (N_tot - 1) * delta_nu * t_obs) * (SEFD_MK**2 * N_bsl_MK + SEFD_SKA**2 * N_bsl_MKplus + SEFD_SKA * SEFD_MK * N_bsl_MK_MKplus) / N_bsl_tot )
    #print('MeerKATplus image sensitivity: ',image_sensitivity_MKplus,'[Jy]')

    return(image_sensitivity_MKplus)


def obs_band(obsfreq):
    """
    return the observation band
    https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/277315585/MeerKAT+specifications
    """
    if obsfreq > 1.749E9 and obsfreq < 3.51E9:
        return('SBAND')
    if obsfreq > 0.856E9 and obsfreq < 1.712E9:
        return('LBAND')
    if obsfreq > 0.544E9 and obsfreq < 1.088E9:
        return('UHFBAND')

def array_phase_center(arrayname):
    """
    https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/277315585/MeerKAT+specifications
    """
    phase_center = []

    if arrayname == MK:
        phase_center = ['30d,42m,39.8.s','21d26m38.0s','1086.6m']

    return phase_center

def telescope_array_info():
    """
    specific telescope array information
    """

    # provides the Tsys  for the theoretical calculations 
    # NOT used since 06/23 use the SEFD provided in the MK+ 
    # page
    #sys_info = nested_dict({'MEERKAT':{}})  # creates a nested dictionary
    #sys_info['MeerKAT']['LBAND']['MKDISH']['TSYS']   = 22.072 
    #sys_info['MeerKAT']['LBAND']['MKDISK']['ETA_A']  = 1 
    #sys_info['MeerKAT']['LBAND']['SKADISH']['TSYS']  = 17.964 
    #sys_info['MeerKAT']['LBAND']['SKADISK']['ETA_A'] = 1 
    
    return(22.072,1)



def datamodelstats(data,model=[],dostatsmeasure='sumdiffsquare'):
    """
    """
    
    if len(model) > 0:
        if dostatsmeasure == 'sumdiffsquare':
            return(np.sum((data - model)**2))

        elif dostatsmeasure == 'meandiff':
            return(np.mean(data - model))

        elif dostatsmeasure == 'maxdiff':
            return(np.max(data - model))

        elif dostatsmeasure == 'mindiff':
            return(np.min(data - model))

        elif dostatsmeasure == 'kurtosisdiff':
            return(kurtosis(data - cmodel,nan_policy='omit'))

        elif dostatsmeasure == 'skewnessdiff':
            return(skew(cdata - cmodel,nan_policy='omit'))

        else:
            'CAUTION STATS METHOD NOT KNOWN'
            sys.exit(-1)
    else:

        if dostatsmeasure == 'datamean':
            return(np.mean(data))

        elif dostatsmeasure == 'datamin':
            return(np.max(data))

        elif dostatsmeasure == 'datamax':
            return(np.max(data))

        elif dostatsmeasure == 'datakurtosis':
            return(kurtosis(data,nan_policy='omit'))

        elif dostatsmeasure == 'dataskewness':
            return(skew(cdata,nan_policy='omit'))

        else:
            'CAUTION STATS METHOD NOT KNOWN'
            sys.exit(-1)


def getbase2dim(imdim):
    """
    return the base 2 dimension
    and check if the nedx dimension is to large
    """
    
    bin_size = 2
    for d in range(int(imdim)):
        size = bin_size ** d
        if size >= imdim:
            bin_size = size
            break_d = d
            break

    up   = bin_size - imdim
    low  = imdim -  2 ** (break_d-1)

    if up > low:
        bin_size = 2 ** (break_d-1)
        
    return bin_size
        

# this class is for json dump
# https://stackoverflow.com/questions/75475315/python-return-json-dumps-got-error-typeerror-object-of-type-int32-is-not-json
# https://docs.python.org/3/library/json.html
#
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)

def save_to_json(data,dodatainfoutput,homedir):
    """
    safe information into a json file
    """

    with open(homedir + dodatainfoutput, 'w') as fout:
        json_dumps_str = json.dumps(data,indent=4,sort_keys=False,separators=(',', ': '),cls=NumpyArrayEncoder)
        print(json_dumps_str, file=fout)
    return homedir + dodatainfoutput

