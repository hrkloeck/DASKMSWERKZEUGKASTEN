from daskms import xds_from_ms,xds_from_table
import dask
import dask.array as da

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
import numpy as np
import sys


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

def ms_obs_info(msdata):

    ob_info = xds_from_table(msdata+'::OBSERVATION',group_cols='__row__')

    obs_info = {}
    obs_info['TELESCOPE_NAME'] = list(ob_info[0].TELESCOPE_NAME.values)[0]
    obs_info['PROJECT'] = list(ob_info[0].PROJECT.values)[0]

    ob_infof = xds_from_table(msdata+'::FEED')
    obs_info['ANTS'] = list(ms_unique_antenna(msdata))  

    ob_infoa = xds_from_table(msdata+'::ANTENNA')
    obs_info['DISH_DIAMETER'] = list(ob_infoa[0].DISH_DIAMETER.values)  


    return obs_info


def ms_unique_antenna(msdata):
    """
    """
    ob_infof = xds_from_table(msdata+'::FEED')

    return(np.unique(list(ob_infof[0].ANTENNA_ID.values)))


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


def ms_get_bsl_data(msdata,field_idx=0,setspwd=-1,bsls=[[0,1],[1,2]],bsl_idx=[0,2]):
    """
    return bsl selected dataset for a single field/source
    in a nested dictionary

    Dictionary keywords are: 
    [baseline index][spectral window][DATA]
    [baseline index][spectral window][FLAG]
    [baseline index][spectral window][MODEL]   (only if data is present)
    [baseline index][TIME]
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

                        if get_model_data != -1:
                            sub_data_bsl[blidx]['MODEL'] = []
                else:

                    if msds.attrs['DATA_DESC_ID'] == spwd:
                        sub_data_bsl[blidx] = {}
                        sub_data_bsl[blidx]['DATA']      = []
                        sub_data_bsl[blidx]['FLAG']      = []

                        if get_model_data != -1:
                            sub_data_bsl[blidx]['MODEL'] = []


                # Selecting baseline data
                #
                sel_bsl    = da.logical_and(msds.ANTENNA1.data == bsls[blidx][0],msds.ANTENNA2.data == bsls[blidx][1])    

                #bsl_data   = dask.compute(msds.DATA.data[sel_bsl])

                # store DATA and FLAG's 
                sub_data_bsl[blidx]['DATA'].append(msds.DATA.data[sel_bsl])
                sub_data_bsl[blidx]['FLAG'].append(msds.FLAG.data[sel_bsl])

                # if MODEL presentg inculde also MODEL
                if get_model_data != -1:
                    sub_data_bsl[blidx]['MODEL'].append(msds.MODEL_DATA.data[sel_bsl])


                if msds.attrs['DATA_DESC_ID'] == 0:
                    # get the time
                    # CAUTION THIS COULD BE DANGEROUS IF SPWD HAVE DIFFERENT TIME-RANGES
                    sub_data_bsl[blidx]['TIME_CENTROID'] = msds.TIME_CENTROID.data[sel_bsl]

                    # get the frequency
                    sub_data_bsl[blidx]['CHAN_FREQ'] = spwd_freq

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
       pfile  = open(filename,'r')
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


def SEFD(diameter,T_sys,eta_a):
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


def image_sensitivity_inhomogenious_array(SEFD1,SEFD2,t_int,bw,n_pol,eta_s=1):
    """
    Based on Wrobel & Walker 1999 Synthesis Imaging -- page 

    equation: 9-14

    SEFD        [J/m^2]                           # array per antenna 
    t_int      = integration time [s]
    bw         = band width       [Hz]
    n_pol      = number of polarisation (XX YY or RR LL --> maximum 2)
    eta_s      = overall system efficiency (electronic and digital losses)

    the error is calculated based on the baseline sensitivities
    """
    import numpy as np
    from math import sqrt

    # Getting the individual combinations
    baselines      = [ [i,j] for i in range(len(SEFD1)) for j in range(len(SEFD2)) if i<j ]

    SEFD_FULL   = 0
    for bl in baselines:
        # sum over the individual errors
        # (delta x)**2 = sum_1_N (delta_xi / N)**2
        SEFD_FULL += ( baseline_sensitivity(SEFD1[bl[0]],SEFD2[bl[1]],bw,t_int) / len(baselines) )**2

    delta_array = 1/eta_s * sqrt(SEFD_FULL) / sqrt(n_pol)

    return(delta_array)

def obs_band(obsfreq):
    """
    return the observation band
    """
    if obsfreq > 1.7E9 and obsfreq < 3.5E9:
        return('SBAND')
    if obsfreq > 0.8E9 and obsfreq < 1.7E9:
        return('LBAND')

def telescope_array_info():
    """
    specific telescope array information
    """

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

        





#h = telescope_array_info()
#print(h)

#sys.exit(-1)
#import numpy as np

# input for SEFD
#diameter = np.array([32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32])
#tsys     = np.array([45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45])
#eta_a    = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#
#
#t_observing = 600
#bandwidth   = 0.8E9
#nstokes     = 2
#
#eta_c       = 1

#nantenna    = len(diameter)

#print(image_sensitivity(SEFD(diameter[0],tsys[0],eta_a[0]),nantenna,t_observing,bandwidth,nstokes))

#sefd_array = SEFD(diameter,tsys,eta_a)
#print(image_sensitivity_inhomogenious_array(sefd_array,sefd_array,t_observing,bandwidth,nstokes,eta_c))

#sys.exit(-1)

#ssssa = SEFD(32,45,1)
#print(image_sensitivity(ssssa,10,60,1E-9,1))

#eta_calc = image_sensitivity_inhomogenious(ssss,ssss,10,60,1E-9,1)/image_sensitivity(ssssa,10,60,1E-9,1)
#print(eta_calc)

#print(image_sensitivity_inhomogenious(ssss,ssss,10,60,1E-9,1,eta_calc))

#print(SEFD(13.5,22.072,1))
