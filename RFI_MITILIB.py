# HRK 2023
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
# this is a first attemp to a RFI library to
# handle bad data in a waterfall spectrum
#
# Hope you enjoy it
# 
# --------------------------------------------------------------------


import numpy as np
import numpy.ma as ma
from copy import deepcopy
from scipy.signal import convolve2d


def data_stats(data,stats_type='mean'):
    """
    return mean and derivation of the input data
    """

    if stats_type == 'mad':
        #
        from astropy.stats import mad_std, median_absolute_deviation
        #
        data_mean      = median_absolute_deviation(data)
        data_std       = mad_std(data)

    elif stats_type == 'median':
        data_mean      = np.median(data)
        data_std       = np.std(data)

    elif stats_type == 'kdemean':
        data_mean,data_std = kdemean(data,accucary=1000)

    else:
        data_mean      = np.mean(data)
        data_std       = np.std(data)


    return data_mean, data_std, stats_type


def boundary_mask_data(data,reference_data,sigma,stats_type='mean',do_info=False):
    """
    use upper and lower thresholds to mask out data
    data is a unflagged (e.g. compressed dataset)
    reference_data original data

    """

    # determine the mean and std of the data
    #
    data_mean,data_std,stats_type = data_stats(data,stats_type)

    # selecing all data within the boundaries ion
    #
    select = np.logical_and(reference_data > data_mean - sigma * data_std, reference_data < data_mean + sigma * data_std)

    # Note in NUMPY MASKED arrays a bolean value of True (1) is considered to be masked out
    #
    # default all data is bad
    #
    data_shape    = np.array(reference_data).shape
    mask          = np.ones(data_shape)

    # so good data is indicated by zero 
    #
    mask[select]  = 0

    if do_info:
        print('data ',np.cumprod(data_shape)[-1],' markes as bad ',np.cumprod(data_shape)[-1]-np.count_nonzero(select))
        
    return mask.astype(bool)



def complete_fg_mask(mask,axis=0,percentage=0,complete_boundary=9):
    """
    Assuming a waterfall spectrum (frequencey(x-axis) versus time(y-axis)) 
    check the appearance of masking in time or channel (need to do axis 0)
    and determine if the entire channel or time should be masked
    completes_boundary up to X pixles

    assuming that True (masked) is bad data
    """

    new_mask   = deepcopy(mask.astype(int))

    max_fg     = new_mask.shape[axis]
    fg_sum     = mask.sum(axis=axis)/max_fg

    select       = fg_sum >= percentage/100.
    fg_axis      = np.arange(len(fg_sum))
    complete_fgs = fg_axis[select]

    if axis == 0:
        for fgc in range(len(complete_fgs)-1):
            if complete_fgs[fgc+1] - complete_fgs[fgc] < complete_boundary: 
                new_mask[:,complete_fgs[fgc]:complete_fgs[fgc+1]] = 1
    else:
        for fgc in complete_fgs:
            if complete_fgs[fgc+1] - complete_fgs[fgc] < complete_boundary: 
                new_mask[complete_fgs[fgc]:complete_fgs[fgc+1],:] = 1

    return new_mask.astype(bool)


def smooth_kernels(smk_type):
    """
    """
    # ----------------------------------------------
    #
    # here are some examples of kernels for an ambitions user that may want to play with it
    #
    if smk_type == 'box':
        kernel      = [[1,1,1],[1,1,1],[1,1,1]]        # boxcar
    if smk_type == 'cross':
        kernel      = [[0,1,0],[1,1,1],[0,1,0]]        # cross
    if smk_type == 'robx':
        kernel      = [[1,0],[0,-1]]                   # Roberts operator di/dx 
    if smk_type == 'roby':
        kernel      = [[0,1],[-1,0]]                   # Roberts operator di/dy
    if smk_type == 'scharrx':
        kernel      = [[-3,0,3],[-10,0,10],[-3,0,3]]   # Scharr operator di/dx
    if smk_type == 'scharry':
        kernel     = [[3,10,3],[0,0,0],[-3,-10,-3]]   # Scharr operator di/dy
    if smk_type == 'sobelx':
        kernel      = [[-1,0,1],[-2,0,2],[-1,0,1]]     # Sobel operator di/dx
    if smk_type == 'sobely':
        kernel      = [[1,2,1],[0,0,0],[-1,-2,-1]]     # Sobel operator di/dy
    if smk_type == 'canny':
        kernel       = [[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]] 
    if smk_type == 'prewittx':        
        kernel   = [[-1,0,1],[-1,0,1],[-1,0,1]]   # Prewitt operator di/dx
    if smk_type == 'prewitty':
        kernel   = [[1,1,1],[0,0,0],[-1,-1,-1]]   # Prewitt operator di/dy

    #ddxxfilter  = [[1,-2,1]]                    # differential
    #ddyyfilter  = [[1],[-2],[1]]                # differential 
    #dddxyfilter = [[-1/4.,0,1/4.],[0,0,0],[1/4.,0,-1/4.]]  # differential 
    #

    return kernel


def apply_multiple_magnitude_convolutions(data,data_mask,kernels,sigma,stats_type):
    """
    apply multiple convolutions
    """

    merged_data = np.zeros(data_mask.shape)

    for k in kernels:

        if isinstance(k, str):
            sm_kernel = smooth_kernels(k)
        else:
            sm_kernel = k

        # convolution image with filter kernel
        #
        conv_data        = convolve2d(data,sm_kernel,mode='same',boundary='symm')
        merged_data     += conv_data**2 

    merged_data      = np.sqrt(merged_data)

    conv_data_masked = ma.masked_array(merged_data,mask=data_mask,fill_value=np.nan)

    new_mask         = boundary_mask_data(conv_data_masked.compressed(),merged_data,sigma,stats_type,do_info=False)

    return new_mask



def apply_multiple_convolutions(data,data_mask,kernels,sigma,stats_type):
    """
    apply multiple convolutions
    """

    new_mask = np.zeros(deepcopy(data_mask).shape).astype(bool)

    nmasks   = []
    for k in kernels:

        # Check kernel
        #
        if isinstance(k, str):
            # get the smooth kernel
            sm_kernel = smooth_kernels(k)
        else:
            sm_kernel = k

        # convolution image with filter kernel
        #
        conv_data        = convolve2d(data,sm_kernel,mode='same',boundary='symm')
        conv_data_masked = ma.masked_array(conv_data,mask=new_mask,fill_value=np.nan)
        #
        nmasks           = boundary_mask_data(conv_data_masked.compressed(),conv_data,sigma,stats_type,do_info=False)

        new_mask = combine_masks(new_mask,[nmasks])

    return new_mask




def mask_2d_convolve(data,data_mask,smooth_kernel,sigma,stats_type):
    """
    CAUTION: data needs to be zero patted for the FFT treatment by numpy
    """

    # convolution f with g can also be seen as the Multiplication of their Fourier components:
    #
    # Fourier pairs (convolution =*, multiplication = x):    f * g    = F(f) x F(g)
    #                                                        F(f x g) = F(f) * F(g)
    # Mask_Array = Array x Mask
    # 
    # Convolving a masked array with a filter:  Mask_Array * Filter = F(Mask_Array) x F(Filter) = F(Array x Mask) x F(Filter) = (F(Array) * F(Mask)) x F(Filter) 
    #
    # So essentially to do a convolution of a masked array you need to first convolve the unmasked array with the mask and multiply it with the
    # Fouriertransform of the Filter.


    # convolution image with filter kernel
    #
    conv_data        = convolve2d(data,smooth_kernel,mode='same',boundary='symm')
    conv_data_masked = ma.masked_array(conv_data,mask=data_mask,fill_value=np.nan)

    new_mask         = boundary_mask_data(conv_data_masked.compressed(),conv_data,sigma,stats_type,do_info=False)

    return new_mask



def convolve_1d_data(data,smooth_type='hanning',smooth_kernel=3):
    """
    """
    from scipy.signal import wiener,gaussian,medfilt,convolve
    from scipy.signal.windows import hamming #, hanninng #hanning,convolve,hamming,gaussian,medfilt


    if smooth_type == 'hamming':
        sm_kernel = hamming(smooth_kernel)
        sm_data   = convolve(data,sm_kernel,mode='same') / sum(sm_kernel)

    elif smooth_type == 'gaussian':
        sm_kernel = gaussian(smooth_kernel,smooth_kernel)
        sm_data   = convolve(data,sm_kernel,mode='same') / sum(sm_kernel)

    elif smooth_type == 'median':
         sm_data = medfilt(data,smooth_kernel)

    elif smooth_type == 'wiener':
         sm_data = wiener(data,smooth_kernel)

    else:
        sm_data = deepcopy(data)

    return sm_data


def mask_convolve(data,data_mask,smooth_type='hanning',smooth_kernel=3,sigma=3,stats_type='mean',do_info=False):
    """
    basic flagging applies scipy filter function to the data 
    """
    from scipy.signal import wiener,hanning,convolve,hamming,gaussian,medfilt


    # get a smooth spectrum
    #
    sm_data = convolve_1d_data(data,smooth_type,smooth_kernel)


    # here divide the original data with the smoothed one
    n_data        = data/sm_data
    n_data_masked = ma.masked_array(n_data,mask=data_mask,fill_value=np.nan)
    
    # get the new mask
    #
    new_mask      = boundary_mask_data(n_data_masked.compressed(),n_data,sigma,stats_type,do_info=False)

    return new_mask



def scipy_masking(data,masktype='triangle',window_size=101):
    """
    """

    mask         = np.zeros(np.array(data).shape).astype(bool)

    if masktype == '':
        return mask.astype(bool)

    if masktype == 'triangle':
        from skimage.filters import threshold_triangle

        thresh_triangle = threshold_triangle(data)
        select = data > thresh_triangle


    if masktype == 'sauvola':
        from skimage.filters import threshold_sauvola

        thresh_sauvola = threshold_sauvola(data)
        select = data > thresh_sauvola


    if masktype == 'niblack':
        from skimage.filters import threshold_niblack

        thresh_niblack = threshold_niblack(data, window_size=window_size)
        select = data > thresh_niblack

    if masktype == 'niblack':
        from skimage.filters import threshold_sauvola

        thresh_sauvola = threshold_sauvola(data, window_size=window_size)
        select = data > thresh_sauvola


    if masktype == 'local':
        from skimage.filters import threshold_local

        thresh_local = threshold_local(data, block_size=window_size)
        select = data > thresh_local

    mask[select] = True

    return mask.astype(bool)


def kdemean(x,accucary=1000):
    """
     use the Kernel Density Estimation (KDE) to determine the mean
    
    (http://jpktd.blogspot.com/2009/03/using-gaussian-kernel-density.html )
    """
    from scipy.stats import gaussian_kde
    from numpy import linspace,min,max,std,mean
    from math import sqrt,log
    
    if mean(x) == std(x):
            print('kde mean = std')
            return(mean(x),std(x))

    max_range = max(np.abs([min(x),max(x)]))

    # create instance of gaussian_kde class
    gk     = gaussian_kde(x)

    vra    = linspace(-1*max_range,max_range,accucary)
    vraval = gk.evaluate(vra)

    # get the maximum
    #
    x_value_of_maximum = vra[np.argmax(vraval)]

    # Devide data
    difit = vraval / max(vraval)
    #
    # and select values from 0.5
    sel = difit >= 0.4999

    idx_half_power = list(difit).index(min(difit[sel]))

    if idx_half_power >= accucary -1:
        return(mean(x),std(x))

    delta_accuracy = max([abs(vra[idx_half_power-1] - vra[idx_half_power]),\
                              abs(vra[idx_half_power+1] - vra[idx_half_power])])

    fwhm = abs(x_value_of_maximum - vra[idx_half_power])


    # factor 2 is because only one side is evaluated
    sigma = 2*fwhm/(2*sqrt(2*log(2)))

    # safety net
    # is the KDE is not doing a good job
    #
    if sigma > std(x):
        return(mean(x),std(x))

    return(x_value_of_maximum,abs(sigma)+delta_accuracy)



def combine_masks(mask,listofmask):
    """
    combine mask of a list with an input mask
    """
    new_mask = deepcopy(mask.astype(bool))
    #
    for k in range(len(listofmask)):
        new_mask = np.logical_or(new_mask,listofmask[k].astype(bool))

    return new_mask


def difference_mask(mask,orgmask):
    """
    difference mask between mask and orgmask
    """
    
    new_mask             = deepcopy(mask).astype(bool)

    equal_mask           = np.logical_and(mask.astype(bool),orgmask.astype(bool))

    new_mask[equal_mask] = False

    return new_mask


def recover_coordinante_mask(mask,axis0,axis1,concat_freq_per_sw,concat_chan_per_sw,info_fg_str=''):
    """
    return the coordinates of the data
    """
    
    mask_int            = mask.astype(int)
    fg_mask_0,fg_mask_1 = np.nonzero(mask_int)

    mask_ccords = []
    for c in range(len(fg_mask_0)):
        obs_time    = axis0[fg_mask_0[c]]
        obs_freq    = axis1[fg_mask_1[c]]
        obs_freq_sw = concat_freq_per_sw[fg_mask_1[c]]
        obs_chan_sw = concat_chan_per_sw[fg_mask_1[c]]
        mask_ccords.append([obs_time,obs_freq,obs_chan_sw,obs_freq_sw,info_fg_str])

    return mask_ccords


def mask_into_spwd(final_mask,concat_time,concat_freq,concat_freq_per_sw,concat_chan_per_sw):
    """
    convert the data back into spectral windows
    """

    # Number of spwds
    #
    spwds            = int(concat_freq_per_sw[-1] + 1)

    if spwds > 1:
        #
        # reshape the mask into spwd,time,frequncy
        #
        data_shape              = final_mask.shape
        mask_spwd               = final_mask.reshape((spwds,data_shape[0],int(data_shape[1]/spwds)))
        concat_freq_per_sw_spwd = concat_freq_per_sw.reshape((spwds,int(data_shape[1]/spwds)))
        concat_chan_per_sw_spwd = concat_chan_per_sw.reshape((spwds,int(data_shape[1]/spwds)))
        concat_freq_spwd        = concat_freq.reshape((spwds,int(data_shape[1]/spwds)))
    else:
       mask_spwd               = final_mask 
       concat_freq_per_sw_spwd = concat_freq_per_sw
       concat_chan_per_sw_spwd = concat_chan_per_sw
       concat_freq_spwd        = concat_freq
       
    return mask_spwd,concat_freq_spwd,concat_time,concat_freq_per_sw_spwd


def interpolate_mask_data(data_x,data_y,org_data_x,mask):
    """
    """
    from scipy.interpolate import CubicSpline
    from scipy.interpolate import pchip_interpolate


    # get the bondaries for not fitting the edges
    #
    get_outer_bondaries = [np.min(list(np.argwhere(mask == 0))),np.max(list(np.argwhere(mask == 0)))]

    # optain good data
    #
    x,y = [],[]
    for i in range(len(data_x)):
        if mask[i] == False:
            x.append(data_x[i])
            y.append(data_y[i])

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    #
    cs    = CubicSpline(x, y,bc_type='not-a-knot',extrapolate=None)
    new_y = cs(org_data_x[get_outer_bondaries[0]:get_outer_bondaries[1]])
    new_y = pchip_interpolate(x, y,org_data_x[get_outer_bondaries[0]:get_outer_bondaries[1]])

    return get_outer_bondaries,new_y

def mask_true_false(mask,threshold=0.01):
    """
    input is mask int
    return a mask
    """
    new_mask = deepcopy(mask)
    new_mask = np.zeros(np.array(mask).shape)    

    # switch to be used
    #
    if threshold > 0:
        select   = mask > threshold
        new_mask[select] = 1.0

    return new_mask.astype(bool)

def check_mask(mask):
    """
    return info  of mask 
    """

    mask_zero_indx = len(np.argwhere(mask == 0))
    mask_ones_indx = len(np.argwhere(mask == 1))
    max_mask       = np.cumprod(mask.shape)[-1]
    
    return max_mask,mask_zero_indx,mask_ones_indx


def flag_data(data,inputmask,sigma,stats_type,percentage,smooth_kernels,threshold,flagbyhand):
    """
    return a new mask to be used for flagging
    """

    mask = mask_true_false(inputmask,threshold).astype(bool)


    # ==============

    # mask based on convolution of of the spectrum
    # this handles the background mitigation better 
    # then upper and lower boundary flag
    #
    #
    mask_conv  = apply_multiple_convolutions(data,mask,smooth_kernels,sigma,stats_type)


    #mask_conv  = apply_multiple_magnitude_convolutions(data,mask_conv,['prewittx','prewitty'],sigma,stats_type)


    # scipy flagging 
    #
    # window_size    = 101
    # scipy_masktype = 'triangle'
    #
    # scipy_mask    = scipy_masking(dyn_spec_std,scipy_masktype,window_size)


    # combine all the mask into a final one
    #
    combi_stuff     = combine_masks(mask,[mask_conv])


    # complete channels that are partly masked
    #
    final_mask      = complete_fg_mask(combi_stuff,axis=0,percentage=percentage,complete_boundary=9)


    # clean up the mask itself
    #
    cleanup_kernel = [[0,0,0],[1,1,1],[0,0,0]]
    final_mask     = mask_2d_convolve(final_mask.astype(np.float32),final_mask.astype(np.float32),cleanup_kernel,sigma,stats_type)
    #
    cleanup_kernel = [[1,1,1],[1,0,1],[1,1,1]]
    final_mask     = mask_2d_convolve(final_mask.astype(np.float32),final_mask.astype(np.float32),cleanup_kernel,sigma,stats_type)


    # Flag by hand
    #
    if len(flagbyhand[0]) > 1:
        for fg in flagbyhand:

            if len(fg) == 2:
                #print('flag channel')
                final_mask[:,fg[0]:fg[1]+1] = 1.0
            if len(fg) == 4:
                #print('flag region')
                final_mask[fg[1]:fg[3]+1,fg[0]:fg[2]+1] = 1.0

    # hand over a boolean mask
    # 
    final_mask     = final_mask.astype(bool)

   
    return final_mask


def flag_impact(final_mask,inputmask):
    """
    provide info about the flagging impact
    """

    # in case to check the impact of the flagging process
    #
    f_mask                = difference_mask(final_mask.astype(bool),inputmask.astype(bool))
    f_mask_info           = check_mask(f_mask)


    return f_mask_info




