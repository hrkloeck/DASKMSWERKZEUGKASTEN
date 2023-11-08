# DASKMSWERKZEUGKASTEN

---

The purpose of this Werkzeugkasten (toolbox) is to extract information and access raw
data of a
['Measurement Set'](https://casa.nrao.edu/Memos/229.html),
having in mind an astronomers perspective who wants to play with the
data. Hope you will enjoy it.

Of course there are many other solutions on the market to do that, but
I was looking for a general tool that allows to access, but also to edit
the MS datasets in a more coherent framework and I found the nice
stuff from Simon Perkins
['MS-DASK'](https://github.com/ska-sa/dask-ms).


## Installation 

Please have a look ['Singularity'](https://github.com/hrkloeck/DASKMSWERKZEUGKASTEN/tree/main/Singularity)

Once the singularity container is produced. Clone the
DASKMSWERKZEUGKASTEN into your working directory, where your
Measurement Set (MS) is present.

git clone https://github.com/hrkloeck/DASKMSWERKZEUGKASTEN.git


## Obtain Information of your Observations 
=============

```
$ singularity exec --bind ${PWD}:/work /PATH_TO_SINGULARITY_CONTAINER/WK.simg python3 /work/DASKMSWERKZEUGKASTEN/GET_MS_INFO.py -h
```

list of arguments

```
Options:
  -h, --help            show this help message and exit
  --MS_FILE=MSFILE      MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms
  --WORK_DIR=CWD        Points to the current working directory (e.g. usefull for
                        containers)
  --DOINFO_TAB=GETINFOTAB
                        Show MS table info only [default ALL tables, else use table
                        name]
  --DO_MS_INFO_JSON=DODATAINFOUTPUT
                        Output file name. Information stored in JSON format.
  --NOT_PRINT_MS_INFO   Stop printing MS info. Useful in pipelines

```

Lets assume your ragavi container sits in /SOFTWARE/CONTAINER and your
MS file is in the current working directory

```
$ singularity exec --bind ${PWD}:/work /PATH_TO_SINGULARITY_CONTAINER/WK.simg python3 /work/DASKMSWERKZEUGKASTEN/GET_MS_INFO.py --WORK_DIR=/work/ --MS_FILE=/work/1678454471_sdp_l0.ms.hann.spw.split
```


Example output (Note if the JSON option is used even more info is stored):

```
telescope                      :   MeerKAT
project ID                     :   20230209-0012
observation timerange (UTC)    :   2023-03-10 13:28:46.988  ---  2023-03-10 15:23:19.607
number of individual scans     :   18
number of antennas             :   52
antenna diameter        [m]    :   [13.5]
array type                     :   HOMOGENEOUS
field of view  (FoV)  [deg]    :   [ 1.09 , 1.58 ]
baseline length         [m]    :   [ 29.26 , 7697.55 ]
angular resolution [arcsec]    :   [ 2.88 , 4.16 ]
imagesize           [pixel]    :   5910.14
cellsize     [arcsec/pixel]    :   0.960535
polarisation property          :   ['XX', 'XY', 'YX', 'YY']
spectral windows     [SPWD]    :   16
total frequency range  [Hz]    :   1.968750e+09   --   2.843536e+09
center frequency       [Hz]    :   2.406143e+09 
total bandwidth        [Hz]    :   8.747864e+08 
total number of channels       :   4096
channels width         [Hz]    :   [2.136230e+05 , 2.136230e+05 ]
observed sources               :   ['J0521+1638', 'J0252-7104', 'J0413-8000', 'J0408-6545']
field id                       :   ['0', '1', '2', '3']
time per source         [s]    :   [777, 945, 4191, 478]
integration time        [s]    :   [ 2.0 , 2.0 ]
image sensitivity      [Jy]    :   [7.091437421061913e-06, 6.43026512980777e-06, 3.053415836789292e-06, 9.0412990848
15918e-06]
baseline sensitivity   [Jy]    :   [0.007191466759137397]
antennas close to array center :   ['m029', 'm028', 'm026', 'm003', 'm027', 'm000']


detailed frequency information

      --------------
        SPWD_ID         : 0
        frequencies     : 1.968750e+09  --  2.023224e+09
        bandwidth       : 5.447388e+07
        channels        : 256
        channel range   : [0, 256]
        channel width   : 2.136230e+05 , 2.136230e+05
        --------------
        SPWD_ID         : 1
        frequencies     : 2.023438e+09  --  2.077911e+09
        bandwidth       : 5.447388e+07
        channels        : 256
        channel range   : [256, 512]
        channel width   : 2.136230e+05 , 2.136230e+05
        --------------

        ...

        --------------
        SPWD_ID         : 15
        frequencies     : 2.789062e+09  --  2.843536e+09
        bandwidth       : 5.447388e+07
        channels        : 256
        channel range   : [3840, 4096]
        channel width   : 2.136230e+05 , 2.136230e+05



detailed source information

       --------------
         J0521+1638 
                         [RA, DEC] ['05h21m09.89s +16d38m22.1s'] 
                         [RA, DEC] [80.29120833333333, 16.63947222222222] 
                         [gl,gb]   [187.4049449769659, -11.343472072144888]
         J0252-7104 
                         [RA, DEC] ['02h52m46.15s -71d04m35.3s'] 
                         [RA, DEC] [43.19229166666666, -71.07647222222222] 
                         [gl,gb]   [290.01697438965937, -42.87928165462044]
         J0413-8000 
                         [RA, DEC] ['04h13m26.4s -80d00m00s'] 
                         [RA, DEC] [63.36, -80.0] 
                         [gl,gb]   [293.71889610733257, -33.18594156918596]
         J0408-6545 
                         [RA, DEC] ['04h08m20.38s -65d45m09.1s'] 
                         [RA, DEC] [62.08491666666668, -65.75252777777777] 
                         [gl,gb]   [278.6439190098787, -40.878622771911154]

        --------------
         J0521+1638     angular distance to      J0252-7104      91.3201319121865  [deg]
         J0521+1638     angular distance to      J0413-8000      97.05563497164333  [deg]
         J0521+1638     angular distance to      J0408-6545      83.529236270583  [deg]
         J0252-7104     angular distance to      J0413-8000      10.119502720097923  [deg]
         J0252-7104     angular distance to      J0408-6545      8.693548534832816  [deg]
         J0413-8000     angular distance to      J0408-6545      14.251582653311665  [deg]
        --------------




detailed source information

        --------------
         J0521+1638 | SCAN_ID  2  |  2023-03-10 13:28:46.988 --- 2023-03-10 13:33:45.536
         J0521+1638 | SCAN_ID  31  |  2023-03-10 15:15:20.727 --- 2023-03-10 15:23:19.607
        --------------
         J0252-7104 | SCAN_ID  4  |  2023-03-10 13:34:59.672 --- 2023-03-10 13:36:57.889
         J0252-7104 | SCAN_ID  7  |  2023-03-10 13:47:41.070 --- 2023-03-10 13:49:41.291


        ...

	     J0408-6545 | SCAN_ID  29  |  2023-03-10 15:05:53.686 --- 2023-03-10 15:13:52.566
        --------------

```


Obtain Spectra of your Observations 
=============

```
$ singularity exec --bind ${PWD}:/work /PATH_TO_SINGULARITY_CONTAINER/WK.simg python3 /work/DASKMSWERKZEUGKASTEN/DYNAMIC_SPECTRUM_PLOTTER.py -h
```

list of arguments

```
Options:
  -h, --help            show this help message and exit
  --MS_FILE=MSFILE      MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms
  --DATA_TYPE=DATACOLUMN
                        which data column to use e.g. CORRECTED_DATA [default
                        DATA]
  --FIELD_ID=FIELD_ID   if MS contains muliple field define on field
  --SCAN_NUMBER=SCAN_NUM
                        select scan number [no default]. Note overwrites
                        FIELD_ID.
  --DOPLTSTDDATA        plot the stddata instead of the data
  --DOBSLWATERFALLSPEC  produce waterfall spectrum
  --DOPLOTAVGSPECTRUM   produce an average spectrum
  --DOPLOTNOMASKING     produce the plots using no mask
  --PLOTFILEMARKER=PLTF_MARKER
                        add file indicator in front of the file [defaut =
                        PLT_]
  --SHOW=SHOWPARAMETER  = Show the amplitude [AMP] or the [PHASE] [default is
                        AMP]
  --DONOTSORTUVDIS      use original sequence of baselines. [default sort
                        versus UV-distance ]
  --CHANNELSLIDE=CHNSLIDE
                        select channel range to plot [channel1,channel2]
  --SELECT_SPWD=SELECT_SPWD
                        select spectral window (default all)
  --SELECT_BSL=SELECT_BSL
                        select baselines (e.g. [[ANT1,ANT2],[ANT3,ANT8]])
  --SELECT_ANT=SELECT_ANT
                        select antennas (e.g. [ANT1,ANT2,ANT3])
  --SELECT_UVDIS=SELECT_UVDIS
                        select baselines via UV distance (e.g. [0,100] in
                        meter)
  --DO_SAVE_AVERAGE_DATA=DODATAINFOUTPUT
                        Generate dump of the averaged data via pickle file.
  --SWAPFIGURESIZE      show progress bar
  --DOPROGRESSBAR       show progress bar
  --WORK_DIR=CWD        Points to the working directory if output is produced
                        (e.g. usefull for containers)
  --DOPLTFIG            Plot Figure instead of printing.
  --DONOTICKS           show pixel instead of time ticks (good for hand flag)


```


![Example of a waterfall dynamic spectrum](https://github.com/hrkloeck/DASKMSWERKZEUGKASTEN/blob/main/Example_Images/DYNAMIC_SPECTRUM_WF_EXAMPLE.png)



Plot data versus time and model per basline (VPLOT)
=============

```
singularity exec --bind ${PWD}:/work /PATH_TO_SINGULARITY_CONTAINER/WK.simg python3 /work/DASKMSWERKZEUGKASTEN/VPLOT_DATA_MODEL.py --h
```

list of arguments

```
Options:
  -h, --help            show this help message and exit
  --MS_FILE=MSFILE      MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms
  --DATA_TYPE=DATACOLUMN
                        which data column to use [default DATA]
  --SHOW=SHOWPARAMETER  = Show the amplitude [AMP] or the [PHASE] [default is
                        AMP]
  --FIELD_ID=FIELD_ID   if MS contains muliple field define on field
  --SELECT_BSL=SELECT_BSL
                        select baselines (e.g. [[ANT1,ANT2],[ANT3,ANT8]])
  --SELECT_ANT=SELECT_ANT
                        select antennas (e.g. [ANT1,ANT2,ANT3])
  --SELECT_UVDIS=SELECT_UVDIS
                        select baselines via UV distance (e.g. [0,100] in
                        meter)
  --CHANNELSLIDE=CHNSLIDE
                        select channel range to plot [channel1,channel2]
  --PLOTFILEMARKER=PLTF_MARKER
                        add file indicator in front of the file [defaut =
                        PLT_]
  --DONOTSORTUVDIS      use original sequence of baselines. [default sort
                        versus UV-distance ]
  --DOPROGRESSBAR       show progress bar

```


Flag a dataset 
=============

```
$ singularity exec --bind "$PWD:/work" /PATH_TO_SINGULARITY_CONTAINER/ragavi_1.7.3.sif python3 /work/DASKMSWERKZEUGKASTEN/FLAG_IT.py --h
```

list of arguments

```
Options:
  -h, --help            show this help message and exit
  --MS_FILE=MSFILE      MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms
  --FGMASK_FILE=FGFILE  pickle - FG mask file
  --FGMASKALLDATA       apply the FG mask to all baselines (ignoring
                        selection)
  --ERASEALLFG          erase all the FG information
  --CASAFGTABNAME=CASAFGTABFILE
                        CASA FG table name
  --CASAFGSAVE          save the FG CASA table (using casa flagmanager)
  --CASAFGRESTORE       restore FG table in MS file (using casa flagmanager)
  --WORK_DIR=CWD        Points to the working directory if output is produced
                        (e.g. usefull for containers)

```

To flag a dataset you need to follow the procedure desriped below

1) produce an average spectrum pickle file with DYNAMIC_SPECTRUM_PLOTTER.py using the setting 
    --DO_SAVE_AVERAGE_DATA=
	
   Note: in case you do not use the selection option the average is
    base on all baselines (most flags but also the cleanest one)

```
$ singularity exec --bind ${PWD}:/work
/PATH_TO_SINGULARITY_CONTAINER/WK.simg python3
/work/DASKMSWERKZEUGKASTEN/DYNAMIC_SPECTRUM_PLOTTER.py --MS_FILE=/work/YOURDATA.ms
--WORK_DIR=/work/ --DOBSLWATERFALLSPEC --DO_SAVE_AVERAGE_DATA=AVERAGE_DATA
```

2) based on the averages (std of the averages is actually used) a new fg mask can be produced via the DYNAMIC_SPECTRUM_PICKLE_PLTFLG.py
    --DOFLAGDATA --DO_SAVE_FLAG_MASK=

```
$ singularity exec --bind ${PWD}:/work
/PATH_TO_SINGULARITY_CONTAINER/WK.simg python3
/work/DASKMSWERKZEUGKASTEN/DYNAMIC_SPECTRUM_PICKLE_PLTFLG.py --AVG_FILE=/work/PLT_J0408-6545_AVERAGE_DATA_pickle.py --DOFLAGDATA --DOBSLWATERFALLSPEC --DO_SAVE_FLAG_MASK=FLAG_DATA --WORK_DIR=/work/

```

3) before you apply the new mask you may want to save the current
flags use FLAG_IT.py --CASAFGSAVE 

4) use that output to load into FLAG_IT.py as --FGMASK_FILE=

```
$ singularity exec --bind ${PWD}:/work
/PATH_TO_SINGULARITY_CONTAINER/WK.simg python3 /work/DASKMSWERKZEUGKASTEN/FLAG_IT.py --MS_FILE=/work/YOURDATA.ms --WORK_DIR=/work/ --FGMASK_FILE=/work/DYNFLAG_J0408-6545_FLAG_DATA_pickle.py

```


![Example of a waterfall dynamic spectrum](https://github.com/hrkloeck/DASKMSWERKZEUGKASTEN/blob/main/Example_Images/DYNAMIC_SPECTRUM_WF_EXAMPLE.png)


As a Note
=============

If you want to plot very efficient the visibilities please do not use
this tool use instead Olegs ['shadeMS'](https://github.com/ratt-ru/shadeMS).


DASK Documentation
=============

https://dask-ms.readthedocs.io

https://gitter.im/dask-ms/community

