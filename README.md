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

Maybe the easiest way to get going is to run a container based
installation from the
['stimela project'](https://github.com/ratt-ru/Stimela) from Sphe
Makhathini. Of course you need to get
['singularity'](https://sylabs.io/docs/#singularity) or
['docker '](https://docs.docker.com/get-docker/) installed
first on your machine.

After that build an image, here e.g. we build a singularity image:

```
$ singularity pull docker://stimela/ragavi:1.7.3
```



Obtain Information of your Observations 
=============

```
$ singularity exec --bind "$PWD" /PATH_TO_SINGULARITY_CONTAINER/ragavi_1.7.3.sif python3 GET_MS_INFO.py --h
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
$ singularity exec --bind "$PWD":/work /SOFTWARE/CONTAINERS/ragavi_1.2.6.sif python3 /work/GET_MS_INFO.py --WORK_DIR=/work/ --MS_FILE=J2339-5523_multispw_small.ms
```

Example output:

```
telescope                   :   MeerKAT                                                                                                    
project ID                  :   20200612-0003                                                                                              
observation timerange (UTC) :   2020-06-14 05:51:27.925  ---  2020-06-14 06:09:59.455                                                      
number of individual scans  :   1                                                                                                          
number of antennas          :   59                                                                                                         
antenna diameter        [m] :   [13.5]
array type                  :   HOMOGENEOUS
field of view  (FoV)  [deg] :   [ 1.46 , 1.58 ]
baseline length         [m] :   [ 29.26 , 7697.58 ]
angular resolution [arcsec] :   [ 7.69 , 8.35 ]
imagesize           [pixel] :   2220.22
cellsize     [arcsec/pixel] :   2.564174
polarisation property       :   ['XX', 'YY']
spectral windows     [SPWD] :   3
total frequency range  [Hz] :   9.815865e+08   --   1.065180e+09
center frequency       [Hz] :   1.023383e+09 
total bandwidth        [Hz] :   8.359375e+07 
total number of channels    :   201
channels width         [Hz] :   [4.179688e+05 , 4.179688e+05 ]
observed sources            :   ['J2339-5523']
field id                    :   ['0']
time per source         [s] :   [1111]
integration time        [s] :   [ 8.0 , 8.0 ]
image sensitivity      [Jy] :   [1.688879303107692e-05]
baseline sensitivity   [Jy] :   [0.011645080590217273]

detailed frequency information
        --------------
        SPWD_ID         : 0
        frequencies     : 9.815865e+08  --  1.009172e+09
        bandwidth       : 2.758594e+07
        channels        : 67
        channel width   : 4.179688e+05 , 4.179688e+05
        --------------
        SPWD_ID         : 1
        frequencies     : 1.009590e+09  --  1.037176e+09
        bandwidth       : 2.758594e+07
        channels        : 67
        channel width   : 4.179688e+05 , 4.179688e+05
        --------------
        SPWD_ID         : 2
        frequencies     : 1.037594e+09  --  1.065180e+09
        bandwidth       : 2.758594e+07
        channels        : 67
        channel width   : 4.179688e+05 , 4.179688e+05



detailed source information
        --------------
         J2339-5523 | SCAN_ID  12  |  2020-06-14 05:51:27.925 --- 2020-06-14 06:09:59.455
        --------------

```


Obtain Spectra of your Observations 
=============

```
$ singularity exec --bind "$PWD" /PATH_TO_SINGULARITY_CONTAINER/ragavi_1.7.3.sif python3 DYNAMIC_SPECTRUM_PLOTTER.py --h
```

list of arguments

```
Options:
  -h, --help            show this help message and exit
  --MS_FILE=MSFILE      MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms
  --DATA_TYPE=DATACOLUMN
                        which data column to use [defaul DATA]
  --FIELD_ID=FIELD_ID   if MS contains muliple field define on field
  --DOBSLWATERFALLSPEC  produce waterfall spectrum per baseline
  --DOPLOTAVGWATERFALLSPEC
                        produce an averaged waterfall sepctrum
  --DOPLOTAVGSPECTRUM   produce an average spectrum
  --PLOTFILEMARKER=PLTF_MARKER
                        add file indicator in front of the file [defaut =
                        PLT_]
  --SHOW=SHOWPARAMETER  = Show the amplitude [AMP] or the [PHASE] [default is
                        AMP]
  --DONOTSORTUVDIS      use original sequence of baselines. [default sort
                        versus UV-distance ]
  --CHANNELSLIDE=CHNSLIDE
                        select channel range to plot [channel1,channel2]
  --SELECT_BSL=SELECT_BSL
                        select baselines (e.g. [[ANT1,ANT2],[ANT3,ANT8]])
  --SELECT_ANT=SELECT_ANT
                        select antennas (e.g. [ANT1,ANT2,ANT3])
  --SELECT_UVDIS=SELECT_UVDIS
                        select baselines via UV distance (e.g. [0,100] in
                        meter)
  --TESTFLAG=TESTFG     test flag data channels
                        [[channel1,channel2],[channel1,channel2]]
  --DOPROGRESSBAR       show progress bar

```


Plot data versus time and model per basline (VPLOT)
=============

```
$ singularity exec --bind "$PWD" /PATH_TO_SINGULARITY_CONTAINER/ragavi_1.7.3.sif python3 VPLOT_DATA_MODEL.py --h
```

list of arguments

```
Options:
  -h, --help            show this help message and exit
  --MS_FILE=MSFILE      MS - file name e.g. 1491291289.1ghz.1.1ghz.4hrs.ms
  --DATA_TYPE=DATACOLUMN
                        which data column to use [defaul DATA]
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
$ singularity exec --bind ${PWD}:/data /PATH_TO_SINGULARITY_CONTAINER/ragavi_1.7.3.sif python3 /data/FLAG_IT.py --h
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

2) based on the averages a new fg mask can be produced via the DYNAMIC_SPECTRUM_PICKLE_PLTFLG.py
    --DOFLAGDATA --DO_SAVE_FLAG_MASK=

3) before you apply the new mask you may want to save the current
flags  use FLAG_IT.py --CASAFGSAVE 

4) use that output to load into FLAG_IT.py as --FGMASK_FILE=




As a Note
=============

If you want to plot very efficient the visibilities please do not use
this tool use instead Olegs ['shadeMS'](https://github.com/ratt-ru/shadeMS).


DASK Documentation
=============

https://dask-ms.readthedocs.io

https://gitter.im/dask-ms/community

