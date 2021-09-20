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
array type                  :   HOMOGENIOUS
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


More to come 
=============
- Perecentage of flagging
- Baseline waterfall spectra


As a Note
=============

If you want to plot very efficient the visibilities please do not use
this tool use instead Olegs ['shadeMS'](https://github.com/ratt-ru/shadeMS).


DASK Documentation
=============

https://dask-ms.readthedocs.io

https://gitter.im/dask-ms/community

