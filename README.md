# DASKMSWERKZEUGKASTEN

---

The purpose of this toolbox is to provide information from radio
observations and access the raw data from an astronomers perspective
playing with the data. Hope you will enjoy it.


There are various ways to extract information from a ['Measurement
Set'](https://casa.nrao.edu/casadocs/casa-5.1.1/reference-material/measurement-set)
using CASA, casacore, etc and the varaity adds an additional threshold if
one just want to figure some things out.


I was looking for a general tool that allows to acces but also to edit
the MS datasets in a more coherent framework, so I'm using
['MS-DASK'](https://github.com/ska-sa/dask-ms) and will add to all the
confusing software packages availibe on the market and put together
the DASKMS Werkzeugkasten.

## Installation 
The only thing to use it you may want to get ['MS-DASK'](https://github.com/ska-sa/dask-ms)
installed. 


Maybe the easiest is to do a container based on the ['stimela project'](https://github.com/ratt-ru/Stimela)

get singularity installed on your machine

.. code-block:: bash

	singularity pull docker://stimela/ragavi:1.7.3

this includes all you need to get running.


Example how to run the singularity image
=============

.. code-block:: bash

	singularity exec --bind "$PWD":/work
	/PATH_TO_SINGULARITY_CONTAINER/ragavi_1.2.6.sif python3
	/work/DASKMS_PLAYGROUND.py --HELP



DASK Documentation
=============

https://dask-ms.readthedocs.io

https://gitter.im/dask-ms/community

