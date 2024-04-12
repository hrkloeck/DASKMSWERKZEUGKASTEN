# DASKMSWERKZEUGKASTEN Singularity 

---

## Installation 

Maybe the easiest way to get going is to run a container based
installation. For this you need to get
['singularity'](https://sylabs.io/docs/#singularity)
on your machine first.

Build the singularity image with (note you need fakeroot permission):

```
singularity build --fakeroot WK.simg singularity.werkzeugkasten.recipe
```

