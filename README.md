PlateMate
====

PlateMate is a python package that helps you importing and analyzing data from plates with multiple weels (e.g., 96-well plates). The main idea is to construct a package capable of handling the most common tasks needed when analyzing data from multiple-well plates, and fully compatible with Jupyter Notebooks. This is still under development, but if you are interested in using and testing it, feel free to contact us.

It was originally written for the iGEM 2015 competition.


Example of applications of this package 
---

* [iGEM 2015, Team Brasil-USP](https://github.com/thmosqueiro/modeligem/wiki)
* [iGEM 2016, Team USP_UNIFESP-Brazil](https://github.com/VandroiyLabs/modeliGEM2016)



Installation
---

PlateMate is compatible with setup.py and pip. You can install PlateMate using pip:
```
pip install git+https://github.com/VandroiyLabs/platemate
```

It is very common to have Continuum's Anaconda installed and set as main Python environment -- especially in Mac OS X systems. If you are using Anaconda and wants to install it in Anaconda's environment, you can (for now) use the ```setup.py``` script. In summary,

```
git clone https://github.com/VandroiyLabs/platemate/
python setup.py install
```

If you don't want to use git, you can skip the first command above and simply download the latest version of PlateMate by [clicking here](https://github.com/VandroiyLabs/platemate/archive/master.zip). If all dependencies are satisfied, it should install platemate without a problem. 

The required dependences are minimal, and there is a high chance that you have them already.

* Python 2.7
* Numpy
* Matplotlib
* Pandas
* openpyxl
* itertools



Main collaborators:
---

* [Thiago Mosqueiro](http://thmosqueiro.vandroiy.com)
* [Jaqueline Brito](https://github.com/jaquejbrito)
* Lais Ribovski

If you have collaborated, please add your name and create a Pull Request.


License
---

PlateMate is free to use, free as in freedom and as in free beer. PlateMate is provided under GPL 3.0. Find a copy of the license in file [LICENSE](https://github.com/VandroiyLabs/platemate/blob/master/LICENSE).
