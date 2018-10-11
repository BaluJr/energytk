# EnergyTK: Energy Toolkit
Energy Toolkit has been startted as an extension of NilmTK. It became apparent that the utilized basic classes are perfectly capable to also support additional usecases apart from mere Nilm. 
Short-Term Load Forecasting and Simulation of Powerflows 

Non-Intrusive Load Monitoring (NILM) is the process of estimating the
energy consumed by individual appliances given just a whole-house
power meter reading.  In other words, it produces an (estimated)
itemised energy bill from just a single, whole-house power meter.

NILMTK is a toolkit designed to help *researchers* evaluate the
accuracy of NILM algorithms. 

**NILMTK is not being actively developed now. However, we believe it does the job it's intended to do! 
It may take time for the original NILMTK authors to get back to you regarding queries/issues. However, you are more than welcome to propose changes, support!**.

# Documentation

[NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual)

# Why a toolkit for NILM?

We quote our [NILMTK paper](http://arxiv.org/pdf/1404.3878v1.pdf)
explaining the need for a NILM toolkit:

  > Empirically comparing disaggregation algorithms is currently
  > virtually impossible. This is due to the different data sets used,
  > the lack of reference implementations of these algorithms and the
  > variety of accuracy metrics employed.


# What NILMTK provides

To address this challenge, we present the Non-intrusive Load Monitoring
Toolkit (NILMTK); an open source toolkit designed specifically to enable
the comparison of energy disaggregation algorithms in a reproducible
manner. This work is the first research to compare multiple
disaggregation approaches across multiple publicly available data sets.
NILMTK includes:

-  parsers for a range of existing data sets (8 and counting)
-  a collection of preprocessing algorithms
-  a set of statistics for describing data sets
-  a number of [reference benchmark disaggregation algorithms](https://github.com/nilmtk/nilmtk/wiki/NILM-Algorithms)
-  a common set of accuracy metrics
-  and much more!


# Publications

Please see our [list of NILMTK publications](http://nilmtk.github.io/#publications).  If you use NILMTK in academic work then please consider citing our papers.

Please note that NILMTK has evolved *a lot* since these papers were published! Please use the
[online docs](https://github.com/nilmtk/nilmtk/tree/master/docs/manual)
as a guide to the current API.


# Keeping up to date with NILMTK

* [NILMTK-Announce mailing list](https://groups.google.com/forum/#!forum/energytk-announce): stay up to speed with NILMTK.  This is a low-traffic mailing list.  We'll just announce new versions, new docs etc. Not yet online!


# History

* April 2014: NilmTK v0.1 released
* June 2014: NILMTK presented at [ACM e-Energy](http://conferences.sigcomm.org/eenergy/2014/)
* July 2014: NilmTK v0.2 released
* Nov 2014: NILMTK wins best demo award at [ACM BuildSys](http://www.buildsys.org/2014/)
* Mai 2018: EnergyTK is announced as the official successor of NilmTK, introducing several additional usecases apart from Nilm.
