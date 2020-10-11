# Code accompanying the paper "Multivariate cross-frequency coupling via generalized eigendecomposition"
## Michael X Cohen, 2017, Elife

https://elifesciences.org/articles/21792

Each "method" script runs on its own and produces plots. Hopefully it's sufficiently commented to allow modifications.

If you have the eeglab toolbox, you can replace the topoplotIndie function with the topoplot function. topoplotIndie.m is a modified version of the eeglab topoplot.m function (all credit goes to eeglab); the Indie function works without dependencies, though with some loss of functionality.

Method 4 also uses the eeglab function eegfilt (included; all credit to eeglab).

Questions? -> mikexcohen@gmail.com

