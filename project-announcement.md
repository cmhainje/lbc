# SDSS-V Project 0197: Announcement

## Title

Reducing calibration overheads with a generative model for spectrograph calibration data

## Description

Currently arc exposures are taken for each BOSS visit, in part because gravitational and environmental effects on the telescope-mounted spectrographs affect the calibration state substantially. We believe that this calibration state, including the wavelength solution and trace location, may depend on only a small number of latent and measurable parameters, such as temperature and positioning (or history of positioning) of the instrument. If a forward model of this relationship can be discovered, it could significantly reduce the frequency of calibrations needed. We will use machine learning methods to build generative models of the relationships, i.e., models that can generate calibration data from the relevant housekeeping data (including possibly information gleaned from the science exposures). Our methods will probably be similar to *excalibur* (Zhao+ 2021, [arXiv:2010.13786](https://arxiv.org/pdf/2010.13786.pdf)), but we will instead use non-linear models (as will be required in the case of BOSS). An accurate generative model could significantly reduce calibration overhead or increase calibration precision or both, and thus improve overall survey efficiency. If successful, we may do the same for APOGEE, which might have significantly different considerations.
