# TODO
## General
- see literature for two and three color correction
- Often, use blue to correct for red errors
- see matlab software for additional features to implement
- could do dose form txt/csv file, or directly from filename
- can add plotting of three different curves
- definitely, add circular selection
- ~~add additional comments to the calibration~~ -> DONE
- Add auxiliary class? -> Not necessary

## Analysis
- Add interpolation of calibrations

## Dose Plot
- _Low pass filter to remove Dose noise_ -> not really necessary but nice to have

## Dose Histogram
- Add histogram of Dose vs mm (or px), taking a slice through the image, giving back a histogram of the dose
- To the above, add rectangle selection, average it on one axis(e.g. y), plot it vs the other axis (e.g. x)

## Calibration
- Implement list of files to use for calibration. This would prevent ROI overwriting in case of extracting ROI from non-calibration files.
