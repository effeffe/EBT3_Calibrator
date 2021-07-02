from Calib import Calibrate, Fitting
d1 = Calibrate('./Calibration/24hours','./Calibration/24hours/ROI')
d1.calibrate()
d1.save('Calibrate/24h')
d7 = Calibrate('./Calibration/7days','./Calibration/7days/ROI')
d7.calibrate()
d7.save('Calibrate/7d')
