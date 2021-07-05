from Calib import Calibrate, Fitting
d1 = Calibrate('./Calibration/24hours','./Calibration/24hours/ROI')
d1.calibrate()
d1.save('Calibration/24h')
d7 = Calibrate('./Calibration/7days','./Calibration/7days/ROI')
d7.calibrate('7d')
d7.save('Calibration/7d')
