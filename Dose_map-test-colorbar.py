from EBT3 import *
d = Calibrate('202105_UoB_Microbeam/Films/','Calibration/ROI')
d.load('Calibration/Calibration_UoB')
f = Fitting(d.Data)
f.fit()
OD = d.OD(6, roi=0)
Dose = f.coeff[0]*(OD**2)+f.coeff[1]*(OD)+f.coeff[2]
fig = plt.subplots()
plt.imshow(Dose)
plt.colorbar()
plt.show()
