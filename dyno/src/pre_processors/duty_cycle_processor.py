from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

f = np.genfromtxt('non-filtered-duty-cycle.csv', delimiter = ',')
print(f)

torque = savgol_filter(f[:,2], 29, 2)

plt.figure()
plt.plot(f[:,0], f[:,1])

plt.figure()
plt.plot(f[:,0], f[:,2])

plt.figure()
plt.plot(f[:,0], torque)

# plt.show()

f[:,2] = -1*torque
f[:,1] *= -31
f[1:,:] = np.nan_to_num(f[1:,:])
f[-1,2] = 0

np.savetxt('duty_cycle.csv', f, delimiter=",")
