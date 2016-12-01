import matplotlib.pyplot as plt
import numpy as np

ev_file = './DB/ev.score'
data_file = './DB/fv.list'

data = np.loadtxt(data_file)
ev = np.loadtxt(ev_file)

time_1 = data[:, 0:1]
time_2 = data[:, 1:2]
time_3 = data[:, 2:3]

plt.figure(1)

plt.subplot(221)
plt.plot(time_1.ravel(), ev.ravel(), 'bo')
plt.axis([-12, 12, -1, 17])

plt.subplot(222)
plt.plot(time_2.ravel(), ev.ravel(), 'ro')
plt.axis([-12, 12, -1, 17])

plt.subplot(223)
plt.plot(time_3.ravel(), ev.ravel(), 'ro')
plt.axis([-12, 12, -1, 17])

plt.subplot(224)
plt.plot(time_3.ravel(), ev.ravel(), 'ro')
plt.axis([-12, 12, -1, 17])

plt.show()
