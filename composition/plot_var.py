from matplotlib import pyplot
import numpy as np

a = np.loadtxt("var.list")
x = np.arange(1,500)
cdf = np.cumsum(a)
print cdf
print len(x)
print len(cdf)
pyplot.plot(x, cdf[0:499])
pyplot.show()    

    
