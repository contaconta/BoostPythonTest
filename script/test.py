import time
import numpy as np
import libnumpytest
A = np.arange(50000*5000).reshape(50000, 5000).astype(np.float64)
#A = np.arange(8).reshape(2, 4).astype(np.float64)
libnumpytest.debug_print(A)

for i in range(5):
    print "-----add: {0}-------".format(i)
    print "------ A ---------"
    print A.shape

    time1 = time.clock()

    B = libnumpytest.add2d(A, A)

    time2 = time.clock()
    print 'add2d: {0}[s]'.format(float(time2 - time1))
    #print B

    time1 = time.clock()

    B = A+A

    time2 = time.clock()
    print 'python: {0}[s]'.format(float(time2 - time1))

for i in range(5):
    print "-----mul {0}-------".format(i)
    print "------ A ---------"
    print A.shape

    time1 = time.clock()

    B = libnumpytest.mul2d(A, A)

    time2 = time.clock()
    print 'add2d: {0}[s]'.format(float(time2 - time1))
    #print B

    time1 = time.clock()

    B = np.multiply(A, A)

    time2 = time.clock()
    print 'python: {0}[s]'.format(float(time2 - time1))

