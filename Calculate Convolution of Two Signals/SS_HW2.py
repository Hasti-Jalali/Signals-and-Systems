import numpy as np
from matplotlib import pyplot as plt

# def conv_calculator(h,x) :
#     return np.convolve(h,x)

def conv_calculator(h,x,t):
    output = []
    start =int (t.min())
    end = int(t.max())
    step = (end - start)/t.size
    for t0 in range(int(-t.size/2),int(t.size/2)):
        sum = 0
        for k in range(-300,300):
            sum += h(t0*step - k*step) * x(k*step)
        output.append(sum)
    return output

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

start1 = -25
end1 = 25
n1 = np.linspace(start1,end1,int(end1 - start1))
h1 = lambda n1: np.heaviside(n1 - 10, 1) - np.heaviside(n1 + 10, 1)
x1 = lambda n1: np.power(1/4,n1) * (np.heaviside(n1 - 5, 1) - np.heaviside(n1 + 5, 1))
y1 = conv_calculator(h1,x1,n1)
ax1.stem(n1,y1)

start2 = -15
end2 = 15
step2 = 0.1
t2 = np.linspace(start2,end2,int((end2 - start2)/step2))
h2 = lambda t2:  np.heaviside(t2 + 3, 1) - np.heaviside(t2 - 3, 1)
x2 = lambda t2: 1/2 * np.exp(2 * t2) * np.heaviside(-t2, 1)
y2 = conv_calculator(h2,x2,t2)
ax2.plot(t2,y2)

plt.show()