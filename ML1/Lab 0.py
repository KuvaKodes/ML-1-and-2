import numpy as np
import matplotlib.pyplot as plt
# make your plot outputs appear and be stored within the notebook

x = np.linspace(0,20, 100)

plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red
plt.xlabel("X")
plt.ylabel("Sin(X)")
plt.title("A Sine Curve")
plt.plot(x, np.sin(x))
plt.show()
