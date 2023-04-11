import matplotlib.pyplot as plt
import numpy as np

def plot_ellipse(center, a, b, angle):
    # Generate x and y values for the ellipse
    t = np.linspace(0, 2*np.pi, 100)
    x = center[0] + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)
    y = center[1] + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)

    # Plot the ellipse
    plt.plot(x, y)

# Example usage
center = (0, 0)
a = 3
b = 1
angle = np.pi/4
plot_ellipse(center, a, b, angle)
plt.axis('equal')
plt.show()