import matplotlib.pyplot as plt


n_pts = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
serial_time = [0.511648, 1.08961, 2.66103, 6.24908,
               13.6745, 28.4869, 59.3343, 130.781]
omp_time = [0.0610362, 0.087355, 0.139106,
            0.232138, 0.481802, 1.04082, 2.18011, 4.4162]

plt.plot(n_pts, serial_time, 'x-')
plt.plot(n_pts, omp_time, 'x-')
plt.xscale('log')
plt.yscale('log')
plt.title("Time Complexity of Particle Collision Simulations")
plt.xlabel("Number of particles")
plt.ylabel("Simulation time")
plt.show()
