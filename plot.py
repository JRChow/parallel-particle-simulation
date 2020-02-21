import matplotlib.pyplot as plt

# Linearity proof

n_pts = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
serial_time = [0.511648, 1.08961, 2.66103, 6.24908,
               13.6745, 28.4869, 59.3343, 130.781]
omp_time = [0.0610362, 0.087355, 0.139106,
            0.232138, 0.481802, 1.04082, 2.18011, 4.4162]
quadratic_time = [10.218, 40.5256, 161.284,
                  644.248, 2633.67, 11927.5]

fig = plt.figure(figsize=(12, 8))  # in inches!
plt.plot(n_pts, serial_time, 'x-', label='Linear serial')
plt.plot(n_pts, omp_time, 'x-', label='Linear OpenMP')
plt.plot(n_pts[:len(quadratic_time)], quadratic_time,
         'x-', label="Brute-force serial")
plt.xscale('log')
plt.yscale('log')
plt.title("Time Complexity of Particle Collision Simulations")
plt.xlabel("Number of particles")
plt.ylabel("Simulation time (sec)")
plt.legend(loc="upper right")
# plt.show()
plt.savefig('linear.png')

# Strong scaling
n_threads = [17, 34, 68, 136, 272]
strong_time = [191.804, 89.4106, 42.4856, 21.7177, 11.7018]

fig = plt.figure(figsize=(12, 8))
plt.plot(n_threads, strong_time, '-o')
plt.title("Strong Scaling of Our OpenMP Solution on 1M Particles")
plt.xlabel("Number of threads")
plt.ylabel("Simulation time (sec)")
# plt.show()
plt.savefig('strong-scale.png')

# Weak scaling
weak_time = [2.69585, 3.14073, 3.41831, 3.79983, 4.39289]

fig = plt.figure(figsize=(12, 8))
plt.plot(n_threads, weak_time, '-o')
plt.title("Weak Scaling of Our OpenMP Solution With 1,470 Particles/Thread")
plt.xlabel("Number of threads")
plt.ylim(0, 7)
plt.ylabel("Simulation time (sec)")
# plt.show()
plt.savefig('weak-scale.png')
