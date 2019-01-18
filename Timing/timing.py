import numpy as np
import matplotlib.pyplot as plt

timings = {}
with open("timing.txt","r") as file:
    next(file)
    for line in file:
        info = line.split(" ")
        timings[int(info[0])] = float(info[1])

plt.plot(timings.keys(),timings.values())
print(timings.keys())
plt.xticks(list(timings.keys()),list(timings.keys()))
plt.title('Scaling Analysis for Pixel Binning Algorithm')
plt.xlabel("Number of Processors")
plt.ylabel("Time (s)")
plt.savefig("timing_analysis.png")