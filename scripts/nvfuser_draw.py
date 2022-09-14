from matplotlib import pyplot as plt
import numpy as np

labels = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

time_before = np.array([0.01585595131,
0.02701902866,
0.05243816376,
0.08773333549,
0.08773333549])

time_after = np.array([0.0139317131,
0.0240680027,
0.04217479706,
0.07210779667,
0.07210779667])

pred = np.array([0.001419264,
0.002136064,
0.006350848,
0.009275392,
0.009275392])

time_before_minus_pred = time_before - pred
width = 0.35
x = np.arange(len(labels))

plt.bar(x - width/2, time_before_minus_pred, width=width, edgecolor="black")
rects1 = plt.bar(x - width/2, pred, bottom=time_before_minus_pred, width=width, label="predicted acceleration", color="cyan", edgecolor="black")
rects2 = plt.bar(x + width/2, time_after, width=width, label="running time after fusion", color="orange", edgecolor="black")
plt.xticks(x, labels)
plt.ylabel("Time (s)")
plt.legend()

# plt.bar_label(rects1, padding=3)
# plt.bar_label(rects2, padding=3)
plt.savefig('nvfuser_resnets.png')