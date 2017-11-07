import numpy as np
import matplotlib.pyplot as plt
import re
import fnmatch
import os

logdir = "logs/BreakoutNoFrameskip-v4/"
def read_file(filename, pattern="eval_avg_score", eval_interval=1.0):
	table = []
	with open(logdir+filename) as f:
		text = f.read()
		text = re.split('\n', text)
		for row in text:
			if pattern in row:
				table.append(re.split(' +', row))

	data = []
	for row in table:
		if row[-2] != "":
			data.append(np.float32(row[-2]))
	nRows = len(data)

	if pattern == "eval_avg_score":
		time = np.asarray([i*eval_interval for i in range(nRows)])
	else:
		time = np.asarray([i*eval_interval/125 for i in range(nRows)])

	return time, data

# log_files = sorted(fnmatch.filter(os.listdir('.'), "*log*.txt"))
# log_files = ["2017-11-02-17-31-06-331713-log.txt",
# 			 "2017-11-02-19-55-40-032464-log.txt",
# 			 "2017-11-05-11-57-30-812910-log-WIP.txt",
# 			 "2017-11-05-12-05-20-545401-log-WIP.txt"]
log_files = ["2017-11-06-14-36-11-435842/log.txt",
			 "2017-11-06-14-36-40-634245/log.txt",
			 "2017-11-06-14-37-10-051730/log.txt",
			 "2017-11-06-14-37-53-921575/log.txt",
			 "2017-11-06-14-38-38-830422/log.txt",
			 "2017-11-06-14-39-56-109458/log.txt"]
# labels = ['Step 1. Switch to Q function',
# 		  'Step 2. Switch ADV from (Yn-V) to (Q-V)',
# 		  'Step 2. (Q-V), parallel by 32',
# 		  'Step 2. (Q-V), parallel by 32']
labels = ['nsteps=1',
		  'nsteps=2',
		  'nsteps=3',
		  'nsteps=4',
		  'nsteps=5',
		  'nsteps=8']
eval_interval = [.2, .4, .6, .8, 1.0, 1.6]

def visualize(log_files, labels, metric):
	for filename, label, interval in zip(log_files, labels, eval_interval):
		time, data = read_file(filename, metric, interval)
		if "explained" in metric:
			data = np.maximum(data, -2)
		plt.plot(time, data, label=label)
		if "explained" in metric:
			plt.ylim([-2,1])
	plt.legend()
	plt.ylabel(metric)

plt.subplot(221)
visualize(log_files, labels, "eval_avg_score")
plt.subplot(222)
visualize(log_files, labels, "explained_variance")
plt.subplot(223)
visualize(log_files, labels, "value_loss")
plt.subplot(224)
visualize(log_files, labels, "entropy")

plt.show()
plt.close()