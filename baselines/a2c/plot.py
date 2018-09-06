import numpy as np
import matplotlib.pyplot as plt
import re
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

log_files = ["2017-11-06-14-36-11-435842/log.txt",
			 "2017-11-06-14-36-40-634245/log.txt"]
labels = ['AAC',
		  'MAC']

def visualize(log_files, labels, metric):
	for filename, label in zip(log_files, labels):
		time, data = read_file(filename, metric)
		if "explained" in metric:
			data = np.maximum(data, -2)
		plt.plot(time, data, label=label)
		if "explained" in metric:
			plt.ylim([-2,1])
	plt.legend()
	plt.ylabel(metric)

visualize(log_files, labels, "eval_avg_score")
plt.show()
plt.close()
