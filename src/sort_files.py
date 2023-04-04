# import OS module
import os
import csv
import pandas as pd

# Get the list of all files and directories
path = "./mars-dataset/images"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

for i in dir_list:
	print(i)

df = pd.read_csv('./mars-dataset/image-labels.csv')

print(df.to_string())

# Move each image to a folder with its label
for i in range(len(df)):
	label = df.iloc[i, 1]
	filename = df.iloc[i, 0]
	if not os.path.exists(path + '/' + str(label)):
		os.makedirs(path + '/' + str(label))
	os.rename(path + '/' + filename, path + '/' + str(label) + '/' + filename)
	# print(filename + ' moved to ' + str(label))
