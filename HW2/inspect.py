import csv
import math
import sys
import numpy as np

def entropy( num1, num2):
	prob1 = num1/( num1 + num2)
	prob2 = num2/( num1 + num2)

	return -prob1*math.log2(prob1) - prob2*math.log2(prob2)

def helper_error( num1, num2):
	if num1 >= num2:
		return num2/(num1 + num2)
	else:
		return num1/(num1 + num2)


if __name__ == '__main__':

	infile  = sys.argv[1]
	outfile = sys.argv[2]

	num1, num2 = 0, 0

	with open(infile,'r') as f1:
		reader = csv.reader(f1, delimiter = '\t')
		content = list(reader)
		attr_name = content[0]
		data = np.array( content[1:])

	label = list( set(data[:,-1] ) )

	for val in data[:,-1]:
		if val == label[0]:
			num1 += 1
		else:
			num2 += 1


	val_entropy = entropy(num1, num2)
	val_error   = helper_error(num1, num2)

	with open(outfile,'w') as f2:
		f2.write("entropy: " + str(val_entropy) +'\n' )
		f2.write("error: " + str(val_error)     )






