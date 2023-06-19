from __future__ import print_function
import sys

if __name__ == '__main__':

	infile = sys.argv[1]
	outfile = sys.argv[2]
	print("The input file is: %s" %(infile) )
	print("The output file is: %s" % (outfile))

	with open('example.txt') as f1:
		content = f1.readlines()

	with open('output.txt','w') as f2:
		for i in range( len(content)-1, -1, -1 ):
			f2.write(content[i])