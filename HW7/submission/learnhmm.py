import sys 
import numpy as np 

if __name__ == '__main__':
    

	train_input = sys.argv[1]
	path_index_to_word = sys.argv[2]
	path_index_to_tag = sys.argv[3]


	hmmprior = sys.argv[4]
	hmmemit = sys.argv[5]
	hmmtrans = sys.argv[6]

	content = []
	with open(train_input) as f:
		for line in f.readlines():
			content.append(line.strip('\n'))

	index = 0
	index_to_tag = {}
	with open(path_index_to_tag) as f:
		for line in f.readlines():
			index += 1 
			tag = line.strip('\n')
			index_to_tag[tag] = index	

	index = 0
	index_to_word = {}
	with open(path_index_to_word) as f:
		for line in f.readlines():
			index += 1
			word = line.strip('\n')
			index_to_word[word] = index

	num_states = len(index_to_tag)
	num_words = len(index_to_word)
	prior = np.ones( (num_states, 1) )
	prob_emit = np.ones( (num_states, num_words ) )
	prob_trans = np.ones( (num_states, num_states ) )

	for line in content:
		words, tags = [], []
		for word_to_tag in line.split(' '):
			word, tag = word_to_tag.split('_')
			words.append(word)
			tags.append(tag)   
		prior[ index_to_tag[ tags[0] ] - 1 ] += 1 

		for i in range(0, len(tags)- 1 ):
			state1 = index_to_tag[ tags[i] ]
			state2 = index_to_tag[ tags[i+1] ]
			prob_trans[state1 - 1][state2 - 1] += 1 

		for i in range( len(words) ):
			state = index_to_tag[ tags[i] ]
			word = index_to_word[ words[i] ]
			prob_emit[state - 1][word-1] += 1 	

	prior /= np.sum(prior)		

	count = np.sum(prob_trans, axis = 1)
	count = count.reshape( num_states, -1 )
	prob_trans /= count
	
	count = np.sum(prob_emit, axis = 1)
	count = count.reshape( num_states, -1 )
	prob_emit /= count

	with open(hmmprior, 'w') as f:
		for val in prior:
			f.write( str(val[0]) + '\n' )


	with open(hmmtrans, 'w') as f:
		for line in prob_trans:
			for num in line:
				f.write( str(num) + ' ')
			f.write('\n')

	with open(hmmemit, 'w') as f:
		for line in prob_emit:
			for num in line:
				f.write( str(num) + ' ')
			f.write('\n')

