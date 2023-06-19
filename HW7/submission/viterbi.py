import sys 
import numpy as np 

def load_index(filename):
	index = 0
	word_to_index, index_to_word = {}, {}

	with open(filename) as f:
		for line in f.readlines():
			index += 1 
			word = line.strip('\n')
			word_to_index[word] = index 
			index_to_word[index] = word 
	return word_to_index, index_to_word

def load_hmm_prob(filename):
	hmm_prob = []
	with open(filename) as f:
		lines = f.readlines()

	for line in lines:
		line = line.strip(' \n')
		nums = line.split(' ')
		hmm_prob.append([ float(num) for num in nums] )
	return np.array(hmm_prob)

def load_hmm_prior(filename,num_states):
	prior = []
	with open(filename) as f:
		for _ in range(num_states):
			line = f.readline()
			num = line.strip('\n')
			prior.append( float(num) )
	prior = np.array(prior).reshape((num_states, -1) )
	return prior

def word_tag_separation( line ):
	word_to_tag = line.split(' ')
	words,tags = [], []
	for combination in word_to_tag:
		word, tag = combination.split('_')
		words.append(word)
		tags.append(tag)
	return words,tags


def forward(words, word_to_index, prior, A, B, num_states):
	nt = len(words)
	alpha = np.zeros( (num_states, nt) ) # alpha table for forward propagation
	pt = np.zeros( (num_states, nt) )

	# Intialize the first column of alpha
	word = words[0]
	for i in range(num_states):
		alpha[i][0] = np.log( prior[i] ) + np.log( B[i][ word_to_index[word] - 1] )
		pt[i][0] = i + 1 

	# moving forward
	for t in range(1,nt):
		word = words[t]

		for j in range(num_states):
			lw = []
			for k in range(num_states):
				log_likelihood = np.log( B[j][word_to_index[word] - 1 ]) + np.log( A[k][j] ) + alpha[k][t-1]
				lw.append( log_likelihood)
			alpha[j][t] = max(lw)
			pt[j][t] = lw.index( max(lw) ) + 1 

	return alpha, pt 

def predict(alpha, pt, index_to_tag):

	predicted_tag = []
	index = np.argmax( alpha[:,-1] )
	predicted_tag.append( index_to_tag[index + 1] )

	for t in range( len(alpha[0]) - 1,  0, -1 ):
		index = pt[ int(index) , t ]
		predicted_tag.append( index_to_tag[index] )

	return list( reversed( predicted_tag )  )

if __name__ == "__main__":

	test_input = sys.argv[1]
	index_to_word_file = sys.argv[2]
	index_to_tag_file = sys.argv[3]

	hmmprior = sys.argv[4]
	hmmemit = sys.argv[5]
	hmmtrans = sys.argv[6]

	predicted_file = sys.argv[7]
	metric_file = sys.argv[8]


	content = []
	with open( test_input ) as f:
		for line in f.readlines():
			content.append( line.strip('\n'))

	word_to_index, index_to_word = load_index( index_to_word_file )
	tag_to_index, index_to_tag = load_index( index_to_tag_file )

	num_states, num_words = len(index_to_tag), len(index_to_word)

	prior = load_hmm_prior(hmmprior,num_states)
	A = load_hmm_prob(hmmtrans)
	B = load_hmm_prob(hmmemit)


	predicted_res = []
	count, error = 0, 0
	for line in content:
		words, tags = word_tag_separation(line)
		alpha, pt = forward(words, word_to_index, prior, A, B, num_states )
		predicted_tag = predict(alpha, pt, index_to_tag)

		count += len(tags)

		for i in range( len(tags) ):
			if tags[i] != predicted_tag[i]:
				error += 1 

		new_line = []
		for i in range(len(words) ):
			new_line.append( "_".join([words[i], predicted_tag[i] ] ) )
		predicted_res.append( " ".join(new_line))

	with open(predicted_file,'w') as f:
		for line in predicted_res:
			f.write( line + '\n')

	accuracy = 1 - error/count 
	with open(metric_file, 'w') as f:
		f.write("Accuracy: %.6f"% accuracy )
















