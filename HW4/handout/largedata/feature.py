import sys
import csv


def bag_to_word(content, word_dict, feature_flag,file_out):

	f = open(file_out,'w')
	for line in content:
		label = line[0]
		comment = line[1].split()

		word_in_comment_cnt = {}
		for word in comment:
			if word in word_dict:
				if word not in word_in_comment_cnt:
					word_in_comment_cnt[word] = 1
				else:
					word_in_comment_cnt[word] += 1

		feature_out = []

		for key, val in word_in_comment_cnt.items():
			if feature_flag == 1:
				index_to_value = str( word_dict[key] ) + ":" + str(1)
				feature_out.append(index_to_value)
			if feature_flag == 2:
				if val < 4:
					index_to_value = str( word_dict[key] ) + ":" + str(1)
					feature_out.append(index_to_value)

		f.write( str(label) + '\t' )
		f.write( '\t'.join(feature_out) + '\n')
	f.close()





if __name__ == '__main__':

	train_input = sys.argv[1]
	validation_input = sys.argv[2]
	test_input = sys.argv[3]
	dict_input = sys.argv[4]
	formatted_train_out = sys.argv[5]
	formatted_validation_out = sys.argv[6]
	formatted_test_out = sys.argv[7]
	feature_flag = int( sys.argv[8] )



	word_dict = dict()
	with open(dict_input,'r') as f:
		for line in f.readlines():
			[word, num] = line.split()
			word_dict[word] = int(num)
	

	with open(train_input,'r') as f:
		reader = csv.reader(f,delimiter = '\t')
		content_train = list(reader)

	with open(validation_input,'r') as f:
		reader = csv.reader(f, delimiter = '\t')
		content_validation = list(reader)

	with open(test_input,'r') as f:
		reader = csv.reader(f,delimiter = '\t')
		content_test = list(reader)

    # training data
	bag_to_word(content_train, word_dict, feature_flag,formatted_train_out)

	# validation data
	bag_to_word(content_validation, word_dict, feature_flag,formatted_validation_out)

	# test data
	bag_to_word(content_test, word_dict, feature_flag,formatted_test_out)




