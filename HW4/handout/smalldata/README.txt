python feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1


python lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60