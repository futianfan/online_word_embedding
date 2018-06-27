## the only input is ./data/text8

## preprocess
python create_cooccurance.py 

### IMF
python IMF.py


### online IMF: split the vocabulary into 10 parts. 
### compute the performance on word_analogy task for 10 times. 
num_word=`wc -l data/dictionary | awk '{print $1}'`
split_num=10
for ((i=0; i<$split_num; i++)); 
do 
	python2 onlineIMF.py $split_num $num_word $i 
	awk '{print $1}' result/W_matrix > result/W_dict
	cd word_analogy_evaluate
	str="result_"
	python eval/python/evaluate.py \
		--vocab_file ../result/W_dict \
		--vectors_file ../result/W_matrix > ${str}$i
	cd ..
done



