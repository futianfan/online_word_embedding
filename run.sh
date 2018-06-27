## the only input is ./data/text8

##python create_cooccurance.py 

### IMF
##python IMF.py

### online IMF
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



