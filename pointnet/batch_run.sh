for var in {0..9}
do
	python generate_grasp_classification_data.py $var 10 &
done
