dataset:
	nvcc -O3 -ccbin g++-11 -std=c++20 -expt-relaxed-constexpr -diag-suppress 177 collect_dataset.cu -o cuda_collect_dataset
	rm -f dataset_cuda.csv
	./cuda_collect_dataset
	head dataset_cuda.csv
