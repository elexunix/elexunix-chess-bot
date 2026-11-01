#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $0 <filename.csv>";
  exit 1;
fi;

input_fname=$1;
base_name="${input_fname%.csv}";
train_fname=${base_name}_train.csv;
test_fname=${base_name}_test.csv;

header=$(head -n1 $input_fname);
echo $header > $train_fname;
echo $header > $test_fname;
n=$(wc -l < $input_fname);
echo $n lines;
n_train=$((n*8/10));
sed -n 2,${n_train}p $input_fname | shuf >> $train_fname;
sed -n $((n_train+1)),$((n-1))p $input_fname | shuf >> $test_fname;
