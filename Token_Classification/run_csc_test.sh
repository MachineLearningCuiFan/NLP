

accelerate launch run_csc_test.py \
  --model_name_or_path "test_csc" \
  --test_file data/test13.csv \
  --label_column_name labels \
  --pad_to_max_length \
  --per_device_test_batch_size 200 \

