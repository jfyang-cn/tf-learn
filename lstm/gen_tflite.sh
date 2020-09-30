#!/bin/sh

tflite_convert --output_file=model.tflite \
	--graph_def_file=lstm_model.pb \
	--inference_type=QUANTIZED_UINT8  \
	--inference_input_type=QUANTIZED_UINT8 \
	--input_arrays=input_ \
	--output_arrays=s_out/Softmax \
	--mean_values=0 \
	--std_dev_values=1
