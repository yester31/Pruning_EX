# Pruning_EX

### 0. Introduction
- Goal : model compression using by Pruning
- Base Model : Resnet18 
- Dataset : Imagenet100 
- Pruning Process : 
    1. Train base model with Imagenet100 dataset 
    2. Generate pruning model with channel pruning (max_ch_sparsity : 0.2) and fine tuning 
    2. Generate pruning model with channel pruning (max_ch_sparsity : 0.1) and fine tuning 
    4. Performance check with pytorch model
	5. Convert tensorRT model using by onnx
    6. Performance check with tensorRT model
---

### 1. Development Environment
- Device 
  - Windows 10 laptop
  - CPU i7-11375H
  - GPU RTX-3060
- Dependency 
  - cuda 12.1
  - cudnn 8.9.2
  - tensorrt 8.6.1
  - pytorch 2.1.0+cu121

---

### 2. Code Scheme
```
    TensorRT_ONNX/
    ├── .gitignore
    ├── 0_expert_onnx.py 					# onnx export example
    ├── 0_infer.py 							# test inference code 
    ├── 0_train.py							# train code
    ├── 1_prun.py							# ch pruning refrence code
    ├── 1_prun0.py							# custom model weight dirstribution
    ├── 2_tp_prun_ex.py						# torch pruning lib example code
    ├── 2_tp_prun_fine_tuning_comp.py		# performance check with dataset
    ├── 2_tp_prun_fine_tuning_temp.py		
    ├── 2_tp_prun_fine_tuning.py			# pruning and fine turning 
    ├── 3_simple_infer.py					# performance check with one batch model
    ├── 3_trt_infer.py						# performance check with tensorRT model
	├── common.py							# for trt
    ├── LICENSE
	├── prun_utils.py
    ├── README.md
    └── utils.py
```

---

### 3. Performance Evaluation
- Calculation 10000 iteration with one input data [1, 3, 224, 224]
- Only inference time
- One batch model

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Base Model</strong></td>
            <td><strong>Prun Model 1</strong></td>
            <td><strong>Prun Model 2</strong></td>
        </tr>
		<tr>
			<td>Accuracy</td>
            <td>82.26</td>
            <td>82.26</td>
            <td>81.84</td>
		</tr>
		<tr>
			<td>Params [Mb]</td>
            <td>11.23 Mb</td>
            <td>7.17 Mb</td>
            <td>5.79 Mb</td>
		</tr>
        <tr>
			<td>MACs</td>
			<td>1.82 G</td>
			<td>1.18 G</td>
			<td>0.95 G</td>
		</tr>
	</tbody>
</table>
        
<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>Pytorch (FP32) </td>
			<td><strong>Base Model</strong></td>
            <td><strong>Prun Model 1</strong></td>
            <td><strong>Prun Model 2</strong></td>
        </tr>
        <tr>
			<td>Avg Latency [ms]</td>
			<td>3.432 ms</td>
			<td>3.427 ms</td>
			<td>3.130 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>291.35 fps</td>
			<td>291.72 fps</td>
			<td>319.39 fps</td>
		</tr>
		<tr>
			<td>Gpu Memory [Mb]</td>
			<td>341 Mb</td>
			<td>295 Mb</td>
			<td>271 Mb</td>
		</tr>
	</tbody>
</table>

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>TensorRT (FP32)</td>
			<td><strong>Base Model</strong></td>
            <td><strong>Prun Model 1</strong></td>
            <td><strong>Prun Model 2</strong></td>
		</tr>
        <tr>
			<td>Avg Latency [ms]</td>
			<td>1.486 ms</td>
			<td>1.168 ms</td>
			<td>1.024 ms</td>
		</tr>
		<tr>
			<td>Avg FPS [frame/sec]</td>
			<td>672.59 fps</td>
			<td>855.62 fps</td>
			<td>975.79 fps</td>
		</tr>
		<tr>
			<td>Gpu Memory [Mb]</td>
			<td>165 Mb</td>
			<td>151 Mb</td>
			<td>143 Mb</td>
		</tr>
	</tbody>
</table>

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>TensorRT (FP16)</td>
			<td><strong>Base Model</strong></td>
            <td><strong>Prun Model 1</strong></td>
            <td><strong>Prun Model 2</strong></td>
		</tr>
        <tr>
			<td>Avg Latency [ms]</td>
			<td>0.542 ms</td>
			<td>0.518 ms</td>
			<td>0.501 ms</td>
		</tr>
		<tr>
			<td>Avg FPS [frame/sec]</td>
			<td>1845.07 fps</td>
			<td>1928.98 fps</td>
			<td>1997.79 fps</td>
		</tr>
		<tr>
			<td>Gpu Memory [Mb]</td>
			<td>135 Mb</td>
			<td>129 Mb</td>
			<td>125 Mb</td>
		</tr>
	</tbody>
</table>

## reference   
* Torch-Pruning : <https://github.com/VainF/Torch-Pruning>
* mit-han-lab(6s965-fall2022) : <https://github.com/mit-han-lab/6s965-fall2022>
* imagenet100 : <https://www.kaggle.com/datasets/ambityga/imagenet100>