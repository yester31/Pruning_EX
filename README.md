# Pruning_EX

### 0. Introduction
- Goal : model compression using by Pruning
- Base Model : Resnet18 
- Dataset : Imagenet100 
- Pruning Process : 
    1. Train base model(python) 
    2. local pruning (max_ch_sparsity : 0.2) -> fine tuning 
    3. local pruning (max_ch_sparsity : 0.1) -> fine tuning 
    4. pytorch model -> onnx -> tensorRT model
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
    ├── 0_expert_onnx.py
    ├── 0_infer.py
    ├── 0_train.py
    ├── 0_utils.py
    ├── 1_prun_utils.py
    ├── 1_prun.py
    ├── 1_prun0.py
    ├── 2_tp_prun_ex.py
    ├── 2_tp_prun_fine_tuning_comp.py
    ├── 2_tp_prun_fine_tuning_temp.py
    ├── 2_tp_prun_fine_tuning.py
    ├── 3_common.py
    ├── 3_simple_infer.py
    ├── 3_trt_infer.py
    ├── LICENSE
    └── README.md
```

---

### 3. Performance Evaluation
- Comparison of calculation average execution time of 10000 iteration and FPS for one image [224,224,3]

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>Pytorch (FP32)</td>
			<td><strong>Base Model</strong></td>
            <td><strong>Prun Model 1</strong></td>
            <td><strong>Prun Model 2</strong></td>
        </tr>
		<tr>
			<td>acc1</td>
            <td>82.26</td>
            <td>82.26</td>
            <td>81.84</td>
		</tr>
		<tr>
			<td>Params</td>
            <td>11.23 M</td>
            <td>7.17 M</td>
            <td>5.79 M</td>
		</tr>
        <tr>
			<td>MACs</td>
			<td>1.82 G</td>
			<td>1.18 G</td>
			<td>0.95 G</td>
		</tr>
        <tr>
			<td>Avg latency [ms]</td>
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
			<td>Avg latency [ms]</td>
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
			<td>Avg latency [ms]</td>
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
	</tbody>
</table>

## reference   
* Torch-Pruning : <https://github.com/VainF/Torch-Pruning>
* mit-han-lab(6s965-fall2022) : <https://github.com/mit-han-lab/6s965-fall2022>
* imagenet100 : <https://www.kaggle.com/datasets/ambityga/imagenet100>