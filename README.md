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
    ├── 0_expert_onnx.py                # onnx export example
    ├── 0_infer.py                      # test inference code
    ├── 0_train.py                      # train code
    ├── 1_prun.py                       # ch pruning refrence code
    ├── 1_prun0.py                      # custom model weight dirstribution
    ├── 2_tp_prun_ex.py                 # torch pruning lib example code
    ├── 2_tp_prun_fine_tuning_comp.py   # performance check with dataset
    ├── 2_tp_prun_fine_tuning_temp.py
    ├── 2_tp_prun_fine_tuning.py        # pruning and fine turning
    ├── 3_simple_infer.py               # performance check with one batch model
    ├── 3_trt_infer.py                  # performance check with tensorRT model
    ├── common.py                       # for trt
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
      <td>2.308 ms</td>
      <td>2.472 ms</td>
      <td>2.328 ms</td>
    </tr>
    <tr>
      <td>Avg FPS [frame/sec]</td>
      <td>433.26 fps</td>
      <td>404.51 fps</td>
      <td>429.54 fps</td>
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
      <td>1.203 ms</td>
      <td>1.029 ms</td>
      <td>0.912 ms</td>
    </tr>
    <tr>
      <td>Avg FPS [frame/sec]</td>
      <td>830.60 fps</td>
      <td>971.66 fps</td>
      <td>1096.38 fps</td>
    </tr>
    <tr>
      <td>Gpu Memory [Mb]</td>
      <td>179 Mb</td>
      <td>161 Mb</td>
      <td>147 Mb</td>
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
      <td>0.519 ms</td>
      <td>0.500 ms</td>
      <td>0.486 ms</td>
    </tr>
    <tr>
      <td>Avg FPS [frame/sec]</td>
      <td>1924.28 fps</td>
      <td>1996.28 fps</td>
      <td>2054.25 fps</td>
    </tr>
    <tr>
      <td>Gpu Memory [Mb]</td>
      <td>135 Mb</td>
      <td>129 Mb</td>
      <td>123 Mb</td>
    </tr>
  </tbody>
</table>

## reference
* Torch-Pruning : <https://github.com/VainF/Torch-Pruning>
* mit-han-lab(6s965-fall2022) : <https://github.com/mit-han-lab/6s965-fall2022>
* imagenet100 : <https://www.kaggle.com/datasets/ambityga/imagenet100>