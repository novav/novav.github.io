---
title: TensorRT int8
tags:
---

### Int8 calibration in Python[mnist_demo](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#int8_caffe_mnist)

### step:

- 1 create a INT8 calibrator

- build and calibrate an engine for INT8 mode

- run interence in INT8 mode



### Tourist

[CFG_tourist](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#unique_204330530)

#### Mode_setting

Enable INT8 mode by setting the builder flag:

builder.int8_mode = True



INT8 calibration can be used along with the dynamic range APIs. Setting the dynamic range manually will override the dynamic range generated from INT8 calibration.

与C ++ API类似，您可以选择每个激活张量使用动态范围 动态范围 或使用INT8校准。

INT8校准可与动态范围API一起使用。**手动设置动态范围将覆盖INT8校准生成的动态范围**。



#### Setting Per-Tensor Dynamic Range Using Python 

TensorRT需要网络中每个张量的动态范围。有两种方法可以为网络提供动态范围：

- 使用手动设置每个网络张量的动态范围 setDynamicRange API

- 使用INT8校准使用校准数据集生成每张量动态范围。

动态范围API也可以与INT8校准一起使用，这样手动设置范围将优先于校准生成的动态范围。如果INT8校准不能为某些张量产生令人满意的动态范围，则可能出现这种情况。



you must set the *dynamic range* for **each network tensor**

```
layer = network[layer_index]
tensor = layer.get_output(output_index)
tensor.dynamic_range = (min_float, max_float)
```



You also need to set the dynamic range for the **network input**:
```
input_tensor = network.get_input(input_index)
input_tensor.dynamic_range = (min_float, max_float)
```



#### INT8 Calibration Using Python

​	The following steps illustrate how to create an INT8 calibrator object using the Python API. By default, TensorRT supports INT8 calibration.



1. Import TensorRT:

   ```
   import tensorrt as trt
   ```

2. Similar to test/validation files, use set of input files as calibration files dataset. Make sure the calibration files are representative of the overall inference data files. For TensorRT to use the calibration files, we need to create batchstream object. Batchstream object will be used to configure the calibrator.

   ```
   NUM_IMAGES_PER_BATCH = 5
   batchstream = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
   ```

3. Create an Int8_calibrator object with input nodes names and batch stream:

   ```
   Int8_calibrator = EntropyCalibrator(["input_node_name"], batchstream)
   ```

4. Set INT8 mode and INT8 calibrator:

   ```
   trt_builder.int8_calibrator = Int8_calibrator
   ```

   余下的引擎的创建推理类似于[Importing From ONNX Using Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_onnx_python).

## Question：

1. 校准执行，多少批次，如何feed数据方式，生成的校准文件格式（内容）

   - 大约500个图像足以校准ImageNet分类网络。
   - 构建器调用校准器如下：首先，它调用getBatchSize（）来确定期望的输入批处理的大小然后，它反复呼叫 getBatch（）获得批量输入。批次应该与批次大小完全相同getBatchSize（）。当没有批次时，getBatch（）应该回来 False。
2. 校准器的选择

	- IEntropyCalibratorV2 :  这是首选校准器，是DLA所必需的，因为它支持每个激活张量缩放。

   - IEntropyCalibrator :  这是传统的熵校准器，支持每通道缩放。这比传统校准器简单并且产生更好的结果。

   - ILegacyCalibrator :  该校准器用于与2.0EA兼容。它已弃用，不应使用。
3. 构建器的执行流程
   - 构建INT8引擎时，构建器执行以下步骤：
   - 1-构建一个32位引擎，在校准集上运行它，并记录激活值分布的每个张量的直方图。
   - 2-根据直方图构建校准表。
   - 3-从校准表和网络定义构建INT8引擎。
4. 校准文件再加载inference流程，
   - 校准表可以缓存。在多次构建同一网络时（例如，在多个平台上），缓存非常有用。它捕获从网络和校准集中获得的数据。参数记录在表中。如果网络或校准集发生更改，则应用程序负责使缓存无效。
   - 缓存使用如下：
     - 如果找到校准表，则跳过校准，
     - 否则：校准表由直方图和参数构建
     - 然后INT8网络由网络定义和校准表构建。
5. 如何查看校准差异

### demo : [official](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html)

## [8. Performing Inference In INT8 Using Custom Calibration](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#int8_sample)

示例INT8执行INT8校准和推理。

此示例演示了如何以8位整数（INT8）执行推理。

INT8推断仅适用于计算能力为6.1或7.x的GPU。在校准网络以便在INT8中执行之后，缓存校准的输出以避免重复该过程。

/usr/src/tensorrt/samples/sampleINT8/

### 为非Caffe用户生成批处理文件

对于未使用Caffe或无法轻松转换为Caffe的开发人员，可以通过输入训练数据上的以下一系列步骤生成批处理文件。

- 从数据集中减去标准化均值。
- 将所有输入数据裁剪为相同的尺寸。
- 将数据拆分为每个批处理文件所在的批处理文件 ñ 预处理的图像和标签。
- 根据批处理文件中指定的格式生成批处理文件以进行校准。

以下示例描述了要运行的命令序列 ./sample_int8 mnist 没有Caffe。
- 导航到samples数据目录并创建INT8 MNIST 目录：

  ```
cd <TensorRT>/samples/data  
mkdir -p int8/mnist/batches  
cd int8/mnist  
ln -s <TensorRT>/samples/mnist/mnist.caffemodel .  
ln -s <TensorRT>/samples/mnist/mnist.prototxt .  
  ```
- 将生成的批处理文件复制到 INT8 / MNIST /批次/ 目录。
- 从中执行sampleINT8 箱子 使用以下命令构建后的目录： ./sample_int8 mnist



## [9. Performing Inference In INT8 Precision](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#int8_api_sample)

示例sampleINT8API在**不使用INT8校准器**的情况下执行INT8推理; **使用用户提供的每个激活张量动态范围**。INT8推断仅适用于计算能力为6.1或7.x的GPU，并支持图像分类ONNX模型，如ResNet-50，VGG19和MobileNet。

/usr/src/tensorrt/samples/sampleINT8API/




## [24. INT8 Calibration In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#int8_caffe_mnist)



/usr/src/tensorrt/samples/python/int8_caffe_mnist



During calibration: total 1003 barches, 100 each

​	calibrator.py: 简化了read write 校准的过程

1、RUN sample 

```	python
python3 sample.py [-d DATA_DIR]
```

2、 Verify ran successfully 

1. ```
   Expected Predictions:
   [1. 6. 5. 0. 2. 8. 1. 5. 6. 2. 3. 0. 2. 2. 6. 4. 3. 5. 5. 1. 7. 2. 1. 6.
   9. 1. 9. 9. 5. 5. 1. 6. 2. 2. 8. 6. 7. 1. 4. 6. 0. 4. 0. 3. 3. 2. 2. 3.
   6. 8. 9. 8. 5. 3. 8. 5. 4. 5. 2. 0. 5. 6. 3. 2. 8. 3. 9. 9. 5. 7. 9. 4.
   6. 7. 1. 3. 7. 3. 6. 6. 0. 9. 0. 1. 9. 9. 2. 8. 8. 0. 1. 6. 9. 7. 5. 3.
   4. 7. 4. 9.]
   Actual Predictions:
   [1 6 5 0 2 8 1 5 6 2 3 0 2 2 6 4 3 5 5 1 7 2 1 6 9 1 9 9 5 5 1 6 2 2 8 6 7
   1 4 6 0 4 0 3 3 2 2 3 6 8 9 8 5 3 8 5 4 5 2 0 5 6 3 2 8 3 9 9 5 7 9 4 6 7
   1 3 7 3 6 6 0 9 0 1 9 4 2 8 8 0 1 6 9 7 5 3 4 7 4 9]
   Accuracy: 99.0%
   ```

