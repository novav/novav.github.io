---
title: Paper-CV-PoseEstimation
date: 2019-11-22 16:04:11
tags:
---

--

Pose Estimation

|           | 论文                                                         |      |
| --------- | ------------------------------------------------------------ | ---- |
| OpenPose  | Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields |      |
| DensePose |                                                              |      |
| AlphaPose |                                                              |      |
| DeepPose  |                                                              |      |



## OpenPose 

### Keypoint-json

output_overview:

```bash
CUDA_VISIBLE_DEVICES='1' ./build2/examples/openpose/openpose.bin  --image_dir ./image --write_images ./output_image --write_json ./output_json     --display 0

```
v1.1-25

```
// Result for BODY_25 (25 body parts consisting of COCO + foot)
// const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
//     {25, "Background"}
```



v1.0-18

```
// C++ API call
 #include <openpose/pose/poseParameters.hpp>
 const auto& poseBodyPartMappingCoco = getPoseBodyPartMapping(PoseModel::COCO_18);
 const auto& poseBodyPartMappingMpi = getPoseBodyPartMapping(PoseModel::MPI_15);
 
 // Result for COCO (18 body parts)
 // POSE_COCO_BODY_PARTS {
 //     {0,  "Nose"},
 //     {1,  "Neck"},
 //     {2,  "RShoulder"},
 //     {3,  "RElbow"},
 //     {4,  "RWrist"},
 //     {5,  "LShoulder"},
 //     {6,  "LElbow"},
 //     {7,  "LWrist"},
 //     {8,  "RHip"},
 //     {9,  "RKnee"},
 //     {10, "RAnkle"},
 //     {11, "LHip"},
 //     {12, "LKnee"},
 //     {13, "LAnkle"},
 //     {14, "REye"},
 //     {15, "LEye"},
 //     {16, "REar"},
 //     {17, "LEar"},
 //     {18, "Background"},
 // }
```

![1574412724600](Paper-CV-PoseEstimation/1574412724600.png)

### heatmap order

对于**热图存储格式**，而不是分别保存67个热图(18个主体部分+背景+ 2 x 19个PAFs)，库将它们连接成一个巨大的(宽度x #热图)x(高度)矩阵(即。，由列连接)。例如：列[0，单个热图宽度]包含第一个热图，列[单个热图宽度+ 1, 2 *单个热图宽度]包含第二个热图，等等。注意，一些图像查看器由于尺寸的原因无法显示结果图像。然而，Chrome和Firefox能够正确地打开它们。

PAFs遵循“getposepartpair (const PoseModel PoseModel)”和“getPoseMapIndex(const PoseModel PoseModel)”中指定的顺序。例如，假设COCO(参见下面的示例代码)，COCO中的PAF通道从19开始(‘getPoseMapIndex’中最小的数字，等于#body parts + 1)，到56结束(最高的一个)。