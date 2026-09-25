---
type: "explain"
date: "2026-09-25T15:27:14.285810+00:00"
question: "how does CNN works here"
contributor: "graphify"
source_nodes: ["BreedClassifier"]
---

# Q: how does CNN works here

## Answer

Expanded from original query via vocab: [convolutional, efficientnet, backbone, model, cbam, attention]. The CNN (Convolutional Neural Network) in this project operates as the primary feature extractor, leveraging the EfficientNet-Lite architecture. Here is how it functions step-by-step: 1) Image Input: The preprocessed 260x260 image is fed into the EfficientNet-Lite backbone. 2) MBConv Stages: The image passes through a series of Mobile Inverted Bottleneck Convolution (MBConv) blocks (grouped into Stages 0 to 6). These layers use depthwise separable convolutions, which extract hierarchical visual patterns (from simple edges to complex shapes like horns) while using significantly fewer parameters than standard convolutions. 3) Attention: A CBAM (Convolutional Block Attention Module) is inserted after Stage 3 to apply spatial and channel attention, helping the CNN focus on the animal rather than the background. 4) Feature Pooling: Finally, an AdaptiveAvgPool2d layer compresses the spatial output into a flat, 1280-dimensional feature vector, which is then routed to the parallel classification heads for breed prediction.

## Source Nodes

- BreedClassifier