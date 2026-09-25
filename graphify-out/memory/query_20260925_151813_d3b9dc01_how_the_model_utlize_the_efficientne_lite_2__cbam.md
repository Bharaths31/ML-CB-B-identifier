---
type: "explain"
date: "2026-09-25T15:18:13.802784+00:00"
question: "how the model utlize the efficientne lite 2, cbam ,ood"
contributor: "graphify"
source_nodes: ["BreedClassifier"]
---

# Q: how the model utlize the efficientne lite 2, cbam ,ood

## Answer

Expanded from original query via vocab: [efficientnet, lite, cbam, ood, energy, model]. The model utilizes these three components in a sequential pipeline to ensure high accuracy and reject invalid images: 1) EfficientNet-Lite: Serves as the lightweight feature extraction backbone. It processes the input image through its stages to generate a 1280-dimensional feature vector. 2) CBAM (Convolutional Block Attention Module): Inserted directly after Stage 3 of the EfficientNet backbone. It applies channel and spatial attention to force the network to focus on critical morphological animal traits (like horns or humps) rather than background noise. 3) OOD (Out-Of-Distribution) Guard: An energy-based filter that evaluates the logits from the classification heads. If the calculated energy score is too high (e.g. > -25.0), the model recognizes the image as 'non-bovine' (like a car or dog) and immediately rejects it with 0% confidence, preventing blind guesses.

## Source Nodes

- BreedClassifier