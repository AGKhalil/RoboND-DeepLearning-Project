# Project Writeup
## Architecture
The network architecture used for this project is pretty simple. It is a Fully Convolutional Network (FCN) that consists of five layers: two encoders, one 1x1 convolution layer, and 2 decoders. Figure 1 contains a schematic of each layer and its dimensions respectively. Figure 2 contains the model summary.

![Network_Architecture](/writeup_images/Network_Architecture.png)

Basically what the network is doing here is first looking at finer details of the input image. Then with each layer, the network starts to combine its understanding of those features into more complex structures. With each layer, the network gets deeper, which means there are more filters working on understanding the many features in the image. The beauty of a FCN is that is takes a look at the fine details of the input without losing sight of their spatial locations. This is essential for this project as the drone is following someone.

While designing the network, some parameters were chosen based on conceptual reasoning, some were experimental, and some were chosen just because. In this section I will explain the network architecture and the parameters chosen.

![Model_Summary](/writeup_images/Model_Summary.png)

### Encoder Block
Each encoder layer is essentially a separable convolution layer that is batch normalized. Separable convolutions are used to reduce the total number of parameters in the network, thereby decreasing its run time. They output the same result as a regular convolutional layer, but with better performance. Separable convolutions do this by decoupling the spatial and channel dimensions in creating the output. It first applies a `1x1xn` convolution on the input resulting in `n` feature maps with the same height and width of the input. Afterwards, a regular kernel is applied to each feature map to look at the spatial aspects. The final result is identical to a regular convolution layer, but with less parameters, thus better performance.

After the output layer is produced, it is passed through a ReLU activation function, adding a non-linearity. 

The kernel size used here is 3x3 with a stride of 2. The chosen kernel size seemed a reasonable initial choice that is not too computationally intensive and it was not changed. The stride was chosen as 2 to reduce computation as opposed to a stride of 1; however, a stride of 3 would have been two high, resulting in an output layer a third of the size of its input. A 3 might have been chosen if the input layer's dimensions were perceived as too large.

The encoders used here have a few advantages. They are efficient and use less computation power, in comparison to a regular convolution network, due to their dependance on separable convolutional layers. They have many non-linearities added to them due to the ReLU activation functions, which allows the network to learn better as it can map non-linear relationships. 

### 1x1 Convolution Layer
The purpose of this layer is to increase the previous layer's depth, while retaining the same spatial dimensions. This layer is fundamentally a regular convolution with a kernel size of 1x1 and a stride of 1. Furthermore, the layer is passed through a ReLU function adding more non-linearity. 

This layer is really there to add more information, introduce more non-linearity into the network, and retain the network's spatial dimensions.

### Decoder Block
The decoder block upsamples its predecessor layer while concatenating it with an earlier layer. This acts as "skip connections," which helps in retaining some of the finer details learned in the earlier layers. Afterwards, the concatenated layer undergoes a separable convolution layer to better learn those finer details added from the earlier layers.

It is important to note that while the layer concatenation does not require the two input layers to have the same depth, they must have the same spatial dimensions. That is easy to do here since all the layers here have spatial dimensions that are factors of 2 from one another. That is why the upsampling factor is 2. So a small layer of 32x32 becomes 64x64, which can be concatenated with one of the earlier layers, specifically a 64x64.

The separable convolution layer used in the decoder blocks is identical to the one used in the encoder blocks.

### Architecture Reasoning
Given the kernel size, stride, and upsampling factor chosen in the encoder and decoder blocks, the spatial dimensions between each layer and the next were either doubled or halved, excluding the transition from layer 2 to 3. By halving the layer sizes in layers 1 and 2, we reduced the computational demands on the GPU and allowed us to allocated more resources for making the network deeper. A deeper network results in better learning of the input features as opposed to a wider network.

I decided to start with a simple network and go from there. I actually did not need much parameter tuning to reach the course goal. I am assuming that has to do with the quality of the data provided. If the quality was any worse, I believe I would have had to further tune my parameters and maybe add more layers to the network.

It is important to note that in my work the output and input share spatial dimensions, but differ in channel depth. To have them share the same depth dimension, I could have introduced a 1x1 convolution to reduce the depth of the second decoder's output. However, I did not do that and when I trained the model, I reached the desired goal. This leads me to think that what is important here is retaining the spatial dimensions and not the channel depth.

### 1x1 Convolutions vs. Fully Connected Layers
I want to take a moment here to explain why the architecture incorporates one and not the other. 1x1 convolutions are usually used to either add or remove depth from the previous layer while retaining its spatial dimensions. In this project, the 1x1 convolution was used to add depth and introduce an additional non-linearity. After that layer, decoders were implemented to further train the network as well as upscale it to an output with the same spatial dimensions as the input. It is important to note that in my work the output and input share spatial dimensions, but differ in channel depth. The reason it is desirable to retain spatial dimensions - and basically the reason a 1x1 layer and decoders are used instead of a fully connected layer - is to allow the network to learn the features of an input image while knowing where those features are. This is important when you are trying to locate something in the image to track it. This project demands exactly that. The robot needs to know that the hero is in its vision field as well as where the hero is in that vision field.

A fully connected layer is computationally intensive. It connects every neuron in its input to its output. It also flattens the data into a vector of `n` classes, each corresponding to a certain label. This is used when we want to identify the object in the image, but don't care about where it is in the image. In the flattening process we lose all spatial information.

## Hyper Parameters

```python
learning_rate = 0.02
batch_size = 100
num_epochs = 50
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

The hyper parameters were obtained mostly through trial and error. I went through multiple trials prior to reaching the project goal Intersection over Union ratio.

The learning rate was initially chosen as 0.1; however, the model didn't do too well. I brought it down to 0.02, amongst other changes, and the IoU turned out to be 0.41. I believe 0.1 was a large value, causing the model to not reach the minimum of the error function effectively.

I started out with a batch size of 50 and then increased it to 100. I increased it as more training examples per epoch step should result in a better model. This however used a lot more memory space and significantly slowed down the training process. 

The number of epochs was initially 5, which was very small. I increased the value to 15 then 30. Each time the model performed better; therefore, I increased it to 50, but did not go above as the computation time became very long. With each epoch run, the model had the chance to go through a forward and back propagation improving the model weights.

I do not think I changed the steps per epoch, validation steps, or workers parameters.

## Final Thoughts
Although this model is capable of following a human wearing a red shirt, I do not think it can follow anything else without retraining it on new data. The neural network is limited to the data it trains on. No matter how good it gets at semantic segmentation, if the training data had no dogs, then it wouldn't be able to recognize them. If in the future we want the network to follow a different object, new data containing said object must be collected and the network must be retrained on them.

