# Very Deep Convolutional Networks for Large-Scale Image Recognition

## Pass 1

    Title, Abstract, and conclusion. What the paper is talking about (15min around)

The Visual Geometry Group investigated the effect of convolutional network depth on its accuracy. They conducted a thorough evaluation of networks using very small convolution filters. The results show that pushing the depth of CNNs to 16-19 weight layers significantly improves accuracy. Besides, they found that their models perform well across a wide range of datasets and tasks.

## Pass 2

    From title to end. Omit those details such as proof. But need to knwo all figure and Table and their meaning in detail. May list important references if not familiar

The whole structure is generally similar to other CNN models. However, after comparing it with the old AlexNet, the VGG team dropped some functions, such as normalization, which brought no improvement in performance but led to increased memory consumption and computation time. Besides, the VGG team adopted many 3x3 convolutional layers instead of larger 5x5 or 7x7 layers. 3x3 convolutional layers achieve the same performance as the larger 5x5 or 7x7 layers by stacking two or three convolutional layers together, but use fewer parameters. Moreover, the VGG team used 1x1 layers in their models to increase the nonlinearity of the decision-making part.

VGG uses several blocks that share similar structure in convolutional part.

3x3 layse is the smallest size to capture the notion of left/right, up/down center.

## Pass 3

    Know exactly the meaning of each word and sentence. Simulate whole training process in mind (2 hour around)
