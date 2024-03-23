# Deep Residual Learning for Image Recognition

## Pass 1

    Title, Abstract, and conclusion. What the paper is talking about (15min around)

He's team presented a residual learning framework that simplifies the model training process. Instead of learning unreferenced functions, this new framework reformulated the layers as learing residual functions with reference to the layer inputs. The residual learning method is easier than previous methods and can gain accuracy from considerably increased depth.

However, training aggressive models with thousands layers may still be a problem. He's team trained a 1202 layer newtork which performed worse than a 110 layers network. Researhsers argue that this is due of overfitting.

## Pass 2

    From title to end. Omit those details such as proof. But need to knwo all figure and Table and their meaning in detail. May list important references if not familiar

The residual learning framework adds the identity value to a processed value to produce the output. This output then becomes the input for the next block. He's team developed several different ResNets, each with varying numbers of layers, normalization methods, and shortcut methods.Researchers discovered that these residual learning networks have achieved highly competitive accuracy while preventing the degradation problem. Moreover, the residual network performs well on various datasets, including CIFAR-10 and ImageNet 2012.

## Pass 3

    Know exactly the meaning of each word and sentence. Simulate whole training process in mind (2 hour around)
