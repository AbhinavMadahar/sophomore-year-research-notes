# Notes From Papers I've Read

## Conditional Image Generation with PixelCNN Decoders

This paper introduces a model which can **generate images** from a label.
Unlike a GAN, it can give a probability of how likely the image is to be
accurate. This model can be conditioned on prior information.

This model is based on the PixelRNN model. There is also a PixelCNN model,
which is faster to train because it convonets are better at parallelization,
but the PixelRNN gives better performance. Because the image datasets are
so massive, we want to parallelize as much as possible while maintaining
the RNN's performance, so we use a gated variant of PixelCNN, *gated PixelCNN*.

We then introduce a conditional variant of gated PixelCNN, *Conditional
PixelCNN*.

PixelCNNs model the joint probability distribution of pixels over an image
$\mathbf{x}$ as the following product of condititional distributions, where
$x_i$ is a single pixel:

$$p(x) = \prod_{i = 1}^{n^2} p(x_i | x_1, \cdots, x_{i-1}).$$
