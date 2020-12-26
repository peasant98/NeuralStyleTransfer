import torch


def gram_matrix(x):
    a, b, c, d = x.size()
    # batch size, channels, h, w
    features = x.view(a*b, c*d)
    G = torch.mm(features, features.t())
    # normalization step by amount of features
    """
    dimensions of matrix are independent of image input size.
    dimensions are f X f, where f is the amount of filters
    the matrix multiplication takes care of any h X w
    image dimension, since it's measure between each filter.
    """

    return G.div(a * b * c * d)
