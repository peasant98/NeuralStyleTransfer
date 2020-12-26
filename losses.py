import torch
import torch.nn as nn

from utils import gram_matrix


class StyleLoss(nn.Module):
    """
    gets the style loss.
    """
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mode = "none"
        self.crit = nn.MSELoss()

    def forward(self, x):
        G = gram_matrix(x)

        if self.mode == "capture":
            self.target = G.detach()
        elif self.mode == "loss":
            self.loss = self.crit(G, self.target)

        return x


class ContentLoss(nn.Module):
    """
    content loss.
    """
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mode = "none"
        # criterion, aka loss function
        self.crit = nn.MSELoss()

    def forward(self, x):
        """
        x is the feature map of the content image.
        """
        if self.mode == "capture":
            self.target = x.detach()

        elif self.mode == "loss":
            self.loss = self.crit(x, self.target)

        return x


class TVLoss(nn.Module):
    """
    total variation loss. encourages smoothness in the generated image.
    """
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input
