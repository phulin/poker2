import torch.nn as nn

from p2.core.structured_config import NonlinearityType
from p2.models.activation_utils import get_activation


def test_get_activation_supports_leaky_relu():
    activation = get_activation(NonlinearityType.leaky_relu)

    assert isinstance(activation, nn.LeakyReLU)
    assert activation.negative_slope == 0.01
