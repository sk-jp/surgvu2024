from collections.abc import Sequence
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchseg


class UnetBase(nn.Module):
    def __init__(
        self,
        encoder_name: str = "convnextv2_tiny",
        in_channels: int = 3,
        classes: int = 3,
        encoder_weights: bool = True,
        encoder_depth: int = 4,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        decoder_use_batchnorm: bool = True,             # True, False, "inplace"
        decoder_attention_type: Optional[str] = None,   # None, "scse"
    ) -> None:
        super(UnetBase, self).__init__()

        if "convnext" in encoder_name:
            head_upsampling = 2
        else:
            head_upsampling = 1
            
        self.model = torchseg.Unet(
            encoder_name,
            in_channels=in_channels,
            classes=classes,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type,
            head_upsampling=head_upsampling,
        )

    def forward(self, x) -> tuple:
        y = self.model(x)

        return y

if __name__ == "__main__":
    model = UnetBase("convnextv2_tiny", 3, 3)
    x = torch.rand((4, 3, 256, 256), dtype=torch.float32)
    y = model(x)
    print("y:", y.shape)
