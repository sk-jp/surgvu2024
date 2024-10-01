import torch.nn as nn
from torch import Tensor
import timm
from timm.utils import freeze


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1,
        freeze_stem: bool = True,
    ) -> None:
        super(TimmModel, self).__init__()
            
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
        )
        
        if freeze_stem:
            freeze(self, 'model.stem')

#        data_config = timm.data.resolve_model_data_config(self.model)
#        print("data_config:", data_config)

    def forward(self, x) -> Tensor:
        y = self.model(x)

        return y

if __name__ == "__main__":
    from timm.utils import freeze
    
    model = TimmModel(model_name="convnext_base.fb_in22k_ft_in1k_384",
#                      model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
                      pretrained=True,
                      in_channels=6,
                      num_classes=1,
                      freeze_stem=True)
    
    submodules = [n for n, _ in model.model.stem.named_children()]
    print(submodules)
    submodules = [n for n, _ in model.model.stages.named_children()]
    print(submodules)
    freeze(model, 'model.stem')
#    freeze(model, submodules[:submodules.index('layer2') + 1])
#    print(model.model.stem[0])
#    print(model.model.stem[0].weight.requires_grad)
#    print(model.model.stages[0].weight.requires_grad)    
    
#    x = torch.rand((4, 6, 384, 384), dtype=torch.float32)
#    y = model(x)
#    print("y:", y.shape)
