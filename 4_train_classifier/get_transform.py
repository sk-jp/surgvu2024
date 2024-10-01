import monai.transforms as mt
import numpy as np

from blackd import Blackd


def get_transform(conf_augmentation):
    """ Get augmentation function
        Args:
            conf_augmentation (Dict): dictionary of augmentation parameters
    """
    def get_object(trans):
        if trans.name in {'Compose', 'OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(mt, trans.name)(augs_tmp, **trans.params)

        if trans.name == "NormalizeIntensityd":
            trans.params.subtrahend = np.array(trans.params.subtrahend)
            trans.params.divisor = np.array(trans.params.divisor)

        if hasattr(mt, trans.name):
            return getattr(mt, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]

    return mt.Compose(augs)

if __name__ == '__main__':
    from read_yaml import read_yaml
    config = './convnext.yaml'    
    cfg = read_yaml(fpath=config)
    trans = get_transform(cfg.Transform.train)
    
    from monai.transforms import RandomizableTrait, Transform
    for _tr in trans.flatten().transforms:
        print(_tr)
        if isinstance(_tr, RandomizableTrait) or not isinstance(_tr, Transform):
            print("Yes")
        else:
            print("No")
    
    """
    filename = '/data/MICCAI2023_SurgToolLoc/training_data/images/clip_000000_000180.jpg'
    x = trans({"image": filename})
#    print('x:', x)
    print('x:', x["image"].shape)
    import cv2
    import numpy as np
    cv2.imshow("image", x["image"].numpy().astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])
    cv2.waitKey(0)
    """
    