from monai.transforms import MapTransform


class Blackd(MapTransform):
    def __init__(self, keys, roi):
        assert(len(keys) == 1)
        super(Blackd, self).__init__(keys)
        self.roi = roi

    def __call__(self, data):
        d = dict(data)

        # get image
        image = d[self.keys[0]]

        # paint the area black
        sx = self.roi[0]
        sy = self.roi[1]
        ex = self.roi[2]
        ey = self.roi[3]
        image[:, sy:ey, sx:ex] = 0

        d[self.keys[0]] = image
        
        return d
