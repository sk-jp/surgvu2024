import glob

phase = 'train'
#phase = 'valid'
train_ratio = 0.8

img_dirs = ['/data/EndoVisSub2017-RoboticInstrumentSegmentation/test_960x768/images',
            '/data/EndoVisSub2018-RoboticSceneSegmentation/train_data_960x768/images']

for img_dir in img_dirs:
    img_files = sorted(glob.glob(f'{img_dir}/*.png'))
    num_train_data = int(len(img_files) * train_ratio)
    if phase == 'train':
        img_files = img_files[:num_train_data]
    elif phase == 'valid':
        img_files = img_files[num_train_data:]
    
    for img_file in img_files:
        print(img_file)
            