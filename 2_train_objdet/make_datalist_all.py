import glob

phase = 'train'
#phase = 'valid'

fold = 5
valid_cv = 4

# synth
#img_dir = '/data/MICCAI2024_SurgVU/synthetic/images'
#img_dir = '/data/MICCAI2024_SurgVU/synthetic2/images'
img_dir = '/data/MICCAI2024_SurgVU/synthetic3/images'

img_files = sorted(glob.glob(f'{img_dir}/*.png'))
if phase == 'train':
    for n, img_file in enumerate(img_files):
        if n % fold != valid_cv:
            print(img_file)
elif phase == 'valid':
    for n, img_file in enumerate(img_files):
        if n % fold == valid_cv:
            print(img_file)
            
# endovis
img_dirs = ['/data/EndoVisSub2017-RoboticInstrumentSegmentation/test_960x768/images',
            '/data/EndoVisSub2018-RoboticSceneSegmentation/train_data_960x768/images']

for img_dir in img_dirs:
    img_files = sorted(glob.glob(f'{img_dir}/*.png'))
    if phase == 'train':
        for n, img_file in enumerate(img_files):
            if n % fold != valid_cv: 
                print(img_file)
    elif phase == 'valid':
        for n, img_file in enumerate(img_files):
            if n % fold == valid_cv:       
                print(img_file)
                        
