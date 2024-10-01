import glob
import os
import shutil

dir = 'results'
subdirs = glob.glob(f'{dir}/*/*')

for subdir in subdirs:
    if os.path.isdir(subdir):
        yaml_files = glob.glob(f'{subdir}/*.yaml')
        ckpt_files = glob.glob(f'{subdir}/*.ckpt')
        
        if len(yaml_files) > 0:
            if len(ckpt_files) == 0:
                print('remove:', subdir)
                shutil.rmtree(subdir)
            else:
                print('remain:', subdir)
        else:
            print('skip:', subdir)
