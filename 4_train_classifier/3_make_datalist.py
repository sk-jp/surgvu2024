import csv
import numpy as np
import os

#phase = 'train'
phase = 'valid'

label_dir = '/data/MICCAI2024_SurgVU/image_labels'

cases = {}
cases['train'] = [2, 8, 11, 36, 43, 49, 55, 59, 81, 84, 96, 110, 129, 153]
cases['valid'] = [24, 26, 30, 32, 41, 61, 82, 107, 133]

label_number = {'needle driver': 1,
                'monopolar curved scissors': 2,
                'force bipolar': 0,
                'clip applier': 3,
                'cadiere forceps': 8,
                'bipolar forceps': 4,
                'vessel sealer': 5,
                'permanent cautery hook/spatula': 9,
                'prograsp forceps': 6,
                'stapler': 10,
                'grasping retractor': 7,
                'tip-up fenestrated grasper': 11,
                'unknown': 12,
                'synchroseal': 12,
                'suction irrigator': 12,
                'bipolar dissector': 12,
                'curved scissors': 12,
                'crocodile grasper': 12,
                'tenaculum forceps': 12}
num_labels = 13

task_number = {'Range of motion': 0,
               'Rectal artery/vein': 1,
               'Retraction and collision avoidance': 2,
               'Skills application': 3,
               'Suspensory ligaments': 4,
               'Suturing': 5,
               'Uterine horn': 6,
               'nan': 7}

fps = 60

## main routine
label_files = []
for case in cases[phase]:
    label_files.append(f'{label_dir}/case_{case:03d}.csv')

# print a header
s = 'case,video_id,frame'
for n in range(num_labels):
    s += f',l{n}'
s += ',task'
print(s)

for label_file in label_files:
    case_id = os.path.basename(label_file).split('.')[0]
#    print(case_id)

    with open(label_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip the 1st line
        header = next(reader)
        # for each line
        for row in reader:
            # get data
            vid = row[0]
            sec = row[1]
            tools = row[2:6]
            # remove spaces
            for idx in range(len(tools)):
                tools[idx] = tools[idx].strip()
            task = row[6]
            if task == '':
                task = 'nan'
            
            # make a list
            s = f'{case_id}'
            s += f',{vid}'
            s += f',{int(sec)*fps}'
           
            label_ohv = [0] * num_labels
            for idx, tool in enumerate(tools):
                if tool != 'nan':
                    l = label_number[tool]
                    label_ohv[l] = 1
            if np.all(label_ohv == 0) or task == 'nan':
                continue
                
            for ohv in label_ohv:
                s += f',{ohv}'

            s += f',{task_number[task]:d}'
            print(s)
            
   