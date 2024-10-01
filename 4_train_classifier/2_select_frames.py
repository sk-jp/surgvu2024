import csv

datalist_dir = './datalist'

tool_count = {'train': [0] * 13,
              'valid': [0] * 13}
task_count = {'train': [0] * 8,
              'valid': [0] * 8}

# tool 5,9,10,11
cases = {'train': [2, 8, 11, 36, 43, 49, 55, 59, 81, 84, 96, 110, 129, 153],
         'valid': [24, 26, 30, 32, 41, 61, 82, 107, 133]}

for phase in ['train', 'valid']:
    for case_id in cases[phase]:
        datalist_file = f'{datalist_dir}/case_{case_id:03d}_datalist.csv'
        with open(datalist_file, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip the 1st line
            header = next(reader)
    
            # for each line
            for row in reader:
                # tool
                if int(row[16]) != 7:
                    # task is not 'none'
                    for idx in range(3, 16):
                        tool_count[phase][idx-3] += int(row[idx])
                # task  
                task_count[phase][int(row[16])] += 1

# print
for phase in ['train', 'valid']:
    print("phase:", phase)
    print("cases:", sorted(cases[phase]))
    s = ''
    for i in range(13):
        s += f'{tool_count[phase][i]},'
    print(s)