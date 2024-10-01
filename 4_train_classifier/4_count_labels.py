import csv

#datalist_file = 'datalist/train_datalist.csv'
datalist_file = 'datalist/valid_datalist.csv'

tool_count = [0] * 13
task_count = [0] * 8

with open(datalist_file, 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # skip the 1st line
    header = next(reader)
    
    # for each line
    for row in reader:
        # tool
        for idx in range(3, 16):
            tool_count[idx-3] += int(row[idx])
        # task
        task_count[int(row[16])] += 1

# print
header = ''
for i in range(13):
    header += f'tool{i},'
for i in range(8):
    header += f'task{i},'
print(header)
s = ''
for i in range(13):
    s += f'{tool_count[i]},'
for i in range(8):
    s += f'{task_count[i]},'
print(s)