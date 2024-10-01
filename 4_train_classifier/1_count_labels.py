import csv

datalist_file = 'datalist/all_datalist.csv'

tool_count = {}
task_count = {}
for id in range(155):
    tool_count[f'case_{id:03d}'] = [0] * 13
    task_count[f'case_{id:03d}'] = [0] * 8

with open(datalist_file, 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # skip the 1st line
    header = next(reader)
    
    # for each line
    for row in reader:
        # case id
        case_id = row[0]
        # tool
        if int(row[16]) != 7:
            # if task is not 'nan', count tools
            for idx in range(3, 16):
                tool_count[case_id][idx-3] += int(row[idx])
        # task
        task_count[case_id][int(row[16])] += 1

# print
header = 'case_id'
for i in range(13):
    header += f',tool{i}'
for i in range(8):
    header += f',task{i}'
print(header)
for id in range(155):
    case_id = f'case_{id:03d}'
    s = f'{case_id}'
    for i in range(13):
        s += f',{tool_count[case_id][i]}'
    for i in range(8):
        s += f',{task_count[case_id][i]}'
    print(s)
