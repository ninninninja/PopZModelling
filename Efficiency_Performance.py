import numpy as np
import os

def read_TimeRecord():
    path_here = os.getcwd()
    print(path_here)

    res = {}
    res['Total'] = 0

    with open('output.txt', 'r') as txtfile:
        for line in txtfile.readlines():
            if ':' not in line:
                continue
            try:
                key = line.split(':')[0]
                value = line.split(':')[1]
                res['Total'] += float(value)

                if key not in res:
                    res['{}'.format(key)] = float(value)
                else:
                    res[key] +=float(value)
            except:
                continue

        print('finished')

    return res

ans = read_TimeRecord()
for k in ans:
    print('It cost {0} sec on {1}.'.format(ans[k], k))

