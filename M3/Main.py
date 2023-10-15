import time
from M3 import M3

if __name__ == '__main__':
    begin = time.time()
    model = M3([False, '1s', '2015-01-01', 3, 3, [10, 5], 4, 50, 128, 4, 0],
               ['^KS11'])
    model.run()
    end = time.time()
    time_elapsed = end - begin
    print(f'Time elapsed: {time_elapsed / 3600} hours')
