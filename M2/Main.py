import time
from M2 import M2

if __name__ == '__main__':
    begin = time.time()
    model = M2([False, '1s', '2015-01-01', 3, 3, [10, 5], 5, [.8, .15], 2, 0],
               ['^KS11'])
    model.run()
    end = time.time()
    time_elapsed = end - begin
    print(f'Time elapsed: {time_elapsed / 3600} hours')
