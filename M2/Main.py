import time
from M2 import M2

if __name__ == '__main__':
    begin = time.time()
    model = M2([True, '1s', '2015-01-01', 3, 3, [10, 5], 4, [.8, .15], 2, 0],
               [
                   'IT0000072618_INTESASANPAOLO',
                   'IT0003128367_ENEL',
                   'IT0003261697_AZIMUT',
                   'IT0003796171_POSTEITALIANE',
                   'IT0003856405_LEONARDO',
                   'IT0004176001_PRYSMIAN',
                   'IT0004965148_MONCLER',
                   'IT0005366767_NEXI',
                   'NL0011585146_FERRARI',
                   'NL0015435975_CAMPARI'
               ])
    model.run()
    end = time.time()
    time_elapsed = end - begin
    print(f'Time elapsed: {time_elapsed} seconds')
    print(f'Time elapsed: {time_elapsed / 60} minutes')
    print(f'Time elapsed: {time_elapsed / 3600} hours')
