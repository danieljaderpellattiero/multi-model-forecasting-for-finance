from M2 import M2

if __name__ == '__main__':
    model = M2([False, '1s', '2015-01-01', 3, 3, [10, 5], 4, [.8, .15], 2, 0],
               ['^KS11'])
    model.run()
