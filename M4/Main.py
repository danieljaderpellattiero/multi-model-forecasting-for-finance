from M4 import M4

if __name__ == '__main__':
    model = M4([False, '1s', '2015-01-01', 3, 3, [10, 5], 4, 50, 128, 10, 0],
               ['^KS11'])
    model.run()
