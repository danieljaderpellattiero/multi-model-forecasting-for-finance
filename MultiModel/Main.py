from Ensemble import Ensemble as MultiModel

if __name__ == '__main__':
    multi_model = MultiModel(False, '1s', 3, 'mae',
                             ['^KS11'],
                             ['M1', 'M2', 'M3', 'M4'])
    multi_model.run()
