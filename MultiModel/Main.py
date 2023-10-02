from Ensemble import Ensemble as MultiModel

if __name__ == '__main__':
    multi_model = MultiModel(True, '1s', 1, 'mae',
                             ['IT0000072618_INTESASANPAOLO'],
                             ['M1', 'M2', 'M3'])
    multi_model.run()
