from Ensemble import Ensemble as MultiModel

if __name__ == '__main__':
    multi_model = MultiModel(['MSFT'], 3, ['M1', 'M2', 'M3'], 'mae')
    multi_model.run()
