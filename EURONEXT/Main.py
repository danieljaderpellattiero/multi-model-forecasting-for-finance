from ENX_DL import EuronextDataLoader

if __name__ == '__main__':
    data_loader = EuronextDataLoader([
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
    data_loader.load_data()
    data_loader.prepare_test_runs(1)
