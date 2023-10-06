class Config:

    def __init__(self, model_uuid, params) -> None:
        self.__uuid = model_uuid
        self.__enx_data = params[0]
        self.__enx_data_frequency = params[1]
        self.__period_begin = params[2]
        self.__test_runs_amount = params[3]
        self.__test_runs_delay_year = params[4]
        self.__test_runs_delay_months = params[5]
        self.__test_runs_step_size = params[6]
        self.__test_runs_split_percentages = params[7]
        self.__window_size = params[8]
        self.__verbosity = params[9]

    @property
    def uuid(self) -> str:
        return self.__uuid

    @property
    def enx_data(self) -> bool:
        return self.__enx_data

    @property
    def enx_data_freq(self) -> str:
        return self.__enx_data_frequency

    @property
    def period_begin(self) -> str:
        return self.__period_begin

    @property
    def tr_amt(self) -> int:
        return self.__test_runs_amount

    @property
    def tr_delay_y(self) -> int:
        return self.__test_runs_delay_year

    @property
    def tr_delay_m(self) -> int:
        return self.__test_runs_delay_months

    @property
    def tr_step_size(self) -> int:
        return self.__test_runs_step_size

    @property
    def window_size(self) -> int:
        return self.__window_size

    @property
    def verbosity(self) -> int:
        return self.__verbosity

    def get_test_run_splits(self, shift) -> tuple:
        return (int(shift * self.__test_runs_split_percentages[0]),
                int(shift * (self.__test_runs_split_percentages[0] +
                             self.__test_runs_split_percentages[1])))
