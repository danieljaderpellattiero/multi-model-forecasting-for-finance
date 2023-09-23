class Config:

    def __init__(self, model_uuid, params) -> None:
        self.__uuid = model_uuid
        self.__period_begin = params[0]
        self.__test_runs_amount = params[1]
        self.__test_runs_delay_year = params[2]
        self.__test_runs_delay_month = params[3]
        self.__test_runs_step_size = params[4]
        self.__window_size = params[5]
        self.__verbosity = params[6]

    @property
    def uuid(self) -> str:
        return self.__uuid

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
        return self.__test_runs_delay_month

    @property
    def tr_step_size(self) -> int:
        return self.__test_runs_step_size

    @property
    def window_size(self) -> int:
        return self.__window_size

    @property
    def verbosity(self) -> int:
        return self.__verbosity
