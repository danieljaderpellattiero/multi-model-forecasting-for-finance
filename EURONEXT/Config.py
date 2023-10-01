import pendulum as time

from colorama import Fore, Style


class Config:

    def __init__(self, uuid, params) -> None:
        self.__uuid = uuid
        self.__period_begin = time.parse(params[0], tz=None)
        self.__period_end = time.parse(params[1], tz=None)
        self.__period_duration = (self.__period_end - self.__period_begin).days
        self.__period_trading_days = time.period(self.__period_begin, self.__period_end).range('days')
        self.__daily_test_runs_amount = params[2]
        self.__daily_test_runs_schedules = params[3]
        self.__daily_test_runs_datasets_splits_percentages = params[4]

        if not self.check_test_runs_consistency():
            print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                  f'The specified test runs amount and schedules are inconsistent. {Style.RESET_ALL}')
            exit(1)

    @property
    def duration(self):
        return self.__period_duration

    @property
    def trading_days(self):
        return self.__period_trading_days

    @property
    def dly_tr_amt(self):
        return self.__daily_test_runs_amount

    def check_test_runs_consistency(self) -> bool:
        return True if len(self.__daily_test_runs_schedules) == self.__daily_test_runs_amount + 1 else False

    def get_test_run_schedules(self, dt_instance, shift) -> tuple:
        return (dt_instance.add(hours=self.__daily_test_runs_schedules[shift]).to_atom_string(),
                dt_instance.add(hours=self.__daily_test_runs_schedules[shift + 1]).to_atom_string())

    def get_test_run_splits(self, shift) -> tuple:
        return (int(shift * self.__daily_test_runs_datasets_splits_percentages[0]),
                int(shift * (self.__daily_test_runs_datasets_splits_percentages[0] +
                             self.__daily_test_runs_datasets_splits_percentages[1])))
