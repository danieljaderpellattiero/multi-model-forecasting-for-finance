import os
import matplotlib
import matplotlib.pyplot as plt

from PyEMD import CEEMDAN

matplotlib.use('Agg')


class TSDecomposer:
    data_plot_colors = ['royalblue', 'goldenrod', 'coral']

    def __init__(self) -> None:
        self.__tool = CEEMDAN()
        self.__tool.noise_seed(11022004)

    def plot_components(self, imfs, residue, time_axis, ticker, test_run, phase=1) -> None:
        mode = 'original' if phase == 1 else 'normalized'
        images_path = f'./images/data-preprocessing/{ticker}'
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        figure, axis = plt.subplots(len(imfs) + 1, 1, sharex='all', figsize=(16, 9))
        figure.suptitle(f'CEEMDAN {mode} decomposition of {ticker} (test run {test_run})')
        for imf_index, imf in enumerate(imfs):
            axis[imf_index].plot(time_axis, imf, 'g', label=f'imf_{imf_index}', color=self.data_plot_colors[0])
            axis[imf_index].legend(loc='best')
            axis[imf_index].grid(True)
        axis[len(imfs)].plot(time_axis, residue, 'b', label='residue', color=self.data_plot_colors[1])
        axis[len(imfs)].x_label = 'Time [days]'
        axis[len(imfs)].legend(loc='best')
        axis[len(imfs)].grid(True)
        plt.savefig(f'{images_path}/test_run_{test_run}_components_phase_{phase}.png')
        plt.close()

    @staticmethod
    def export_components(time_series_imfs, time_series_residue) -> dict:
        time_series_components = {}
        for imf_index in range(time_series_imfs.shape[0]):
            time_series_components.update({f'imf_{imf_index}': time_series_imfs[imf_index].reshape(-1, 1)})
        time_series_components.update({'residue': time_series_residue.reshape(-1, 1)})
        return time_series_components

    def decompose(self, time_series_dataframe, ticker, test_run) -> dict:
        self.__tool.ceemdan(time_series_dataframe.to_numpy(copy=True).flatten(), time_series_dataframe.index)
        time_series_imfs, time_series_residue = self.__tool.get_imfs_and_residue()
        self.plot_components(time_series_imfs, time_series_residue, time_series_dataframe.index, ticker, test_run, 1)
        return self.export_components(time_series_imfs, time_series_residue)
