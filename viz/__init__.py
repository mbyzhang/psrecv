import matplotlib.pyplot as plt
import numpy as np

def plot_statistics(pipeline, fs: float, baudrate: float, start: float, end: float):
    plt.figure(1, figsize=(25, 18))
    plt.clf()

    # =======================

    plt.subplot(411)

    for i, x in enumerate(pipeline.cum_data["subcarrier_abs_amp"]):
        plt.plot(x, label=f"Absolute amplitude of f{i}")
    plt.xlim(start * fs, end * fs)
    plt.yscale("log")
    plt.ylim((1e-6, 10.0))
    plt.grid(True)
    plt.legend()

    # =======================

    plt.subplot(412)

    subcarrier_rela_amp = pipeline.cum_data["subcarrier_rela_amp"]
    if subcarrier_rela_amp.ndim == 1:
        plt.plot(subcarrier_rela_amp, label="Relative amplitude")
    else:
        for i, x in enumerate(subcarrier_rela_amp):
            plt.plot(x, label=f"Relative amplitude of f{i}")

    for x in np.where(pipeline.cum_data["symbol_rising_edges"] == 1)[0]:
        plt.axvline(x=x, c="red", alpha=0.5, lw=0.5)

    if "subcarrier_rela_amp_dc_blocked" in pipeline.cum_data:
        subcarrier_rela_amp_dc_blocked = pipeline.cum_data["subcarrier_rela_amp_dc_blocked"]
        if subcarrier_rela_amp_dc_blocked.ndim == 1:
            plt.plot(subcarrier_rela_amp_dc_blocked, label="DC-blocked relative amplitude")

    plt.xlim(start * fs, end * fs)
    # plt.ylim((-1.5, 1.5))
    plt.axhline(y=0.0, color="yellow")
    plt.legend()
    plt.grid(True)

    # =======================

    plt.subplot(413)
    plt.plot(pipeline.cum_data["agc_gain"], label="AGC gain")
    plt.plot(pipeline.cum_data["input_snr"], label="Input SNR")
    plt.xlim(start * fs, end * fs)
    plt.legend()
    plt.grid(True)

    # =======================

    plt.subplot(414)
    plt.plot(pipeline.cum_data["symbols"], 'ro-', label="Symbol")
    plt.xlim(start * baudrate, end * baudrate)
    plt.legend()

    plt.show()
