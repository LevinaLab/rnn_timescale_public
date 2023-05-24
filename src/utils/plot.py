from matplotlib import pyplot as plt


def set_plot_params():
    """Set default plot parameters."""
    plt.rcParams["axes.edgecolor"] = "k"
    plt.rcParams["axes.facecolor"] = "w"
    plt.rcParams["axes.linewidth"] = "0.8"
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['savefig.dpi'] = 300

    plt.rcParams['pdf.fonttype'] = 42  # prepare as vector graphic
    plt.rcParams['ps.fonttype'] = 42

    plt.rcParams["font.family"] = "Helvetica"
    return
