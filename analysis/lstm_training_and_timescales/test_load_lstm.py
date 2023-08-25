from analysis.lstm_training_and_timescales.lstm_utils import load_lstm

lstm_test = load_lstm(
    base_path='../../trained_models',
    n_max=10,
    network_number=1,
    n_min=2,
    init=False,
    device='cpu',
)
