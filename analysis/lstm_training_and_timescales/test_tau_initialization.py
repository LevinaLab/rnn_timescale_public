import numpy as np

from src.lstm_utils.lstm_utils import LSTM_custom, load_lstm


def _get_forget_gate_bias(lstm):
    bias_forget = lstm.lstm.bias_ih_l0[1 * lstm.hidden_size:2 * lstm.hidden_size].detach().numpy()
    bias_forget += lstm.lstm.bias_hh_l0[1 * lstm.hidden_size:2 * lstm.hidden_size].detach().numpy()
    return bias_forget


def _forget_gate_to_tau(forget_gate):
    return 1 / (1 - forget_gate)


def _forget_gate_bias_to_tau(forget_gate_bias):
    return _forget_gate_to_tau(
        1 / (1 + np.exp(-forget_gate_bias))
    )


print("LSTM with tau initialization")
lstm = LSTM_custom(
    hidden_size=500,
    num_readout_heads=100,
    initial_tau=None,
)
bias_forget = _get_forget_gate_bias(lstm)
tau_forget = _forget_gate_bias_to_tau(bias_forget)
print(f"forget_gate_bias = {bias_forget.mean():.5f} +- {bias_forget.std():.5f}")
print(f"tau_forget = {tau_forget.mean():.5f} +- {tau_forget.std():.5f}")

print("\nLSTM from previous training")
lstm_init = load_lstm(
    "../../trained_models",
    None,
    1,
    'cumulative',
    init=True,
)

bias_forget = _get_forget_gate_bias(lstm_init)
tau_forget = _forget_gate_bias_to_tau(bias_forget)
print(f"forget_gate_bias = {bias_forget.mean():.5f} +- {bias_forget.std():.5f}")
print(f"tau_forget = {tau_forget.mean():.5f} +- {tau_forget.std():.5f}")
