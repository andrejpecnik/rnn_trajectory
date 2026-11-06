| Predictor | ADE [px] | Error at 1 s [px] | FDE 1.5 s [px] | Hit rate |
|---|---:|---:|---:|---:|
| GRU (RNN) | 18.1 | 24.7 | 44.7 | 75.0 % |
| LSTM (RNN) | 17.9 | 24.5 | 44.3 | 74.3 % |
| MLP (feed-forward NN) | 19.6 | 26.8 | 47.2 | 70.9 % |
| Constant turn rate | 34.0 | 45.9 | 87.3 | 56.6 % |
| Constant velocity | 33.6 | 46.0 | 77.7 | 55.1 % |
| Static (no prediction) | 58.4 | 76.4 | 108.6 | 35.1 % |

| GRU tank vs. | Rounds | GRU wins | Opponent wins | Draws | GRU accuracy | Opponent accuracy |
|---|---:|---:|---:|---:|---:|---:|
| LSTM (RNN) | 200 | 91 | 109 | 0 | 63.5 % | 66.1 % |
| MLP (feed-forward NN) | 200 | 100 | 96 | 4 | 65.4 % | 64.2 % |
| Constant turn rate | 200 | 136 | 64 | 0 | 67.2 % | 50.6 % |
| Constant velocity | 200 | 126 | 72 | 2 | 66.5 % | 52.6 % |
| Static (no prediction) | 200 | 184 | 16 | 0 | 66.5 % | 28.0 % |
