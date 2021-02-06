import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor

from pytorch_catboost.playground_regression import _custom_loss
import pandas as pd
import pandas_bokeh

raw_diff = np.linspace(-1, 1, 101)
loss = _custom_loss(Tensor(raw_diff)).detach().numpy()
# plt.plot(raw_diff, loss)
# plt.xlabel("diff = (preds - targets)")
# plt.ylabel("loss")
# plt.title("Custom regression loss function - don't undershoot!")
# plt.show()

pandas_bokeh.output_file("Interactive Plot.html")
series = pd.Series(data=loss, index=raw_diff).plot_bokeh()

a = 0
