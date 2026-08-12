import pandas as pd
from synaptix.machineLearning import MachineLearning

import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

size_m2 = np.random.randint(40, 150, n)
rooms = np.random.randint(1, 6, n)

price = (
    size_m2 * 2.2 +
    rooms * 15 +
    np.random.normal(0, 10, n)
)

df = pd.DataFrame({
    "size_m2": size_m2,
    "rooms": rooms,
    "price": price.round(2)
})



ml = MachineLearning(df)
nn = ml.neural_network()

nn.data(
    X=["size_m2", "rooms"],
    y="price",
    task="regression",
    test_size=0.3
)

nn.build(
    input_dim=nn.X_train.shape[1],
    layers=[
        ("dense", 32, "relu"),
        ("dense", 16, "relu")
    ]
)

nn.fit(epochs=300)
nn.plot()
