from synaptix.machineLearning import MachineLearning
from synaptix.datasets import load_dataset
from keras.layers import Dropout, BatchNormalization


df = load_dataset("iris.csv")
df["target"] = (df["species"] == "setosa").astype(int)

ml = MachineLearning(df)
nn = ml.neural_network()


nn.data(X=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
        y="target",
        task="classification",
        test_size=0.3,
        )

nn.build(
    input_dim=nn.X_train.shape[1],
    layers=[
        ("dense", 64, "relu"),
        BatchNormalization(),
        Dropout(0.2),
        ("dense", 8, "relu")
    ]
)
nn.fit(epochs=50)

nn.plot(bins=30)
