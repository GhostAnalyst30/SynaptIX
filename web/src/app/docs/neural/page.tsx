import { Brain } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function NeuralDocs() {
  return (
    <>
      <DocHeader
        title="Neural Networks"
        description="MLP, CNN and LSTM networks over Keras/TensorFlow with training history plots and forecasting. Requires the dl extra — lazy imports keep the core light."
        icon={<Brain className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Requirements</h2>
        <CodeBlock
          title="bash"
          code={`pip install 'synaptix[dl]'   # installs tensorflow`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          Imports are lazy: <code className="code-inline">import synaptix</code> works without
          TensorFlow; only <code className="code-inline">synaptix.neural</code> requires it.
          Note: TensorFlow is not available in the browser playground.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">MLP Classifier</h2>
        <CodeBlock
          title="mlp.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]

net = sx.neural.MLP(task="classification", hidden_layers=(64, 32), dropout=0.2)
net.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, early_stopping=True)
net.evaluate(X, y, metrics=["accuracy", "f1"])
net.predict_proba(X.head())
net.plot_history()   # loss + accuracy curves
net.summary()        # Keras architecture`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">LSTM Forecasting</h2>
        <CodeBlock
          title="lstm.py"
          code={`import synaptix as sx
import numpy as np

series = np.sin(np.linspace(0, 40, 500)) + np.random.default_rng(0).normal(0, 0.05, 500)

net = sx.neural.LSTMNet(window=24, units=(64,))
net.fit(series, epochs=30)
forecast = net.forecast(series, steps=12)   # seeds with the last 24 values
net.evaluate(series)                        # one-step-ahead metrics
net.plot_history()`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Train / Test Evaluation</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          <code className="code-inline">fit_evaluate()</code> splits, trains and reports the chosen
          metrics on train and test. Training arguments go in{" "}
          <code className="code-inline">fit_params</code> (Keras output is silent by default). For{" "}
          <code className="code-inline">LSTMNet</code> the split is temporal: the last{" "}
          <code className="code-inline">test_size</code> values of the series are the test set, scored
          with one-step-ahead predictions.
        </p>
        <CodeBlock
          title="neural_train_test.py"
          code={`net = sx.neural.MLP(task="classification", hidden_layers=(64, 32))
result = net.fit_evaluate(
    X, y, test_size=0.2,
    metrics=["accuracy", "f1", "roc_auc"],
    fit_params={"epochs": 50, "batch_size": 16},
)
print(result.to_frame())

cnn = sx.neural.CNN(input_shape=(28, 28, 1), n_classes=10)
cnn.fit_evaluate(images, labels, test_size=0.25, metrics=["accuracy", "log_loss"])

lstm = sx.neural.LSTMNet(window=24)
lstm.fit_evaluate(series, test_size=0.2, metrics=["rmse", "mae"])`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="MLP"
            signature="MLP(task='regression', hidden_layers=(64, 32), activation='relu', dropout=0.0, scale=True, optimizer='adam')"
            description="Dense network for tabular data. <code className='code-inline'>fit(X, y, epochs=100, batch_size=32, validation_split=0.2, early_stopping=True, verbose=1)</code>, <code className='code-inline'>predict(X)</code>, <code className='code-inline'>predict_proba(X)</code> (classification). Text labels are encoded and decoded automatically."
            parameters={[
              { name: "task", type: "'regression' | 'classification'", description: "Output layer and loss.", default: "'regression'" },
              { name: "hidden_layers", type: "tuple[int, ...]", description: "Neurons per hidden layer.", default: "(64, 32)" },
              { name: "dropout", type: "float", description: "Dropout after each hidden layer.", default: "0.0" },
              { name: "scale", type: "bool", description: "Standardize X before training.", default: "True" },
            ]}
          />

          <MethodCard
            name="CNN"
            signature="CNN(input_shape, n_classes, conv_blocks=(32, 64), dense_units=128, dropout=0.3, optimizer='adam')"
            description="Convolutional network for images. <code className='code-inline'>fit(X, y, epochs=20, batch_size=32, validation_split=0.2, early_stopping=True, verbose=1)</code>, <code className='code-inline'>predict(X)</code>, <code className='code-inline'>predict_proba(X)</code>."
          />

          <MethodCard
            name="LSTMNet"
            signature="LSTMNet(window=10, units=(50,), scale=True, optimizer='adam')"
            description="Recurrent network for univariate series. <code className='code-inline'>fit(series, epochs=50, batch_size=32, validation_split=0.1, early_stopping=True, verbose=1)</code>, <code className='code-inline'>forecast(series, steps=1)</code>, static <code className='code-inline'>make_windows(series, window)</code>, <code className='code-inline'>evaluate(series, verbose=True, metrics=None)</code>, <code className='code-inline'>fit_evaluate(series, test_size=0.2, metrics=None, verbose=True, fit_params=None)</code>."
          />

          <MethodCard
            name="Shared methods"
            signature="evaluate(X, y, verbose=True, metrics=None) · fit_evaluate(X, y, test_size=0.2, metrics=None, ..., fit_params=None) · plot_history() · summary() · save(path='model.keras') · load(path='model.keras')"
            description="Available on all three networks. The underlying Keras model is in <code className='code-inline'>.model</code> and the training history in <code className='code-inline'>.history</code>."
          />
        </div>
      </section>
    </>
  );
}
