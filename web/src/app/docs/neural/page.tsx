import { Brain } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

export default function NeuralDocs() {
  return (
    <>
      <DocHeader
        title="Neural Networks"
        description="MLP, CNN and LSTM networks over Keras/TensorFlow with training history plots and forecasting. Requires the dl extra — lazy imports keep the core light."
        icon={<Brain className="w-6 h-6" />}
        version="0.1.7"
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

net = sx.neural.MLP(hidden_layers=(64, 32), epochs=100)
net.fit(X, y, validation_split=0.2)
net.evaluate(X, y)
net.plot_history()   # loss + accuracy curves`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">LSTM Forecasting</h2>
        <CodeBlock
          title="lstm.py"
          code={`import synaptix as sx
import numpy as np

series = np.sin(np.linspace(0, 40, 500)) + np.random.default_rng(0).normal(0, 0.05, 500)

net = sx.neural.LSTMNet(window=24, units=64)
net.fit(series, epochs=30)
forecast = net.forecast(steps=12)
net.plot_history()`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Networks</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border">
            <h3 className="font-mono text-xs text-accent mb-1">MLP</h3>
            <p className="text-xs text-muted">Dense network for tabular classification/regression.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border">
            <h3 className="font-mono text-xs text-accent mb-1">CNN</h3>
            <p className="text-xs text-muted">Convolutional network for image-shaped inputs.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border">
            <h3 className="font-mono text-xs text-accent mb-1">LSTMNet</h3>
            <p className="text-xs text-muted">Recurrent network for sequences with .forecast().</p>
          </div>
        </div>
        <p className="text-sm text-muted leading-relaxed mt-4">
          All three share <code className="code-inline">fit</code>,{" "}
          <code className="code-inline">evaluate</code>,{" "}
          <code className="code-inline">plot_history</code> and{" "}
          <code className="code-inline">forecast</code> (LSTM).
        </p>
      </section>
    </>
  );
}
