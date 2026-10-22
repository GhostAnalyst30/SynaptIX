import { Gauge } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function MetricsDocs() {
  return (
    <>
      <DocHeader
        title="Metrics"
        description="Evaluation metrics for every task: regression (MAE, MSE, RMSE, R2, MAPE), classification (accuracy, precision, recall, F1) and clustering (silhouette, Calinski-Harabasz, Davies-Bouldin), plus full per-class reports."
        icon={<Gauge className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="metrics.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]
model = sx.supervised.RandomForestClassifier().fit(X, y)
y_pred = model.predict(X)

# Per-class report with confusion matrix
report = sx.metrics.classification_report(y, y_pred)
print(report)

# Individual metrics
print(sx.metrics.classification_metrics(y, y_pred, average="macro"))

# Regression
print(sx.metrics.regression_metrics(y_true_reg, y_pred_reg))

# Clustering
print(sx.metrics.clustering_metrics(X, km.labels_))`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="regression_metrics"
            signature="regression_metrics(y_true, y_pred) -> dict"
            description="MAE, MSE, RMSE, R2 and MAPE in one call. Also available inside every model's <code className='code-inline'>evaluate()</code>."
            returns="dict of float"
            example={`sx.metrics.regression_metrics(y_true, y_pred)`}
          />

          <MethodCard
            name="classification_metrics"
            signature="classification_metrics(y_true, y_pred, average='weighted') -> dict"
            description="Accuracy, precision, recall and F1 with a configurable averaging strategy for multiclass problems."
            parameters={[
              { name: "average", type: "str", description: "'weighted', 'macro', 'micro' or 'binary'.", default: "'weighted'" },
            ]}
            returns="dict of float"
            example={`sx.metrics.classification_metrics(y, y_pred, average="macro")`}
          />

          <MethodCard
            name="clustering_metrics"
            signature="clustering_metrics(X, labels) -> dict"
            description="Silhouette score, Calinski-Harabasz index and Davies-Bouldin index for clustering quality."
            returns="dict of float"
            example={`sx.metrics.clustering_metrics(X, km.labels_)`}
          />

          <MethodCard
            name="classification_report"
            signature="classification_report(y_true, y_pred) -> dict"
            description="Full per-class report: precision, recall, F1, support and the confusion matrix."
            returns="dict"
            example={`report = sx.metrics.classification_report(y_test, y_pred)
print(report["per_class"])
print(report["confusion_matrix"])`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Exporting Metrics</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Every metric result is also an <code className="code-inline">EvaluationResult</code> that can be
          exported to a ViewX dashboard:
        </p>
        <CodeBlock
          title="export.py"
          code={`evaluation = model.evaluate(X_test, y_test)
evaluation.to_html("evaluacion.html", theme="dark_enterprise")   # requires synaptix[viewx]`}
        />
      </section>
    </>
  );
}
