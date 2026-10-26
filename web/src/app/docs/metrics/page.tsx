import { Gauge } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function MetricsDocs() {
  return (
    <>
      <DocHeader
        title="Metrics"
        description="Evaluation metrics for every task — regression, classification and clustering — with a metrics=[...] selector on every function, plus full per-class reports."
        icon={<Gauge className="w-6 h-6" />}
        version="0.2.0"
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

# Pick exactly the metrics you want (case-insensitive)
print(sx.metrics.classification_metrics(
    y, y_pred, metrics=["accuracy", "mcc", "roc_auc"], y_proba=model.predict_proba(X),
))

# Regression
print(sx.metrics.regression_metrics(y_true_reg, y_pred_reg, metrics=["rmse", "r2", "medae"]))

# Clustering (+ external metrics against real labels)
print(sx.metrics.clustering_metrics(X, km.labels_, y_true=y))

# What is available for each task?
sx.metrics.available_metrics("classification")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="regression_metrics"
            signature="regression_metrics(y_true, y_pred, metrics=None) -> dict"
            description="MAE, MSE, RMSE, R2 and MAPE by default. Also available on request: MedAE, explained_variance and max_error. The same selector exists in every model's evaluate() and fit_evaluate()."
            parameters={[
              { name: "metrics", type: "list[str]", description: "Subset to compute (case-insensitive).", default: "None (MAE, MSE, RMSE, R2, MAPE)" },
            ]}
            returns="dict of float"
            example={`sx.metrics.regression_metrics(y_true, y_pred, metrics=["rmse", "medae"])`}
          />

          <MethodCard
            name="classification_metrics"
            signature="classification_metrics(y_true, y_pred, average='weighted', metrics=None, y_proba=None, labels=None) -> dict"
            description="Accuracy, precision, recall and F1 by default. Also available: balanced_accuracy, mcc, cohen_kappa, and — when y_proba is given — roc_auc and log_loss."
            parameters={[
              { name: "average", type: "str", description: "'weighted', 'macro', 'micro' or 'binary'.", default: "'weighted'" },
              { name: "metrics", type: "list[str]", description: "Subset to compute (case-insensitive).", default: "None (accuracy, precision, recall, f1)" },
              { name: "y_proba", type: "ndarray", description: "Predicted probabilities, required for roc_auc and log_loss.", default: "None" },
              { name: "labels", type: "list", description: "Class order of the y_proba columns (model.classes_).", default: "None" },
            ]}
            returns="dict of float"
            example={`sx.metrics.classification_metrics(y, y_pred, average="macro", metrics=["f1", "mcc"])`}
          />

          <MethodCard
            name="clustering_metrics"
            signature="clustering_metrics(X, labels, y_true=None, metrics=None) -> dict"
            description="Silhouette, Calinski-Harabasz and Davies-Bouldin. With y_true it also computes ari, nmi, homogeneity, completeness and v_measure against the real labels."
            returns="dict of float"
            example={`sx.metrics.clustering_metrics(X, km.labels_, y_true=y, metrics=["silhouette", "ari"])`}
          />

          <MethodCard
            name="available_metrics"
            signature="available_metrics(task) -> list[str]"
            description="Names accepted by metrics=[...] for 'regression', 'classification' or 'clustering'."
            returns="list[str]"
            example={`sx.metrics.available_metrics("clustering")`}
          />

          <MethodCard
            name="classification_report"
            signature="classification_report(y_true, y_pred, plot=False, labels=None) -> dict"
            description="Prints scikit-learn's text report and returns the global metrics (accuracy, precision, recall, f1) plus <code className='code-inline'>por_clase</code> with precision, recall, F1 and support per class. <code className='code-inline'>plot=True</code> draws the confusion matrix."
            returns="dict"
            example={`report = sx.metrics.classification_report(y_test, y_pred, plot=True)
print(report["accuracy"])
print(report["por_clase"])`}
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
