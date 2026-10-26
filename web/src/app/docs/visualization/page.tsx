import { BarChart3 } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

const EDA = [
  ["plot_distributions(df, columns=None, bins=30, color='#4c9f70')", "Histograms of the numeric columns in a grid."],
  ["plot_correlation(df, columns=None, method='pearson', cmap='RdYlGn')", "Correlation heatmap (pearson, spearman or kendall)."],
  ["plot_boxplots(df, columns=None, by=None)", "Boxplots per numeric column (useful for outliers), optionally segmented by a categorical column."],
  ["plot_scatter_matrix(df, columns=None, target=None, max_cols=6)", "Pairwise scatter matrix, optionally coloured by a categorical target."],
  ["plot_missing(df)", "Bar chart with the percentage of missing values per column."],
];

export default function VisualizationDocs() {
  return (
    <>
      <DocHeader
        title="Visualization"
        description="Matplotlib/seaborn helpers for model diagnostics (confusion matrix, ROC, learning curve, residuals, decision boundary, feature importance) and exploratory data analysis."
        icon={<BarChart3 className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="plots.py"
          code={`import synaptix as sx
from synaptix import visualization as viz

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]
model = sx.supervised.RandomForestClassifier().fit(X, y)

viz.plot_confusion_matrix(y, model.predict(X))
viz.plot_feature_importance(model, top_n=10)
viz.plot_learning_curve(model, X, y, cv=5)
viz.plot_decision_boundary(model, X, y, features=(2, 3))
report = viz.model_report(model, X, y)   # metrics + the relevant plots

viz.plot_correlation(df)
viz.plot_scatter_matrix(df, target="species")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Model Diagnostics</h2>
        <div className="method-list">
          <MethodCard
            name="plot_confusion_matrix"
            signature="plot_confusion_matrix(y_true, y_pred, labels=None, cmap='Blues')"
            description="Confusion matrix heatmap."
          />
          <MethodCard
            name="plot_roc_curve"
            signature="plot_roc_curve(y_true, y_proba)"
            description="ROC curve with AUC for a binary classifier. Pass the positive-class probability, e.g. <code className='code-inline'>model.predict_proba(X)[:, 1]</code>."
          />
          <MethodCard
            name="plot_learning_curve"
            signature="plot_learning_curve(model, X, y, cv=5, scoring=None)"
            description="Train and validation score versus training-set size — shows over/underfitting."
          />
          <MethodCard
            name="plot_feature_importance"
            signature="plot_feature_importance(model, top_n=15)"
            description="Horizontal bar chart of <code className='code-inline'>model.feature_importances()</code>."
          />
          <MethodCard
            name="plot_predictions / plot_residuals"
            signature="plot_predictions(y_true, y_pred) · plot_residuals(model, X, y)"
            description="Regression: predicted vs. real values, and residuals vs. predictions with their distribution."
          />
          <MethodCard
            name="plot_decision_boundary"
            signature="plot_decision_boundary(model, X, y, features=(0, 1), resolution=200)"
            description="Decision regions over two features (the model is refitted on those two columns)."
          />
          <MethodCard
            name="plot_clusters"
            signature="plot_clusters(X, labels, features=(0, 1), title='Clusters')"
            description="Scatter of two features coloured by cluster label."
          />
          <MethodCard
            name="model_report"
            signature="model_report(model, X, y) -> dict"
            description="Full visual report in one call. Classification: metrics + confusion matrix (+ ROC when binary). Regression: metrics + real vs. predicted + residual analysis."
            returns="dict"
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Exploratory Data Analysis</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Function</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {EDA.map(([f, d]) => (
                <tr key={f} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 pr-4 font-mono text-xs text-accent">{f}</td>
                  <td className="py-2.5 text-muted">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
