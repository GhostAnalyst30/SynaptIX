import { Workflow } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function PipelineDocs() {
  return (
    <>
      <DocHeader
        title="Pipelines"
        description="Automatic machine learning end-to-end: AutoPipeline detects the task, cleans the data, compares models, tunes the best one and evaluates it. Pipeline offers the same steps in declarative form."
        icon={<Workflow className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">AutoPipeline — ML in One Line</h2>
        <CodeBlock
          title="auto.py"
          code={`import synaptix as sx

df = sx.load_dataset("penguins")

auto = sx.AutoPipeline(task="classification")
auto.fit(df, target="species")   # detect -> clean -> compare -> tune -> evaluate

print("Best model:", auto.best_model_)
print("Best params:", auto.best_params_)
print("Evaluation:", auto.evaluate())

# Predict on raw data — preprocessing is applied automatically
auto.predict(df.head(5))`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          If <code className="code-inline">task</code> is omitted, it is detected automatically from the
          target column type (classification for categoricals, regression for numerics).
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Declarative Pipeline</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Build the flow explicitly, step by step:
        </p>
        <CodeBlock
          title="declarative.py"
          code={`import synaptix as sx

df = sx.load_dataset("titanic")

pipe = (
    sx.Pipeline()
    .clean()                                  # missing values, dtypes
    .outliers(method="iqr")                   # optional outlier handling
    .model(sx.supervised.LogisticRegression())
    .tune({"C": [0.1, 1, 10]})               # GridSearch with CV
    .evaluate(cv=5)
)

pipe.fit(df, target="survived")
print(pipe.summary())`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="AutoPipeline"
            signature="AutoPipeline(task=None, models=None, cv=5)"
            description="End-to-end automatic ML. Compares every candidate model for the task, tunes the best one with GridSearch and evaluates with cross-validation."
            parameters={[
              { name: "task", type: "'classification' | 'regression' | None", description: "Task type; detected automatically when None.", default: "None" },
              { name: "models", type: "list | None", description: "Candidate models; defaults to the registry for the task.", default: "None" },
              { name: "cv", type: "int", description: "Cross-validation folds.", default: "5" },
            ]}
            example={`auto = sx.AutoPipeline(task="regression")
auto.fit(df, target="price")`}
          />

          <MethodCard
            name="Pipeline"
            signature="Pipeline()"
            description="Declarative builder with chainable steps: <code className='code-inline'>clean()</code>, <code className='code-inline'>outliers()</code>, <code className='code-inline'>model()</code>, <code className='code-inline'>tune()</code>, <code className='code-inline'>evaluate()</code>."
            example={`pipe = sx.Pipeline().clean().model(sx.supervised.SVR()).tune({"C": [1, 10]})`}
          />

          <MethodCard
            name="detect_task"
            signature="detect_task(y) -> str"
            description="Infer the task type from the target column: classification for categorical/object targets, regression for continuous ones."
            returns="str"
            example={`task = sx.pipeline.detect_task(df["price"])   # 'regression'`}
          />

          <MethodCard
            name="save / load"
            signature="pipe.save(path) / Pipeline.load(path)"
            description="Persist a trained pipeline (or any SynaptixModel) with pickle and reload it for production predictions."
            example={`auto.save("modelo.pkl")
auto2 = sx.AutoPipeline.load("modelo.pkl")`}
          />
        </div>
      </section>
    </>
  );
}
