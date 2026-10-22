import { Filter } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function PreprocessingDocs() {
  return (
    <>
      <DocHeader
        title="Preprocessing"
        description="Data preparation that keeps column names: DataCleaner, Scaler, Encoder, Imputer, outlier detection/removal and train_test_split — every transformer returns DataFrames ready for models or export."
        icon={<Filter className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="prepare.py"
          code={`import synaptix as sx

df = sx.load_dataset("titanic")

cleaner = sx.preprocessing.DataCleaner()
cleaner.fit(df)
df_clean = cleaner.transform(df)      # same columns, clean values
df_new = cleaner.transform(new_data)  # apply the same cleaning later

scaler = sx.preprocessing.Scaler()
X = scaler.fit_transform(df_clean[["age", "fare"]])

encoder = sx.preprocessing.Encoder()
df_enc = encoder.fit_transform(df_clean)   # categorical -> numeric`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="DataCleaner"
            signature="DataCleaner()"
            description="Fit/transform cleaner for missing values, dtypes and constant columns. <code className='code-inline'>transform()</code> applies the fitted rules to new data."
            returns="pd.DataFrame"
            example={`cleaner = sx.preprocessing.DataCleaner()
df2 = cleaner.fit_transform(df)`}
          />

          <MethodCard
            name="Scaler"
            signature="Scaler(method='standard')"
            description="Scale numeric columns with <code className='code-inline'>standard</code> or <code className='code-inline'>minmax</code>; preserves column names and non-numeric columns."
            parameters={[
              { name: "method", type: "str", description: "'standard' (zero mean, unit variance) or 'minmax' (0-1).", default: "'standard'" },
            ]}
            example={`scaler = sx.preprocessing.Scaler(method="minmax")
X_scaled = scaler.fit_transform(X)`}
          />

          <MethodCard
            name="Encoder"
            signature="Encoder()"
            description="Encode categorical columns numerically so any model can consume them."
            example={`df_enc = sx.preprocessing.Encoder().fit_transform(df)`}
          />

          <MethodCard
            name="Imputer"
            signature="Imputer(strategy='mean')"
            description="Fill missing values with a per-column strategy: mean, median or mode."
            parameters={[
              { name: "strategy", type: "str", description: "'mean', 'median' or 'most_frequent'.", default: "'mean'" },
            ]}
            example={`df_imp = sx.preprocessing.Imputer(strategy="median").fit_transform(df)`}
          />

          <MethodCard
            name="detect_outliers / remove_outliers"
            signature="detect_outliers(df, method='iqr') -> pd.Series / remove_outliers(df, method='iqr') -> pd.DataFrame"
            description="Detect or drop outlier rows with the IQR rule or z-score."
            parameters={[
              { name: "method", type: "str", description: "'iqr' or 'zscore'.", default: "'iqr'" },
            ]}
            returns="pd.Series (boolean mask) / pd.DataFrame"
            example={`mask = sx.preprocessing.detect_outliers(df[["age", "fare"]], method="zscore")
print(mask.sum(), "outlier rows")
df_clean = sx.preprocessing.remove_outliers(df)`}
          />

          <MethodCard
            name="train_test_split"
            signature="train_test_split(X, y, test_size=0.2, random_state=None)"
            description="Scikit-learn's split re-exported for convenience — the canonical way to split in SynaptIX."
            returns="tuple of DataFrames/arrays"
            example={`X_tr, X_te, y_tr, y_te = sx.preprocessing.train_test_split(X, y, test_size=0.2, random_state=42)`}
          />
        </div>
      </section>
    </>
  );
}
