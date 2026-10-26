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
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="prepare.py"
          code={`import synaptix as sx

df = sx.load_dataset("titanic")

cleaner = sx.preprocessing.DataCleaner()
cleaner.analyze(df)                              # quality report: nulls, dtypes, duplicates...
df_clean = cleaner.clean(df, target="Survived")  # impute -> encode -> scale (target untouched)
df_new = cleaner.transform(new_data)             # apply the same fitted rules later

scaler = sx.preprocessing.Scaler(method="robust")
X = scaler.fit_transform(df[["Age", "Fare"]])

encoder = sx.preprocessing.Encoder(method="onehot")
df_enc = encoder.fit_transform(df, columns=["Sex", "Embarked"])`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="DataCleaner"
            signature="DataCleaner(impute_strategy='median', encode_method='onehot', scale_method='standard')"
            description="Complete cleaning pipeline: drop duplicates -> impute nulls -> encode categoricals -> scale numerics. The fitted Imputer, Encoder and Scaler are stored in <code className='code-inline'>imputer_</code>, <code className='code-inline'>encoder_</code> and <code className='code-inline'>scaler_</code>."
            parameters={[
              { name: "impute_strategy", type: "str", description: "'mean', 'median' or 'most_frequent'.", default: "'median'" },
              { name: "encode_method", type: "str", description: "'onehot' or 'label'.", default: "'onehot'" },
              { name: "scale_method", type: "str | None", description: "'standard', 'minmax', 'robust' or None to skip scaling.", default: "'standard'" },
            ]}
            example={`cleaner = sx.preprocessing.DataCleaner(scale_method=None)
report = cleaner.analyze(df)            # dict with nulls, dtypes, duplicates, outliers
df_train = cleaner.clean(df, target="price")
df_test = cleaner.transform(df_new)    # same output columns as clean()`}
          />

          <MethodCard
            name="DataCleaner.analyze"
            signature="analyze(df, verbose=True) -> dict"
            description="Reports data quality problems: nulls per column, dtypes, duplicated rows, outliers and high-cardinality columns."
            returns="dict"
          />

          <MethodCard
            name="DataCleaner.clean"
            signature="clean(df, target=None, drop_duplicates=True) -> pd.DataFrame"
            description="Fits and applies the full cleaning. The <code className='code-inline'>target</code> column is kept unchanged and excluded from encoding and scaling."
            returns="pd.DataFrame"
          />

          <MethodCard
            name="DataCleaner.transform"
            signature="transform(df) -> pd.DataFrame"
            description="Applies the transformations fitted by <code className='code-inline'>clean()</code> to new data, producing the same columns."
            returns="pd.DataFrame"
          />

          <MethodCard
            name="Scaler"
            signature="Scaler(method='standard')"
            description="Scales numeric data. Methods: <code className='code-inline'>fit</code>, <code className='code-inline'>transform</code>, <code className='code-inline'>fit_transform</code>, <code className='code-inline'>inverse_transform</code>. Returns a DataFrame when the input is a DataFrame."
            parameters={[
              { name: "method", type: "str", description: "'standard' (zero mean, unit variance), 'minmax' (0-1) or 'robust' (median/IQR).", default: "'standard'" },
            ]}
            example={`scaler = sx.preprocessing.Scaler(method="minmax")
X_scaled = scaler.fit_transform(X)
X_back = scaler.inverse_transform(X_scaled)`}
          />

          <MethodCard
            name="Encoder"
            signature="Encoder(method='onehot')"
            description="Encodes categorical columns. <code className='code-inline'>fit(df, columns=None)</code>, <code className='code-inline'>transform(df)</code>, <code className='code-inline'>fit_transform(df, columns=None)</code>. Without <code className='code-inline'>columns</code> it encodes every object/category column."
            parameters={[
              { name: "method", type: "str", description: "'onehot' (dummy columns) or 'label' (integer codes).", default: "'onehot'" },
            ]}
            example={`df_enc = sx.preprocessing.Encoder("label").fit_transform(df, columns=["Sex"])`}
          />

          <MethodCard
            name="Imputer"
            signature="Imputer(strategy='mean', fill_value=None)"
            description="Fills missing values per column: <code className='code-inline'>fit(df)</code>, <code className='code-inline'>transform(df)</code>, <code className='code-inline'>fit_transform(df)</code>."
            parameters={[
              { name: "strategy", type: "str", description: "'mean', 'median', 'most_frequent' or 'constant'.", default: "'mean'" },
              { name: "fill_value", type: "Any", description: "Value used when strategy='constant'.", default: "None" },
            ]}
            example={`df_imp = sx.preprocessing.Imputer(strategy="median").fit_transform(df)`}
          />

          <MethodCard
            name="detect_outliers"
            signature="detect_outliers(df, columns=None, method='iqr', threshold=1.5) -> pd.DataFrame"
            description="Boolean mask with the same shape as the analysed numeric columns; True marks an outlier value."
            parameters={[
              { name: "columns", type: "list | None", description: "Columns to analyse (default: all numeric).", default: "None" },
              { name: "method", type: "str", description: "'iqr' or 'zscore'.", default: "'iqr'" },
              { name: "threshold", type: "float", description: "IQR multiplier (1.5) or z-score cutoff (e.g. 3).", default: "1.5" },
            ]}
            returns="pd.DataFrame (boolean mask)"
            example={`mask = sx.preprocessing.detect_outliers(df, columns=["Age", "Fare"], method="zscore", threshold=3)
print(mask.sum())   # outliers per column`}
          />

          <MethodCard
            name="remove_outliers"
            signature="remove_outliers(df, columns=None, method='iqr', threshold=1.5) -> pd.DataFrame"
            description="Copy of the DataFrame without the rows that contain outliers in the analysed columns."
            returns="pd.DataFrame"
            example={`df_clean = sx.preprocessing.remove_outliers(df, columns=["Fare"])`}
          />

          <MethodCard
            name="train_test_split"
            signature="train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)"
            description="Scikit-learn's split re-exported for convenience. Every model also offers <code className='code-inline'>fit_evaluate(X, y, test_size=...)</code>, which splits, trains and reports train/test metrics in one call."
            returns="list of DataFrames/arrays"
            example={`X_tr, X_te, y_tr, y_te = sx.preprocessing.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)`}
          />
        </div>
      </section>
    </>
  );
}
