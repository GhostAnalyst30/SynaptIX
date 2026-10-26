import { Boxes } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function UnsupervisedDocs() {
  return (
    <>
      <DocHeader
        title="Unsupervised Models"
        description="Clustering and dimensionality reduction: KMeans with elbow and silhouette analysis, DBSCAN, hierarchical clustering, Gaussian mixtures, PCA with explained variance and TSNE."
        icon={<Boxes className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="cluster.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X = df.drop(columns="species")

# Choose k: elbow (inertia per k) and silhouette per k
sx.unsupervised.KMeans.elbow(X, k_max=10, plot=True)
sx.unsupervised.KMeans.silhouette_scores(X, k_max=10)

km = sx.unsupervised.KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(X)
print("Silhouette:", km.silhouette(X))
print(km.centroids_)
km.plot(X, features=(2, 3))

km.evaluate(X, y=df["species"])   # silhouette, CH, DB + ARI/NMI vs real labels`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Train / Test Evaluation</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Every clustering model and <code className="code-inline">PCA</code> has{" "}
          <code className="code-inline">evaluate(X, metrics=...)</code> and{" "}
          <code className="code-inline">fit_evaluate(X, test_size=...)</code>: the model is fitted on the
          train part and scored on both parts, so you can check whether the structure generalizes.
          KMeans and GaussianMixture assign test points with their own{" "}
          <code className="code-inline">predict</code>; DBSCAN and hierarchical clustering use the
          cluster of the nearest training point.
        </p>
        <CodeBlock
          title="cluster_train_test.py"
          code={`km = sx.unsupervised.KMeans(n_clusters=3)
result = km.fit_evaluate(X, y=df["species"], test_size=0.3)   # y is optional
print(result.to_frame())
# silhouette, calinski_harabasz, davies_bouldin, ari, nmi, ..., n_clusters

sx.unsupervised.DBSCAN(eps=0.6).fit_evaluate(X, test_size=0.3, metrics=["silhouette"])
# also reports ruido_pct (share of noise points)

pca = sx.unsupervised.PCA(n_components=2)
pca.fit_evaluate(X, test_size=0.2)
# varianza_explicada, mse_reconstruccion, rmse_reconstruccion on train and test`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          All clustering models share: <code className="code-inline">fit(X)</code>,{" "}
          <code className="code-inline">fit_predict(X)</code>, <code className="code-inline">predict(X)</code>,{" "}
          <code className="code-inline">labels_</code>, <code className="code-inline">silhouette(X)</code>,{" "}
          <code className="code-inline">plot(X, features=(0, 1))</code>,{" "}
          <code className="code-inline">evaluate(X, y=None, metrics=None, verbose=True)</code> and{" "}
          <code className="code-inline">fit_evaluate(X, y=None, test_size=0.2, metrics=None, random_state=42, verbose=True)</code>.
          The fitted scikit-learn estimator is available in <code className="code-inline">.model</code>.
        </p>
        <div className="method-list">
          <MethodCard
            name="KMeans"
            signature="KMeans(n_clusters=3, random_state=42, **kwargs)"
            description="K-means clustering. Extra: <code className='code-inline'>centroids_</code> and the static helpers <code className='code-inline'>KMeans.elbow(X, k_max=10, plot=True)</code> (inertia per k) and <code className='code-inline'>KMeans.silhouette_scores(X, k_max=10, plot=True)</code>, both returning <code className='code-inline'>{'{'}k: value{'}'}</code>. <code className='code-inline'>fit_evaluate</code> stores the inertia in <code className='code-inline'>result.info</code>."
            parameters={[
              { name: "n_clusters", type: "int", description: "Number of clusters.", default: "3" },
              { name: "random_state", type: "int", description: "Reproducibility seed.", default: "42" },
            ]}
            example={`km = sx.unsupervised.KMeans(n_clusters=3).fit(X)
scores = sx.unsupervised.KMeans.silhouette_scores(X, k_max=8, plot=False)`}
          />

          <MethodCard
            name="DBSCAN"
            signature="DBSCAN(eps=0.5, min_samples=5, **kwargs)"
            description="Density-based clustering — finds arbitrary-shaped clusters and labels noise points as -1. <code className='code-inline'>evaluate</code> reports <code className='code-inline'>n_clusters</code> and <code className='code-inline'>ruido_pct</code>."
            example={`db = sx.unsupervised.DBSCAN(eps=0.8, min_samples=5).fit(X)
print("Clusters:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))`}
          />

          <MethodCard
            name="HierarchicalClustering"
            signature="HierarchicalClustering(n_clusters=3, linkage='ward', **kwargs)"
            description="Agglomerative clustering. <code className='code-inline'>linkage</code>: 'ward', 'complete', 'average' or 'single'."
            example={`hc = sx.unsupervised.HierarchicalClustering(n_clusters=3, linkage="average").fit(X)`}
          />

          <MethodCard
            name="GaussianMixture"
            signature="GaussianMixture(n_components=3, random_state=42, **kwargs)"
            description="Probabilistic clustering with soft assignments via <code className='code-inline'>predict_proba(X)</code>. <code className='code-inline'>fit_evaluate</code> stores BIC and AIC in <code className='code-inline'>result.info</code>."
            example={`gmm = sx.unsupervised.GaussianMixture(n_components=3).fit(X)
print(gmm.predict_proba(X.head(3)))`}
          />

          <MethodCard
            name="PCA"
            signature="PCA(n_components=2, **kwargs)"
            description="Principal component analysis: <code className='code-inline'>fit</code>, <code className='code-inline'>transform</code>, <code className='code-inline'>fit_transform</code>, <code className='code-inline'>inverse_transform</code>, <code className='code-inline'>explained_variance()</code> (pd.Series per component), <code className='code-inline'>plot_variance()</code>, <code className='code-inline'>evaluate(X, metrics=None)</code>, <code className='code-inline'>fit_evaluate(X, test_size=0.2)</code>. A float <code className='code-inline'>n_components</code> (e.g. 0.95) keeps that share of variance."
            returns="np.ndarray"
            example={`pca = sx.unsupervised.PCA(n_components=2)
Z = pca.fit_transform(X)
print(pca.explained_variance())
pca.plot_variance()`}
          />

          <MethodCard
            name="TSNE"
            signature="TSNE(n_components=2, perplexity=30.0, random_state=42, **kwargs)"
            description="Non-linear embedding for 2D/3D visualization: <code className='code-inline'>fit_transform(X)</code> and <code className='code-inline'>plot(X, labels=None)</code>."
            example={`emb = sx.unsupervised.TSNE().fit_transform(X)
sx.unsupervised.TSNE().plot(X, labels=df["species"])`}
          />
        </div>
      </section>
    </>
  );
}
