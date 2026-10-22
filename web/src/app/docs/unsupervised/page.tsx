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
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="cluster.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X = df.drop(columns="species")

# KMeans with built-in diagnostics
km = sx.unsupervised.KMeans(n_clusters=3, random_state=42)
km.fit(X)
print("Inertia:", km.inertia_)

metrics = sx.metrics.clustering_metrics(X, km.labels_)
print(metrics)   # silhouette, Calinski-Harabasz, Davies-Bouldin

km.plot_elbow(max_k=10)   # elbow curve renders inline`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="KMeans"
            signature="KMeans(n_clusters=8, random_state=None)"
            description="K-means clustering with <code className='code-inline'>elbow()</code> / <code className='code-inline'>plot_elbow()</code> and <code className='code-inline'>silhouette_scores()</code> helpers."
            parameters={[
              { name: "n_clusters", type: "int", description: "Number of clusters.", default: "8" },
              { name: "random_state", type: "int | None", description: "Reproducibility seed.", default: "None" },
            ]}
            example={`km = sx.unsupervised.KMeans(n_clusters=3).fit(X)
print(km.silhouette_scores(X, k_max=8))   # silhouette per k`}
          />

          <MethodCard
            name="DBSCAN"
            signature="DBSCAN(eps=0.5, min_samples=5)"
            description="Density-based clustering — finds arbitrary-shaped clusters and marks noise points."
            example={`db = sx.unsupervised.DBSCAN(eps=0.8, min_samples=5).fit(X)
print("Clusters:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))`}
          />

          <MethodCard
            name="HierarchicalClustering"
            signature="HierarchicalClustering(n_clusters=3, linkage='ward')"
            description="Agglomerative clustering over a distance linkage."
            example={`hc = sx.unsupervised.HierarchicalClustering(n_clusters=3).fit(X)`}
          />

          <MethodCard
            name="GaussianMixture"
            signature="GaussianMixture(n_components=3)"
            description="Probabilistic clustering with soft assignments and per-cluster covariance."
            example={`gmm = sx.unsupervised.GaussianMixture(n_components=3).fit(X)
print(gmm.predict_proba(X.head(3)))`}
          />

          <MethodCard
            name="PCA"
            signature="PCA(n_components=None)"
            description="Principal component analysis. After fit, <code className='code-inline'>explained_variance_ratio_</code> shows the retained variance per component."
            returns="transformed DataFrame"
            example={`pca = sx.unsupervised.PCA(n_components=2).fit(X)
Z = pca.transform(X)
print(pca.explained_variance_ratio_)`}
          />

          <MethodCard
            name="TSNE"
            signature="TSNE(n_components=2, perplexity=30)"
            description="Non-linear embedding for 2D/3D visualization of high-dimensional data."
            example={`emb = sx.unsupervised.TSNE(n_components=2).fit_transform(X)`}
          />
        </div>
      </section>
    </>
  );
}
