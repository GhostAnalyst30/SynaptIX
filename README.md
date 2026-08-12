# SynaptIX

**Librería integral de Machine Learning para Python.** Una API unificada y en español que cubre aprendizaje supervisado, no supervisado, por refuerzo, redes neuronales y preprocesamiento de datos — pensada para AI Engineers y analistas de modelos.

```bash
pip install synaptix            # núcleo (sklearn incluido)
pip install synaptix[dl]        # + tensorflow (redes neuronales y DQN)
pip install synaptix[all]       # todo
```

## Inicio rápido

```python
import synaptix as sx

# 1. Cargar un dataset incluido
df = sx.load_dataset("iris")

# 2. Preparar los datos
X, y = df.drop(columns="species"), df["species"]
X_train, X_test, y_train, y_test = sx.preprocessing.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar y evaluar
model = sx.supervised.RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
model.evaluate(X_test, y_test, plot=True)  # métricas + matriz de confusión
```

## Módulos

| Módulo | Contenido |
|---|---|
| `synaptix.preprocessing` | `Scaler`, `Encoder`, `Imputer`, `DataCleaner`, `detect_outliers`, `remove_outliers`, `train_test_split` |
| `synaptix.supervised` | 8 regresores (lineal, ridge, lasso, árbol, random forest, gradient boosting, SVR, KNN) y 7 clasificadores (logística, árbol, random forest, gradient boosting, SVM, KNN, naive bayes) |
| `synaptix.unsupervised` | `KMeans` (con codo y silhouette), `DBSCAN`, `HierarchicalClustering`, `GaussianMixture`, `PCA`, `TSNE` |
| `synaptix.reinforcement` | `QLearningAgent`, `SARSAAgent`, `DQNAgent`, entorno `GridWorld` |
| `synaptix.neural` | `MLP`, `CNN`, `LSTMNet` sobre Keras (requiere `synaptix[dl]`) |
| `synaptix.metrics` | `regression_metrics`, `classification_metrics`, `clustering_metrics`, `classification_report` |
| `synaptix.model_selection` | `cross_validate`, `GridSearch`, `compare_models` (AutoML-lite) |
| `synaptix.visualization` | Gráficas de modelos (matriz de confusión, ROC, curva de aprendizaje, residuos, frontera de decisión, `model_report`) y EDA (distribuciones, correlación, boxplots, scatter matrix, nulos) |
| `synaptix.datasets` | `iris`, `penguins`, `titanic`, `sp500_companies`, `course_completion` |
| `synaptix.legacy` | API clásica v0.x (`IntelligenceArtificial`, `MachineLearning`, ...) |

## Ejemplos por área

### Preprocesamiento

```python
from synaptix.preprocessing import DataCleaner

cleaner = DataCleaner(impute_strategy="median", scale_method="standard")
cleaner.analyze(df)                    # reporte: nulos, outliers, tipos
df_listo = cleaner.clean(df, target="precio")
```

### Comparar modelos automáticamente

```python
from synaptix.model_selection import compare_models

tabla = compare_models(X, y, task="classification", cv=5)
# Entrena 7 modelos con validación cruzada y devuelve un ranking
```

### Clustering (no supervisado)

```python
from synaptix.unsupervised import KMeans

KMeans.elbow(X, k_max=10)              # elegir k con el método del codo
km = KMeans(n_clusters=3)
labels = km.fit_predict(X)
km.plot(X)                             # scatter coloreado por cluster
print(km.silhouette(X))
```

### Aprendizaje por refuerzo

```python
from synaptix.reinforcement import GridWorld, QLearningAgent

env = GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])
agent = QLearningAgent(env.n_states, env.n_actions)
agent.train(env, episodes=500)
agent.plot_rewards()
env.render(agent.policy())             # política aprendida con flechas
```

### Redes neuronales

```python
from synaptix.neural import MLP, LSTMNet

# Clasificación tabular
net = MLP(task="classification", hidden_layers=(64, 32), dropout=0.2)
net.fit(X_train, y_train, epochs=50)
net.evaluate(X_test, y_test)
net.plot_history()

# Series de tiempo
lstm = LSTMNet(window=12, units=(64,))
lstm.fit(serie, epochs=50)
futuro = lstm.forecast(serie, steps=6)
```

### Gráficas a partir de datos y modelos

```python
from synaptix.visualization import (
    plot_distributions,     # histogramas de todas las numéricas
    plot_correlation,       # heatmap de correlación
    plot_boxplots,          # outliers, segmentables por categoría
    plot_missing,           # % de nulos por columna
    plot_residuals,         # diagnóstico de regresión
    plot_decision_boundary, # frontera de decisión 2D
    model_report,           # reporte visual completo en una llamada
)

plot_correlation(df)
plot_boxplots(df, by="species")
model_report(modelo, X_test, y_test)  # métricas + gráficos según la tarea
```

### Búsqueda de hiperparámetros

```python
from synaptix.supervised import RandomForestClassifier
from synaptix.model_selection import GridSearch

search = GridSearch(
    RandomForestClassifier(),
    {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]},
)
mejor_modelo = search.fit(X, y)
print(search.best_params_)
```

## API legacy (v0.x)

Las clases de la versión 0.x siguen disponibles: `IntelligenceArtificial` (agentes, búsqueda, algoritmos genéticos, chatbots), `MachineLearning` y `NaturalLanguageProcessing`. La API key de OpenRouter ahora se lee de la variable de entorno `OPENROUTER_API_KEY`.

```python
from synaptix import IntelligenceArtificial

ia = IntelligenceArtificial(backend_ia=("mistral", ""))  # usa OPENROUTER_API_KEY
```

## Licencia

MIT. Autor: Emmanuel Ascendra — [github.com/GhostAnalyst30/SynaptIX](https://github.com/GhostAnalyst30/SynaptIX)
