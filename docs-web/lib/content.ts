export type DocSection = {
  heading: string;
  body?: string;
  code?: string;
  codeTitle?: string;
  viz?: "kmeans" | "gradient" | "gridworld";
};

export type DocModule = {
  slug: string;
  name: string;
  title: string;
  intro: string;
  sections: DocSection[];
};

export const MODULES: DocModule[] = [
  {
    slug: "preprocessing",
    name: "synaptix.preprocessing",
    title: "Preprocesamiento de datos",
    intro:
      "Todo lo necesario para dejar un DataFrame listo para entrenar: escalado, codificación de categóricas, imputación de nulos, detección de outliers y un limpiador automático.",
    sections: [
      {
        heading: "DataCleaner: limpieza automática",
        body: "Analiza el DataFrame (nulos, tipos, duplicados, outliers, cardinalidad) y aplica el pipeline estándar imputar → codificar → escalar en una sola llamada. La columna objetivo se preserva sin transformar.",
        code: `from synaptix.preprocessing import DataCleaner

cleaner = DataCleaner(
    impute_strategy="median",   # mean | median | most_frequent
    encode_method="onehot",     # onehot | label
    scale_method="standard",    # standard | minmax | robust | None
)

cleaner.analyze(df)             # imprime el reporte de calidad
df_listo = cleaner.clean(df, target="precio")`,
        codeTitle: "data_cleaner.py",
      },
      {
        heading: "Transformadores individuales",
        body: "Scaler, Encoder e Imputer siguen el patrón fit / transform / fit_transform y devuelven DataFrames preservando los nombres de columna.",
        code: `from synaptix.preprocessing import Scaler, Encoder, Imputer

X_escalado = Scaler("robust").fit_transform(X)      # resistente a outliers
df_codificado = Encoder("onehot").fit_transform(df, columns=["ciudad"])
df_completo = Imputer("median").fit_transform(df)   # rellena NaN`,
        codeTitle: "transformers.py",
      },
      {
        heading: "Outliers y división de datos",
        body: "detect_outliers devuelve una máscara booleana por columna (métodos IQR o z-score); remove_outliers elimina las filas afectadas. train_test_split es el de scikit-learn, re-exportado por conveniencia.",
        code: `from synaptix.preprocessing import (
    detect_outliers, remove_outliers, train_test_split
)

mask = detect_outliers(df, method="iqr", threshold=1.5)
print(mask.sum())               # outliers por columna

df_limpio = remove_outliers(df, columns=["precio"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)`,
        codeTitle: "outliers.py",
      },
    ],
  },
  {
    slug: "supervised",
    name: "synaptix.supervised",
    title: "Aprendizaje supervisado",
    intro:
      "Quince modelos de regresión y clasificación que comparten exactamente la misma API: fit, predict, evaluate, summary, save y load. Aceptan DataFrames de pandas, arrays de NumPy o listas.",
    sections: [
      {
        heading: "Clasificación",
        body: "Siete clasificadores: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, SVMClassifier, KNNClassifier y NaiveBayes. evaluate() imprime accuracy, precision, recall y F1; con plot=True muestra la matriz de confusión.",
        code: `import synaptix as sx
from synaptix.supervised import RandomForestClassifier
from synaptix.preprocessing import train_test_split

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
model.evaluate(X_test, y_test, plot=True)
# === Evaluación: RandomForestClassifier ===
#   accuracy    : 0.9667
#   precision   : 0.9697
#   recall      : 0.9667
#   f1          : 0.9666`,
        codeTitle: "clasificacion.py",
      },
      {
        heading: "Regresión",
        body: "Ocho regresores: LinearRegression, RidgeRegression, LassoRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, SVR y KNNRegressor. evaluate() reporta MAE, MSE, RMSE, R² y MAPE.",
        code: `from synaptix.supervised import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)
resultados = model.evaluate(X_test, y_test, plot=True)
print(resultados["R2"])`,
        codeTitle: "regresion.py",
      },
      {
        heading: "El descenso de gradiente, animado",
        body: "Modelos como GradientBoosting y las redes neuronales optimizan sus parámetros siguiendo el gradiente de la función de pérdida. Ajusta el learning rate y observa: muy bajo converge lento, muy alto rebota.",
        viz: "gradient",
      },
      {
        heading: "Inspección y persistencia",
        body: "feature_importances() devuelve una Serie ordenada (árboles y modelos lineales). save/load usan pickle.",
        code: `print(model.feature_importances())
model.summary()

model.save("modelo.pkl")
modelo_cargado = RandomForestClassifier.load("modelo.pkl")`,
        codeTitle: "persistencia.py",
      },
    ],
  },
  {
    slug: "unsupervised",
    name: "synaptix.unsupervised",
    title: "Aprendizaje no supervisado",
    intro:
      "Clustering (KMeans, DBSCAN, jerárquico, mezcla de gaussianas) con selección de k integrada, y reducción de dimensionalidad (PCA, t-SNE) con gráficos de varianza explicada.",
    sections: [
      {
        heading: "K-Means en acción",
        body: "El algoritmo alterna dos pasos: asignar cada punto a su centroide más cercano y recalcular los centroides como el promedio de sus puntos. Presiona paso o auto para verlo converger.",
        viz: "kmeans",
      },
      {
        heading: "Elegir k y agrupar",
        body: "KMeans.elbow grafica la inercia para cada k (busca el codo); silhouette_scores mide qué tan bien separados están los clusters (mayor es mejor).",
        code: `import synaptix as sx
from synaptix.unsupervised import KMeans

df = sx.load_dataset("iris")
X = df.drop(columns="species")

KMeans.elbow(X, k_max=10)              # método del codo
KMeans.silhouette_scores(X, k_max=8)   # análisis de silhouette

km = KMeans(n_clusters=3)
labels = km.fit_predict(X)
km.plot(X)                             # scatter 2D por cluster
print(km.silhouette(X))                # 0.55
print(km.centroids_)`,
        codeTitle: "kmeans.py",
      },
      {
        heading: "Otros algoritmos de clustering",
        body: "DBSCAN agrupa por densidad y marca el ruido con -1 (no requiere elegir k). HierarchicalClustering construye un dendrograma aglomerativo. GaussianMixture es probabilístico: predict_proba da la pertenencia de cada punto.",
        code: `from synaptix.unsupervised import DBSCAN, GaussianMixture

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)             # -1 = ruido

gmm = GaussianMixture(n_components=3)
gmm.fit(X)
probabilidades = gmm.predict_proba(X)  # (n, 3)`,
        codeTitle: "clustering.py",
      },
      {
        heading: "Reducción de dimensionalidad",
        body: "PCA proyecta a menos dimensiones maximizando varianza (acepta n_components como entero o fracción de varianza, ej. 0.95). TSNE es ideal para visualizar datos de alta dimensión en 2D.",
        code: `from synaptix.unsupervised import PCA, TSNE

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
pca.plot_variance()                    # varianza explicada acumulada

TSNE(n_components=2).plot(X, labels=y) # proyección coloreada`,
        codeTitle: "reduccion.py",
      },
    ],
  },
  {
    slug: "reinforcement",
    name: "synaptix.reinforcement",
    title: "Aprendizaje por refuerzo",
    intro:
      "Agentes Q-Learning y SARSA tabulares implementados desde cero con NumPy, un entorno GridWorld para experimentar, y DQN con Keras para espacios de estados continuos.",
    sections: [
      {
        heading: "Un agente aprendiendo, en vivo",
        body: "Este es el mismo algoritmo que implementa QLearningAgent: explora con probabilidad ε (que decae por episodio), actualiza su tabla Q con cada transición y converge a la política óptima. Las flechas muestran la mejor acción aprendida por celda.",
        viz: "gridworld",
      },
      {
        heading: "Entrenar un agente",
        body: "GridWorld sigue la interfaz reset() / step(action) estilo Gymnasium. El agente recibe +10 al llegar a la meta, -1 por paso y -5 por chocar.",
        code: `from synaptix.reinforcement import GridWorld, QLearningAgent

env = GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])

agent = QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,        # tasa de aprendizaje
    gamma=0.99,       # descuento de recompensas futuras
    epsilon=1.0,      # exploración inicial
)

rewards = agent.train(env, episodes=500, verbose=True)
agent.plot_rewards()          # curva de recompensa por episodio
env.render(agent.policy())    # política aprendida con flechas`,
        codeTitle: "qlearning.py",
      },
      {
        heading: "SARSA y DQN",
        body: "SARSAAgent es on-policy: aprende de la acción que realmente toma (más conservador cerca de zonas peligrosas). DQNAgent reemplaza la tabla por una red neuronal con experience replay y red objetivo — para estados continuos. Requiere pip install synaptix[dl].",
        code: `from synaptix.reinforcement import SARSAAgent, DQNAgent

sarsa = SARSAAgent(env.n_states, env.n_actions)
sarsa.train(env, episodes=500)

# Para entornos con estado vectorial (ej. CartPole):
dqn = DQNAgent(state_dim=4, n_actions=2, hidden_layers=(64, 64))
dqn.train(mi_entorno, episodes=200)`,
        codeTitle: "sarsa_dqn.py",
      },
    ],
  },
  {
    slug: "neural",
    name: "synaptix.neural",
    title: "Redes neuronales",
    intro:
      "Builders de alto nivel sobre Keras: MLP para tabulares, CNN para imágenes y LSTMNet para series de tiempo. Detección automática de la tarea, escalado interno, early stopping y curvas de entrenamiento. Requiere pip install synaptix[dl].",
    sections: [
      {
        heading: "MLP: perceptrón multicapa",
        body: "Detecta automáticamente si la clasificación es binaria (sigmoid) o multiclase (softmax), escala las features internamente y aplica early stopping. Las etiquetas pueden ser strings: se codifican y decodifican solas.",
        code: `from synaptix.neural import MLP

net = MLP(
    task="classification",
    hidden_layers=(64, 32),
    dropout=0.2,
)
net.fit(X_train, y_train, epochs=100)   # early stopping incluido
net.evaluate(X_test, y_test)
net.plot_history()                      # curvas de pérdida y accuracy

predicciones = net.predict(X_test)      # etiquetas originales
probabilidades = net.predict_proba(X_test)`,
        codeTitle: "mlp.py",
      },
      {
        heading: "CNN: clasificación de imágenes",
        body: "Bloques Conv2D + MaxPooling configurables. Espera imágenes con forma (n, alto, ancho, canales).",
        code: `from synaptix.neural import CNN

net = CNN(
    input_shape=(28, 28, 1),
    n_classes=10,
    conv_blocks=(32, 64),      # filtros por bloque
    dense_units=128,
)
net.fit(X_train, y_train, epochs=10)
clases = net.predict(X_test)`,
        codeTitle: "cnn.py",
      },
      {
        heading: "LSTM: series de tiempo",
        body: "Convierte automáticamente una serie 1D en ventanas deslizantes, escala a [0,1] y entrena una LSTM. forecast() predice pasos futuros de forma autorregresiva y devuelve valores en la escala original.",
        code: `from synaptix.neural import LSTMNet

lstm = LSTMNet(window=12, units=(64,))
lstm.fit(serie_mensual, epochs=50)

futuro = lstm.forecast(serie_mensual, steps=6)  # 6 meses adelante
lstm.plot_history()`,
        codeTitle: "lstm.py",
      },
      {
        heading: "Cómo optimiza una red",
        body: "Cada época, el optimizador mueve los pesos en dirección contraria al gradiente de la pérdida. Experimenta con el learning rate:",
        viz: "gradient",
      },
    ],
  },
  {
    slug: "metrics",
    name: "synaptix.metrics",
    title: "Métricas de evaluación",
    intro:
      "Funciones unificadas que devuelven diccionarios de métricas para regresión, clasificación y clustering, más un reporte de clasificación completo con matriz de confusión.",
    sections: [
      {
        heading: "Por tipo de problema",
        code: `from synaptix.metrics import (
    regression_metrics,
    classification_metrics,
    clustering_metrics,
)

regression_metrics(y_true, y_pred)
# {"MAE": ..., "MSE": ..., "RMSE": ..., "R2": ..., "MAPE": ...}

classification_metrics(y_true, y_pred)
# {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}

clustering_metrics(X, labels)
# {"silhouette": ..., "calinski_harabasz": ..., "davies_bouldin": ...}`,
        codeTitle: "metricas.py",
      },
      {
        heading: "Reporte de clasificación completo",
        body: "Imprime métricas globales, el desglose por clase de scikit-learn y, con plot=True, la matriz de confusión. Devuelve todo como diccionario para uso programático.",
        code: `from synaptix.metrics import classification_report

reporte = classification_report(
    y_test, y_pred,
    plot=True,
    labels=["setosa", "versicolor", "virginica"],
)
print(reporte["por_clase"]["setosa"]["f1-score"])`,
        codeTitle: "reporte.py",
      },
    ],
  },
  {
    slug: "model-selection",
    name: "synaptix.model_selection",
    title: "Validación y selección de modelos",
    intro:
      "Validación cruzada, búsqueda de hiperparámetros en rejilla y compare_models: un AutoML-lite que entrena todos los modelos disponibles y devuelve un ranking.",
    sections: [
      {
        heading: "compare_models: ¿cuál es el mejor modelo?",
        body: "Entrena 7 clasificadores (o 7 regresores) con validación cruzada y devuelve un DataFrame ordenado por desempeño, con desviación estándar y tiempo de entrenamiento.",
        code: `from synaptix.model_selection import compare_models

tabla = compare_models(X, y, task="classification", cv=5)
#                       modelo  accuracy  desviacion  tiempo_s
#     RandomForestClassifier      0.9667      0.0211     0.512
#              SVMClassifier      0.9667      0.0298     0.023
#         LogisticRegression      0.9600      0.0327     0.041
#                        ...`,
        codeTitle: "comparar.py",
      },
      {
        heading: "Validación cruzada",
        code: `from synaptix.model_selection import cross_validate
from synaptix.supervised import RandomForestClassifier

cv = cross_validate(RandomForestClassifier(), X, y, cv=5)
print(cv["media"], "+/-", cv["desviacion"])`,
        codeTitle: "cross_val.py",
      },
      {
        heading: "GridSearch: ajuste de hiperparámetros",
        body: "Prueba todas las combinaciones de la rejilla con validación cruzada en paralelo y devuelve el modelo re-entrenado con la mejor.",
        code: `from synaptix.model_selection import GridSearch
from synaptix.supervised import RandomForestClassifier

search = GridSearch(
    RandomForestClassifier(),
    {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]},
    cv=5,
)
mejor_modelo = search.fit(X, y)
print(search.best_params_)   # {"max_depth": None, "n_estimators": 200}
print(search.best_score_)    # 0.9667`,
        codeTitle: "gridsearch.py",
      },
    ],
  },
  {
    slug: "datasets",
    name: "synaptix.datasets",
    title: "Datasets incluidos",
    intro:
      "Cinco datasets listos para experimentar, empaquetados dentro de la librería. Se cargan como DataFrames de pandas sin descargar nada.",
    sections: [
      {
        heading: "Cargar datasets",
        code: `import synaptix as sx

sx.list_datasets()
# ["iris", "penguins", "titanic", "sp500_companies", "course_completion"]

df = sx.load_dataset("iris")        # con o sin extensión .csv
df.head()`,
        codeTitle: "datasets.py",
      },
      {
        heading: "Catálogo",
        body: "iris: 150 flores, 4 features, 3 especies — el clásico para clasificación. penguins: pingüinos de Palmer, ideal para clasificación con nulos reales. titanic: supervivencia de pasajeros, clasificación binaria con categóricas. sp500_companies: datos financieros de empresas del S&P 500, útil para regresión y clustering. course_completion: dataset grande de finalización de cursos en línea, para pipelines más realistas.",
      },
      {
        heading: "Gráficas a partir de datos (EDA)",
        body: "El módulo synaptix.visualization incluye gráficas de análisis exploratorio para entender un DataFrame antes de modelar: distribuciones con media y mediana, heatmap de correlación con valores anotados, boxplots segmentables por categoría, matriz de dispersión y porcentaje de nulos.",
        code: `from synaptix.visualization import (
    plot_distributions,   # histogramas de todas las numéricas
    plot_correlation,     # heatmap de correlación
    plot_boxplots,        # outliers, segmentables con by=
    plot_scatter_matrix,  # pares de variables, color por clase
    plot_missing,         # % de nulos por columna
)

df = sx.load_dataset("penguins")
plot_missing(df)
plot_distributions(df)
plot_correlation(df, method="spearman")
plot_boxplots(df, by="species")
plot_scatter_matrix(df, target="species")`,
        codeTitle: "eda.py",
      },
      {
        heading: "Gráficas a partir de modelos",
        body: "Diagnóstico visual de modelos entrenados: matriz de confusión, curva ROC, curva de aprendizaje, importancia de features, análisis de residuos y frontera de decisión 2D. model_report() genera el reporte completo según la tarea en una sola llamada.",
        code: `from synaptix.visualization import (
    plot_confusion_matrix,    # clasificación
    plot_roc_curve,           # binaria, con AUC
    plot_learning_curve,      # diagnóstico de overfitting
    plot_feature_importance,  # árboles y lineales
    plot_residuals,           # regresión: residuos vs predicho
    plot_decision_boundary,   # regiones de decisión en 2D
    model_report,             # todo lo anterior, automático
)

plot_decision_boundary(model, X, y, features=(0, 2))
plot_residuals(modelo_regresion, X_test, y_test)

model_report(model, X_test, y_test)   # métricas + gráficos`,
        codeTitle: "model_plots.py",
      },
    ],
  },
];

export function getModule(slug: string): DocModule | undefined {
  return MODULES.find((m) => m.slug === slug);
}
