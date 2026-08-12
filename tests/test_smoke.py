"""
Test de humo de SynaptIX: ejercita cada submódulo con datos reales.

Ejecutar desde la raíz del repo:
    python tests/test_smoke.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # sin ventanas de gráficos

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

PASSED = []
FAILED = []


def check(name):
    def decorator(func):
        def wrapper():
            try:
                func()
                PASSED.append(name)
                print(f"[OK]    {name}")
            except Exception as error:
                FAILED.append((name, error))
                print(f"[FALLO] {name}: {error}")

        return wrapper

    return decorator


@check("import synaptix (sin tensorflow)")
def test_import():
    import synaptix as sx

    assert sx.__version__ == "0.1.6"
    assert hasattr(sx, "supervised")
    assert hasattr(sx, "preprocessing")


@check("datasets: load_dataset / list_datasets")
def test_datasets():
    import synaptix as sx

    names = sx.list_datasets()
    assert "iris" in names
    df = sx.load_dataset("iris")  # sin extensión
    assert len(df) > 100
    df2 = sx.load_dataset("iris.csv")  # con extensión
    assert df.shape == df2.shape


@check("preprocessing: Scaler, Encoder, Imputer, outliers, DataCleaner")
def test_preprocessing():
    from synaptix.preprocessing import (
        DataCleaner,
        Encoder,
        Imputer,
        Scaler,
        detect_outliers,
        remove_outliers,
        train_test_split,
    )

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(5, 2, 100),
            "cat": rng.choice(["x", "y", "z"], 100),
        }
    )
    df.loc[0:5, "a"] = np.nan
    df.loc[10, "b"] = 100.0  # outlier

    scaled = Scaler("standard").fit_transform(df[["b"]])
    assert abs(scaled["b"].mean()) < 1e-6

    encoded = Encoder("onehot").fit_transform(df, columns=["cat"])
    assert "cat_x" in encoded.columns

    imputed = Imputer("median").fit_transform(df)
    assert imputed["a"].isnull().sum() == 0

    mask = detect_outliers(imputed, columns=["b"])
    assert mask["b"].sum() >= 1
    cleaned_rows = remove_outliers(imputed, columns=["b"])
    assert len(cleaned_rows) < len(imputed)

    cleaner = DataCleaner()
    report = cleaner.analyze(df, verbose=False)
    assert report["nulos"]["a"] == 6
    ready = cleaner.clean(df)
    assert ready.isnull().sum().sum() == 0

    X_train, X_test, y_train, y_test = train_test_split(
        df[["b"]].fillna(0), df["cat"], test_size=0.2, random_state=42
    )
    assert len(X_train) == 80


@check("supervised: clasificación (iris) con 7 modelos")
def test_classification():
    import synaptix as sx
    from synaptix import supervised
    from synaptix.preprocessing import train_test_split

    df = sx.load_dataset("iris")
    X = df.select_dtypes(include=[np.number])
    y = df.iloc[:, -1] if df.iloc[:, -1].dtype == object else df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = [
        supervised.LogisticRegression(),
        supervised.DecisionTreeClassifier(max_depth=4),
        supervised.RandomForestClassifier(n_estimators=30),
        supervised.GradientBoostingClassifier(n_estimators=30),
        supervised.SVMClassifier(),
        supervised.KNNClassifier(),
        supervised.NaiveBayes(),
    ]
    for model in models:
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test, verbose=False)
        assert results["accuracy"] > 0.7, f"{model.name}: {results['accuracy']}"

    # persistencia
    rf = models[2]
    rf.save("_tmp_model.pkl")
    loaded = type(rf).load("_tmp_model.pkl")
    assert (loaded.predict(X_test) == rf.predict(X_test)).all()
    os.remove("_tmp_model.pkl")

    # importancia de features
    imp = rf.feature_importances()
    assert imp is not None and len(imp) == X.shape[1]


@check("supervised: regresión sintética con 8 modelos")
def test_regression():
    from synaptix import supervised
    from synaptix.preprocessing import train_test_split

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 0.3, 300)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = [
        supervised.LinearRegression(),
        supervised.RidgeRegression(),
        supervised.LassoRegression(alpha=0.01),
        supervised.DecisionTreeRegressor(max_depth=6),
        supervised.RandomForestRegressor(n_estimators=30),
        supervised.GradientBoostingRegressor(n_estimators=30),
        supervised.SVR(),
        supervised.KNNRegressor(),
    ]
    for model in models:
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test, verbose=False)
        assert results["R2"] > 0.5, f"{model.name}: {results['R2']}"


@check("unsupervised: KMeans, DBSCAN, jerárquico, GMM, PCA, TSNE")
def test_unsupervised():
    import synaptix as sx
    from synaptix.unsupervised import (
        DBSCAN,
        GaussianMixture,
        HierarchicalClustering,
        KMeans,
        PCA,
        TSNE,
    )

    df = sx.load_dataset("iris")
    X = df.select_dtypes(include=[np.number])

    km = KMeans(n_clusters=3)
    labels = km.fit_predict(X)
    assert len(np.unique(labels)) == 3
    assert km.silhouette(X) > 0.3
    assert km.centroids_.shape[0] == 3

    inertias = KMeans.elbow(X, k_max=5, plot=False)
    assert inertias[1] > inertias[5]

    scores = KMeans.silhouette_scores(X, k_max=4, plot=False)
    assert all(0 <= s <= 1 for s in scores.values())

    assert DBSCAN(eps=0.8, min_samples=5).fit_predict(X) is not None
    assert len(np.unique(HierarchicalClustering(n_clusters=3).fit_predict(X))) == 3
    assert len(np.unique(GaussianMixture(n_components=3).fit_predict(X))) <= 3

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    assert X_2d.shape == (len(X), 2)
    assert pca.explained_variance().sum() > 0.9

    X_tsne = TSNE(n_components=2).fit_transform(X.iloc[:50])
    assert X_tsne.shape == (50, 2)


@check("reinforcement: GridWorld + QLearning + SARSA")
def test_reinforcement():
    from synaptix.reinforcement import GridWorld, QLearningAgent, SARSAAgent

    env = GridWorld(rows=4, cols=4, obstacles=[(1, 1)], max_steps=100)

    q_agent = QLearningAgent(env.n_states, env.n_actions, seed=42)
    rewards = q_agent.train(env, episodes=300)
    assert len(rewards) == 300
    # tras entrenar, la política greedy debe llegar a la meta
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        state, _, done = env.step(q_agent.act(state, greedy=True))
        steps += 1
    assert env.position_of(state) == env.goal, "Q-Learning no llegó a la meta"

    sarsa = SARSAAgent(env.n_states, env.n_actions, seed=42)
    sarsa.train(env, episodes=200)
    assert sarsa.q_table.shape == (env.n_states, env.n_actions)
    assert len(sarsa.policy()) == env.n_states


@check("metrics: regresión, clasificación, clustering")
def test_metrics():
    from synaptix.metrics import (
        classification_metrics,
        clustering_metrics,
        regression_metrics,
    )

    reg = regression_metrics([1, 2, 3, 4], [1.1, 2.1, 2.9, 4.2])
    assert reg["R2"] > 0.9 and "RMSE" in reg

    clf = classification_metrics(["a", "b", "a", "b"], ["a", "b", "b", "b"])
    assert clf["accuracy"] == 0.75

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(5, 0.5, (50, 2))])
    labels = np.array([0] * 50 + [1] * 50)
    clu = clustering_metrics(X, labels)
    assert clu["silhouette"] > 0.5


@check("model_selection: cross_validate, GridSearch, compare_models")
def test_model_selection():
    import synaptix as sx
    from synaptix.model_selection import GridSearch, compare_models, cross_validate
    from synaptix.supervised import DecisionTreeClassifier, RandomForestClassifier

    df = sx.load_dataset("iris")
    X = df.select_dtypes(include=[np.number])
    y = df.iloc[:, -1]

    cv = cross_validate(RandomForestClassifier(n_estimators=20), X, y, cv=3, verbose=False)
    assert cv["media"] > 0.8

    search = GridSearch(
        DecisionTreeClassifier(), {"max_depth": [2, 4]}, cv=3
    )
    best = search.fit(X, y, verbose=False)
    assert best.fitted and search.best_params_["max_depth"] in (2, 4)

    table = compare_models(X, y, task="classification", cv=3, verbose=False)
    assert len(table) == 7 and table["accuracy"].max() > 0.8


@check("visualization: generación de gráficos (backend Agg)")
def test_visualization():
    import synaptix as sx
    from synaptix.supervised import RandomForestClassifier
    from synaptix.visualization import (
        plot_clusters,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_predictions,
        plot_roc_curve,
    )

    df = sx.load_dataset("iris")
    X = df.select_dtypes(include=[np.number])
    y = df.iloc[:, -1]

    model = RandomForestClassifier(n_estimators=20).fit(X, y)
    y_pred = model.predict(X)

    plot_confusion_matrix(y, y_pred)
    plot_feature_importance(model)
    plot_clusters(X, (y == y.iloc[0]).astype(int).values)
    plot_predictions([1, 2, 3], [1.1, 2.2, 2.8])

    y_bin = (y == y.iloc[0]).astype(int)
    model_bin = RandomForestClassifier(n_estimators=20).fit(X, y_bin)
    plot_roc_curve(y_bin, model_bin.predict_proba(X)[:, 1])


@check("visualization: EDA y gráficas de modelos (nuevas en 0.1.6)")
def test_visualization_extra():
    import synaptix as sx
    from synaptix.supervised import LinearRegression, RandomForestClassifier
    from synaptix.visualization import (
        model_report,
        plot_boxplots,
        plot_correlation,
        plot_decision_boundary,
        plot_distributions,
        plot_missing,
        plot_residuals,
        plot_scatter_matrix,
    )

    df = sx.load_dataset("penguins")
    num = df.select_dtypes(include=[np.number])

    # EDA a partir de datos
    plot_distributions(df)
    plot_correlation(df)
    plot_boxplots(df, by="species" if "species" in df.columns else None)
    plot_scatter_matrix(df, max_cols=4)
    plot_missing(df)

    # Gráficas a partir de modelos
    iris = sx.load_dataset("iris")
    X_iris = iris.select_dtypes(include=[np.number])
    y_iris = iris.iloc[:, -1]
    clf = RandomForestClassifier(n_estimators=15).fit(X_iris, y_iris)
    plot_decision_boundary(clf, X_iris, y_iris, features=(0, 2), resolution=60)

    rng = np.random.default_rng(0)
    X_reg = rng.normal(0, 1, (150, 3))
    y_reg = 2 * X_reg[:, 0] + rng.normal(0, 0.2, 150)
    reg = LinearRegression().fit(X_reg, y_reg)
    plot_residuals(reg, X_reg, y_reg)

    resultados = model_report(reg, X_reg, y_reg)
    assert resultados["R2"] > 0.9


@check("legacy: imports y compatibilidad")
def test_legacy():
    from synaptix import (
        DeepLearning,
        IntelligenceArtificial,
        NaturalLanguageProcessing,
        NaturalLanguajeProcessing,
    )

    assert NaturalLanguajeProcessing is NaturalLanguageProcessing

    ia = IntelligenceArtificial(backend_ia=("mistral", ""))
    ia.add_fact("perro", "es", "animal")
    assert not ia.api_key or "OPENROUTER" not in ia.api_key  # sin key hardcodeada

    dl = DeepLearning([1, 2, 3])
    assert len(dl.data) == 3


@check("neural: MLP y LSTM (solo si tensorflow está instalado)")
def test_neural():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("        (tensorflow no instalado; se omite)")
        return

    import synaptix as sx
    from synaptix.neural import LSTMNet, MLP
    from synaptix.preprocessing import train_test_split

    df = sx.load_dataset("iris")
    X = df.select_dtypes(include=[np.number])
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    net = MLP(task="classification", hidden_layers=(16,))
    net.fit(X_train, y_train, epochs=30, verbose=0)
    results = net.evaluate(X_test, y_test, verbose=False)
    assert results["accuracy"] > 0.6

    serie = np.sin(np.linspace(0, 20, 200))
    lstm = LSTMNet(window=10, units=(16,))
    lstm.fit(serie, epochs=5, verbose=0)
    forecast = lstm.forecast(serie, steps=3)
    assert forecast.shape == (3,)


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()

    print(f"\n{'=' * 50}")
    print(f"Pasaron: {len(PASSED)} | Fallaron: {len(FAILED)}")
    if FAILED:
        for name, error in FAILED:
            print(f"  FALLO {name}: {error}")
        sys.exit(1)
    print("Todos los tests de humo pasaron.")
