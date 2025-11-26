from statslibx import DescriptiveStats, InferentialStats, UtilsStats
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Usuario\Documents\Emmanuel Ascendra\Emmanuel Ascendra Personal\Ciencia de Datos\Talento Tech Analisis de Datos\Proyecto Talento Tech 2\estudiantes (1).csv")

stats = DescriptiveStats(data)
stats.help()
#print(stats.summary())

#print(stats.linear_regression(y="nota", X="horas_estudio", engine="statsmodels", show_plot=False).summary())

#infer = InferentialStats(data)

#print(infer.normality_test(column="nota"))
#print(infer.confidence_interval(column="nota", confidence=0.95))

#guardar la imagen
# Primero creas una instancia de la clase
#utils = UtilsStats()

# Luego usas los métodos de la instancia
#utils.plot_distribution(data=data, column="nota", plot_type="all", backend="plotly")
#plt.show()

# hacer un apredicion de las notas con respecto a horas_estudio

# También puedes configurar preferencias
#utils.set_plot_backend('seaborn')
#utils.set_default_figsize((12, 8))

# Y luego usar otros métodos
"""stats = DescriptiveStats(data)
#print(stats.summary())

print(stats.linear_regression(y="nota", X="horas_estudio", engine="scikit-learn", show_plot=True))

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Elimina filas donde haya NaN en horas_estudio o nota
data_clean = data.dropna(subset=["horas_estudio", "nota"])

X = data_clean["horas_estudio"].values.reshape(-1, 1)
y = data_clean["nota"].values.reshape(-1, 1)

model = LinearRegression().fit(X, y)
predictions = model.predict(X)

utils.plot_regression(X, y, predictions)
plt.show()"""



#utils.set_save_fig_options(save_fig=True, fig_format='pdf', fig_dpi=300)
#utils.plot_distribution(data=data, column="nota", plot_type="all", filename="distribucion_nota2", save_fig=True)