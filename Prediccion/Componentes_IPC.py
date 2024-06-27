import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Descargar datos del IPC
ipc_data = yf.download('^MXX', start='2000-01-01', end='2024-03-15')

# Lista de los componentes del IPC (ejemplo)
componentes_ipc = [
    'WALMEX.MX', 'AMXL.MX', 'BIMBOA.MX', 'CEMEXCPO.MX', 'GMEXICOB.MX',
    'FEMSAUBD.MX', 'PE&OLES.MX', 'TLEVICPO.MX', 'GFINBURO.MX', 'ELEKTRA.MX',
    'AC.MX', 'ALFAA.MX', 'ALSEA.MX', 'BBAJIOO.MX', 'BOLSAA.MX', 'GCC.MX',
    'GENTERA.MX', 'GFNORTEO.MX', 'GCARSOA1.MX', 'GRUMAB.MX', 'KIMBERA.MX',
    'LABB.MX', 'LIVEPOLC-1.MX', 'MEGACPO.MX', 'MEXCHEM.MX', 'OMAB.MX', 'PINFRA.MX',
    'RA.MX', 'SANMEXB.MX', 'TLEVISACPO.MX', 'VASCONI.MX', 'VITROA.MX'
]


# Diccionario para almacenar los precios de cierre
close_prices = {}


# Descargar los precios de cierre de los componentes del IPC
for componente in componentes_ipc:
    data = yf.download(componente, start='2000-01-01', end='2024-03-15')
    close_prices[componente] = data['Close']

# Combinar los precios de cierre en un DataFrame
close_prices_df = pd.DataFrame(close_prices)

# Guardar los datos en un archivo CSV (opcional)
close_prices_df.to_csv('ipc_close_prices.csv')

# Mostrar un resumen de los datos
print(close_prices_df.head())

# Mostrar la proporción de valores faltantes en el periodo
print(close_prices_df.isna().sum() /len(close_prices_df))

# Graficar los precios de cierre
plt.figure(figsize=(14, 7))
for column in close_prices_df.columns:
    plt.plot(close_prices_df.index, close_prices_df[column], label=column)

plt.title('Precios de Cierre de los Componentes del IPC')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.show()
