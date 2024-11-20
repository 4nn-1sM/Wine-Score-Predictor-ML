import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
import unidecode

# Cargamos el dataset principal con el que vamos a trabajar. data/raw/winemag-data-2017-2020.csv
df = pd.read_csv("../data/raw/winemag-data-2017-2020.csv")

# Eliminamos filas que tengan duplicado el valor de la columna title, ya que serían los mismos vinos. Después reordenamos el dataset para que tenga más sentido:
df = df.sort_values(by="points", ascending=False).drop_duplicates(subset="title", keep="first").reset_index(drop=True)
df = df[['title', 'vintage', 'winery', 'variety', 'country', 'description', 'designation', 'points', 'price', 'province',
       'region_1', 'region_2', 'taster_name', 'taster_photo', 'taster_twitter_handle']]

# Filtramos y guardamos en nueva variable el dataframe con los datos de los vinos producidos en España
df_esp = df[df["country"] == "Spain"].copy()

# Eliminamos todas las columnas con 0 valores o 1 valor, en este caso. region_2, country, taster_name, taster_photo y taster_twitter_handle, ya que no as vamos a utilizar
# Creamos una nueva columna denominada región. En ella se recogerán  los datos de pertenencia. Si no tiene ninguna región, 
# se guardará la provincia de proveniencia y eliminamos la de region_2 que está vacía.

df_esp = df_esp.drop(columns = df_esp.columns[(df_esp.nunique() <= 1)], axis=1)
df_esp["region"] = np.where(df_esp["region_1"].isna() == True, df_esp["province"], df_esp["region_1"])

# Creamos un diccionario que recoja las variedades de vino que pueden categorizarse según el estilo de vino en tintos, blancos, rosados, espumosos y generosos
style_cat = {
    "Red wine": [
        "Bordeaux-style Red Blend", "Red Blends", "Nebbiolo", "Portuguese Red", "Sangiovese", 
        "Gamay", "Pinot Noir", "Rhône-style Red Blend", "Tempranillo", "Tempranillo Blend", 
        "Aglianico", "Montepulciano", "Cabernet Franc", "Merlot", "Barbera", "Garnacha", 
        "Touriga Nacional", "Nero d'Avola", "Grenache-Syrah", "Cabernet Sauvignon", "Primitivo", 
        "Pinot Nero", "Tinta de Toro", "Malbec-Merlot", "Sagrantino", "Encruzado", "Tannat", 
        "Negroamaro", "Petit Verdot", "Syrah-Grenache", "Tempranillo-Garnacha", "Bobal", 
        "Tempranillo-Cabernet Sauvignon", "Baga", "Carignan", "Nero di Troia", "G-S-M", 
        "Monastrell-Syrah", "Graciano", "Tinta del Toro", "Monastrell",
        "Lagrein", "Garnacha-Syrah", "Tinta Roriz", "Grenache-Carignan", "Syrah-Viognier", 
        "Cabernet Sauvignon-Merlot", "Carignan-Grenache", "Carignano", "Nerello Cappuccio", 
        "Tinta del Pais", "Garnacha Tintorera", "Malbec-Tannat", "Cariñena-Garnacha", "Aragonez", 
        "Perricone", "Cesanese", "Lambrusco Salamino", "Tinta Negra Mole", "Braucol", 
        "Alfrocheiro", "Garnacha-Tempranillo", "Susumaniello", "Negrette", "Monastrell-Petit Verdot", 
        "Abouriou", "Touriga Nacional-Cabernet Sauvignon", "Garnacha Blend", "Tannat-Merlot", 
        "Merlot-Tannat", "Fer Servadou", "Syrah-Mourvèdre", "Grenache Blend", "Mourvèdre-Syrah", 
        "Merlot-Malbec", "Prieto Picudo", "Pallagrello Nero", "Merlot-Cabernet", "Merlot-Syrah", 
        "Syrah-Primitivo", "Tempranillo-Graciano", "Tannat-Cabernet Sauvignon", "Pugnitello", 
        "Cabernet Sauvignon-Cabernet Franc", "Rebo", "Bovale", "Magliocco", "Mataro", "Corvina", 
        "Nielluciu", "Tinta Barroca", "Manseng Noir", "Trousseau", "Merlot-Cabernet Franc", "Mourvèdre", "Gaglioppo", 
        "Syrah-Pinot Noir", "Tempranillo-Merlot", "Primitivo-Susumaniello", "Zinfandel", 
        "Bobal-Tempranillo", "Tannat-Malbec", "Fumin", "Grenache Noir", "Tempranillo-Shiraz", 
        "Touriga Nacional-Merlot", "Cabernet Sauvignon-Syrah", "Tinta Francisca", 
        "Touriga Nacional-Touriga Franca", "Touriga Nacional Blend", "Negroamaro-Malvasia", 
        "Syrah-Carignan", "Garnacha-Graciano", "Sirica", "Mazuelo", "Marzemino", 
        "Garnacha-Cariñena", "Cabernet Sauvignon Grenache", "Malbec Blend", "Cabernet", 
        "Monastrell-Garnacha Tintorera", "Primitivo-Cabernet Sauvignon", "Refosco", "Raboso", 
        "Minutolo", "Touriga Nacional-Syrah", "Malbec-Cabernet Franc", "Tinto Velasco", 
        "Syrah", "Malbec", "Nerello Mascalese", "Mencía", "Provence red blend", "Cannonau",
        "Dolcetto", "Alicante Bouschet", "Tannat-Cabernet", "Frappato", "Pinot Noir-Gamay",
        "Lambrusco Grasparossa", "Lambrusco", "Tempranillo Blanco", "Castelão", 
        "Tannat-Cabernet Franc", "Mondeuse", "Piedirosso", "Ciliegiolo", "Alicante", 
        "Touriga Franca", "Garnatxa", "Vinhão", "Airen", "Ramisco", "Cabernet Franc-Cabernet Sauvignon", 
        "Prunelard", "Tempranillo-Syrah", "Monica", "Sousão", "Tinta Fina", 
        "Monastrell-Cabernet Sauvignon", "Shiraz", "Ekigaïna", "Duras", "Cabernet Blend", 
        "Cabernet Sauvignon-Tempranillo", "Corinto Nero", "Nasco", "Foglia Tonda", "Traminer", 
        "Touriga Nacional-Shiraz", "Sciaccerelli", "Casavecchia", "Uvalino", "Tinto del Pais", 
        "Marselan", "Sumoll", "Gragnano", "Molinara", "Viura-Tempranillo Blanco", "Tannat-Syrah", 
        "Colorino", "Syrah-Garnacha", "Camarlet", "Merlot-Cabernet Sauvignon", "Trincadeira", "Teroldego", "Touriga Nacional-Alicante Bouschet"
    ],
    "White wine": [
        "Chardonnay", "Portuguese White", "Sauvignon Blanc", "Bordeaux-style White Blend", 
        "White Blend", "Pinot Gris", "Albariño", "Pinot Blanc", "Sauvignon", "Chenin Blanc", 
        "Garganega", "Verdejo", "Melon", "Arinto", "Cortese", "Viura", "Friulano", "Verdelho", 
        "Xarel-lo", "Trebbiano", "Auxerrois", "Aligoté", "Catarratto", "Macabeo", "Gros Manseng", 
        "Petit Manseng", "Rolle", "Alvarinho", "Chenin Blanc-Chardonnay",
        "Pansa Blanca", "Passerina", "Muscat Ottonel", "Bical", "Erbaluce", "Grechetto", 
        "Verdejo-Viura", "Mauzac", "Caprettone", "Siria", "Chasselas", "Arneis", "Bual", 
        "Pallagrello Bianco", "Insolia", "Viura-Chardonnay", "Jaen", "Verdeca", "Avesso", 
        "Viosinho", "Inzolia", "Savagnin", "Altesse", "Grolleau", "Arinto-Chardonnay", 
        "Colombard-Ugni Blanc", "Chardonnay-Sauvignon Blanc", "Alvarinho-Chardonnay", 
        "Loureiro-Alvarinho", "Malvar", "Colombard", "Loin de l'Oeil", "Sercial", "Poulsard", 
        "Malvasia Nera", "Cerceal", "Catalanesca", "Moreto", "Moscatel Galego Branco", 
        "Tinta Miúda", "Sauvignon Blanc-Chardonnay", "Viura", "Grüner Veltliner", "Timorasso", 
        "Clairette", "Ondenc", "Macabeo-Sauvignon Blanc", "Mantonico", "Rufete", "Greco Bianco", 
        "Grenache-Tempranillo", "Chardonnay-Macabeo", "Sauvignon-Sémillon", "Sciaccerellu", 
        "Carignan-Syrah", "Cercial", "Pigato", "Verdosilla", "Dorona", "Roussanne-Marsanne", 
        "Chardonnay-Viognier", "Mansois", "Biancolella", "Cococciola", "Chasan", "Vespolina", 
        "Nosiola", "Malvasia di Candia", "Biancu Gentile", "Rabigato", "Grenache Gris", "Macabeo-Chardonnay", "Verdejo-Sauvignon Blanc", "Bombino Bianco", 
        "Viura-Sauvignon Blanc", "Albillo", "Sauvignon Blanc-Semillon", "Sauvignon Gris", 
        "Folle Blanche", "Garnatxa Blanca", "Edelzwicker", "Durello", "Roussanne-Grenache Blanc", 
        "Maria Gomes-Bical", "Sauvignon-Arinto", "Loureiro-Arinto", "Merseguera-Sauvignon Blanc", 
        "Bobal-Cabernet Sauvignon", "Gros Plant", "Verdil", "Verduzzo", "Petite Arvine", 
        "Viura-Verdejo", "Romorantin", "Doña Blanca", "Picolit", "Gouveio", "Petit Courbu", 
        "Malvasia del Lazio", "Incrocio Manzoni", "Avesso-Alvarinho", "Centesimino", 
        "Viognier-Chardonnay", "Aragonês", "Colombard-Gros Manseng", "Moscato Giallo", "Cencibel", 
        "Malvasia-Viura", "Trajadura", "Nascetta", "Ortega", "Trebbiano Spoletino", "Treixadura", 
        "Sauvignon Blanc-Verdejo", "Fiano-Chardonnay", "Ugni Blanc", "Fiano-Malvasia", 
        "Chardonnay-Arinto", "Diagalves", "Bianco d'Alessano", "Semillon-Sauvignon Blanc", 
        "Malvasia Bianca", "Riesling", "Rhône-style White Blend", "Gewürztraminer", "Pinot Grigio", "Turbiana", 
        "Fiano", "Vermentino", "Vernaccia", "Pinot Bianco", "Viognier", "Greco", "Falanghina", 
        "Verdicchio", "Grillo", "Alsace white blend", "Sylvaner", "Godello", "Melon de Bourgogne", 
        "Grenache", "Pecorino", "Malvasia", "Tinto Fino", "Gros and Petit Manseng", "Moscato", 
        "Provence white blend", "Durella", "Carricante", "Loureiro", "Garnacha Blanca", "Marsanne", 
        "Sémillon", "Ribolla Gialla", "Jacquère", "Kerner", "Pinot Meunier", "Fernão Pires", 
        "Colombard-Sauvignon Blanc", "Antão Vaz", "Roussanne", "Pinot Auxerrois", "Albana", 
        "Zibibbo", "Coda di Volpe", "Muscat Blanc à Petits Grains", "Müller-Thurgau", 
        "Alvarinho-Trajadura", "Roviello", "Grenache Blanc", "Favorita", "Samarrinho", "Listán Negro", 
        "Nuragus", "Alvarinho-Loureiro", "Cagnulari", "Marsanne-Roussanne", 
        "Viura-Tempranillo Blanco", "Azal", "Meseguera", "Maturana Blanca", "Chardonnay-Viura", 
        "Hondarrabi Zuri", "Muscadelle", "Manzoni"
    ],
    "Rosée": [
        "Rosé", "Rosato", "Portuguese Rosé", "Lambrusco di Sorbara", "Rosado", "Portuguese Rosé", "Gamay Noir", "Trepat", "Canaiolo", "Rosada", "Espadeiro", "Asprinio", "Pineau d'Aunis"
    ],
    "Sparkling": [
        "Sparkling Blend", "Champagne Blend", "Portuguese Sparkling", "Glera", 
        "Grenache-Mourvèdre", "Grenache-Mourvèdre", "Ugni Blanc-Colombard", "Arinto-Chardonnay", "Touriga Nacional-Cabernet Sauvignon", "Txakoli", "Chardonnay-Moscatel de Alejandría"
    ],
    "Fortified/ Dessert wine": [
        "Port", "Sherry", "Pedro Ximénez", "Palomino", "Moscatel", "Muscat", "Moscatel Galego Branco", "Sercial", "Sherry", "Colombard-Ugni Blanc", "Terrantez", "Moscato Rosa", "Muscat d'Alexandrie", "Moscatel Roxo", "Babosa Negro", 
        "Moscatel de Alejandría", "Muscatel", "Boal", "Sciaccerelli", "Jampal", "Carineña"
    ]
}


def clasificar_variedad(variedad):
    for estilo, lista_variedades in style_cat.items():
        if variedad in lista_variedades:
            return estilo
    return "Unknown" 

df_esp["style"] = df_esp["variety"].apply(clasificar_variedad)

# limpieza de datos. Eliminamos registros sin año y aplicamos máscara para eliminar vinos que puedan tener un error en el año.
#  Nos quedamos con aquellos que sean menores de 2023 (Estamos en 2024)
df_esp = df_esp[df_esp["vintage"] != "NV"].reset_index(drop=True)
df_esp["vintage"] = df_esp["vintage"].astype(int)
df_esp = df_esp[df_esp["vintage"] <= 2023].reset_index(drop=True)

# También eliminamos aquellos que no tengan precio
df_esp = df_esp.dropna(subset=["price"])

# Dado que de Fortified / Dessert wine hay muy poca muestra, voy a eliminarlos del dataset ya que no tiene mucho sentido utilizar tan pocos registros.
df_esp = df_esp[df_esp["style"] != "Fortified/ Dessert wine"].copy()

# También vamos a eliminar el vino más añejo ya que está mal codificado, es un fortified.
df_esp = df_esp.drop(index=1244).copy()


# Generamos una nueva columna que indique el tiempo de maduración del vino
keywords = ['Joven', 'Crianza', 'Reserva', 'Gran Reserva', 'Roble']
def check_keywords(text, keywords):
    if isinstance(text, str):  # Comprobar que el texto sea una cadena
        return [keyword for keyword in keywords if keyword in text]
    else:
        return []  # Devolver una lista vacía si el valor no es una cadena

# Aplicar la función y manejar valores no iterables
df_esp["top_designation"] = df_esp["designation"].apply(lambda x: check_keywords(x, keywords))

# Función para eliminar redundancias de "Reserva" y "Selección"
def remove_redundant_keywords(keyword_list):
    # Eliminar "Reserva" si hay términos más específicos que contienen "Reserva"
    if "Reserva" in keyword_list:
        keyword_list = [k for k in keyword_list if k != "Reserva" or all("Reserva" not in other or k == other for other in keyword_list)]
    return keyword_list

# Aplicar la función para limpiar las listas
df_esp["top_designation_cleaned"] = df_esp["top_designation"].apply(remove_redundant_keywords)
# Rellenar valores nulos y convertir a string
df_esp["top_designation_cleaned_2"] = df_esp["top_designation_cleaned"].fillna('Ns').astype(str)
# Eliminar corchetes y otros caracteres no deseados
df_esp["top_designation_cleaned_3"] = df_esp['top_designation_cleaned_2'].str.replace(r'\[|\]', '', regex=True)
# Dividir los valores por coma y expandirlos en columnas
split_columns = df_esp["top_designation_cleaned_3"].str.split(',', expand=True)
# Renombrar dinámicamente las columnas generadas
split_columns.columns = [f'aging_{i+1}' for i in range(split_columns.shape[1])]
# Eliminar las comillas y espacios en las columnas generadas
for col in split_columns.columns:
    split_columns[col] = split_columns[col].str.replace("'", "").str.strip()
# Reemplazar cadenas vacías (valores vacíos) por "No Aplica"
split_columns = split_columns.replace(r'^\s*$', "No Aplica", regex=True)
# Reemplazar valores None o NaN por "No Aplica" (si hay nulos adicionales)
split_columns = split_columns.fillna("No Aplica")
# Eliminar columnas originales relacionadas con "top"
df_esp = df_esp.drop(columns=[col for col in df_esp.columns if "top" in col])
# Concatenar las nuevas columnas con el DataFrame original
df_esp = pd.concat([df_esp, split_columns], axis=1)

# Sacamos la dop/igp de cada vino comparando listado con la región y cramos nueva columna
dop_igp = [
    "3 Riberas", "Abadía Retuerta", "Abona", "Alella", "Alicante", "Almansa", 
    "Altiplano de Sierra Nevada", "Arlanza", "Arribes", "Aylés", "Bailén", 
    "Bajo Aragón", "Barbanza e Iria", "Betanzos", "Bierzo", "Binissalem", 
    "Bolandin", "Bullas", "Cádiz", "Calatayud", "Calzadilla", "Campo de Borja", 
    "Campo de Cartagena", "Campo de la Guardia", "Cangas", "Cariñena", 
    "Casa del Blanco", "Castelló", "Castilla", "Castilla y León", "Catalunya", 
    "Cava", "Cebreros", "Chozas Carrascal", "Cigales", "Conca de Barberá", 
    "Condado de Huelva", "Córdoba", "Costa de Cantabria", "Costers del Segre", 
    "Cumbres del Guadalfeo", "Dehesa del Carrizal", "Dehesa Peñalba", 
    "Desierto de Almería", "Dominio de Valdepusa", "El Hierro", "El Terrerazo", 
    "El Vicario", "Empordà", "Extremadura", "Finca Élez", "Formentera", 
    "Gran Canaria", "Granada", "Guijoso", "Ibiza", "Illes Balears", 
    "Isla de Menorca", "Islas Canarias", "Jerez-Xérès-Sherry", "Jumilla", 
    "La Gomera", "La Jaraba", "La Mancha", "La Palma", "Laderas del Genil", 
    "Lanzarote", "Laujar-Alpujarra", "Lebrija", "León", "Liébana", 
    "Los Balagueses", "Los Cerrillos", "Los Palacios", "Málaga", "Mallorca", 
    "Manchuela", "Manzanilla-Sanlúcar de Barrameda", "Méntrida", "Mondéjar", 
    "Monterrei", "Montilla-Moriles", "Montsant", "Murcia", "Navarra", 
    "Norte de Almería", "Pago de Arínzano", "Pago de Otazu", "Pago Florentino", 
    "Penedés", "Pla de Bages", "Pla i Llevant", "Prado de Irache", "Priorat", 
    "Rías Baixas", "Ribeira Sacra", "Ribeiras do Morrazo", "Ribeiro", 
    "Ribera del Andarax", "Ribera del Duero", "Ribera del Gállego-Cinco Villas", 
    "Ribera del Guadiana", "Ribera del Jiloca", "Ribera del Júcar", 
    "Ribera del Queiles", "Rioja", "Rueda", "Serra de Tramuntana-Costa Nord", 
    "Sierra de Salamanca", "Sierra Norte de Sevilla", "Sierra Sur de Jaén", 
    "Sierras de Las Estancias y Los Filabres", "Sierras de Málaga", "Somontano", 
    "Tacoronte-Acentejo", "Tarragona", "Terra Alta", "Tierra del Vino de Zamora", 
    "Toro", "Torreperogil", "Txakoli de Álava", "Txakoli de Bizkaia", 
    "Getariako Txakolina", "Uclés", "Urueña", "Utiel-Requena", "Valdejalón", 
    "Valdeorras", "Valdepeñas", "Valencia", "Valle de Güímar", 
    "Valle de La Orotava", "Valle del Cinca", "Valle del Miño-Ourense", 
    "Vallegarcía", "Valles de Benavente", "Valles de Sadacia", "Valtiendas", 
    "Vera de Estenas", "Villaviciosa de Córdoba", "Vinos de Madrid", 
    "Ycoden-Daute-Isora", "Yecla"
]

# Para generar la columana con las denomianciones
# Normalizar denominaciones (sin tildes, minúsculas)
denominaciones_normalizadas = [unidecode.unidecode(d.lower()) for d in dop_igp]

# Crear una nueva columna con la denominación encontrada o un valor por defecto
def find_denominacion(region):
    region_normalizada = unidecode.unidecode(region.lower())
    for denom, denom_normal in zip(dop_igp, denominaciones_normalizadas):
        if denom_normal in region_normalizada:
            return denom
    return "Otros"

# Aplicar la función a cada fila
df_esp['denominacion'] = df_esp['region'].apply(find_denominacion)

# Filtramos dataset y nos quedamos con las columnas que necesitamos para nuestro proyecto

df_esp_def = df_esp[['title', 'vintage', 'winery', 'style', 'variety', 'denominacion', 'price',  'designation','aging_1',
    'region', 'description', 'points']]

# Una vez definido el dataframe con el que vamos a trabajar, vamos a exportarlo a un csv como dataframe procesado 
df_esp_def.to_csv("../data/processed/data_processed.csv", index=False)