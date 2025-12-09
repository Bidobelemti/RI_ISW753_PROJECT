# Sistema de Recuperación de Información (IR System)

Este proyecto implementa un motor de búsqueda y recuperación de información basado en corpus de texto (noticias de la BBC). El sistema permite realizar consultas utilizando tres modelos clásicos de recuperación de información para rankear la relevancia de los documentos.

## Integrantes
* **Morales Mauricio**
* **Rivadeneira Jossue**
* [URL Video e informe](https://epnecuador-my.sharepoint.com/my?id=%2Fpersonal%2Fbryan%5Fmorales06%5Fepn%5Fedu%5Fec%2FDocuments%2FRI%5FPROYECTO%5F2025B%5FBIM1)

## Características
El sistema implementa los siguientes modelos de recuperación:
1. **Modelo Booleano / Jaccard:** Basado en la intersección de conjuntos de términos (binario).
2. **TF-IDF (Term Frequency - Inverse Document Frequency):** Utiliza matrices dispersas (sparse matrices) para eficiencia en memoria.
3. **BM25 (Best Matching 25):** Modelo probabilístico avanzado que considera la longitud del documento y la saturación de términos.

## Requisitos Previos

El proyecto está construido en **Python 3**.

### Dependencias
El código hace uso de `pandas`, `numpy` y `scipy`. Para instalar todo lo necesario rjecuta el comando de instalación:
```bash
pip install -r requirements.txt
```

## Configuración del Dataset

Para que el sistema funcione, es **obligatorio** descargar el dataset de noticias.

1. Descarga el archivo `bbc_news.csv` desde Kaggle:
    * [BBC News Dataset - Kaggle](https://www.kaggle.com/datasets/gpreda/bbc-news)
2. Crea una carpeta llamada `data` en la raíz del proyecto.
3. Coloca el archivo descargado dentro de esa carpeta.

La estructura de archivos debe verse así para que el código fuente funcione:

```text
PROYECTO/
├── data/
│   └── bbc_news.csv       <-- Archivo descargado obligatorio
├── src/
│   └── iswd753.py         <-- Módulo con funciones de preprocesamiento
├── main.py                <-- Archivo principal de ejecución
├── requirements.txt
└── README.md
```
## Ejecución

Para iniciar el sistema de recuperaicón a través de CLI ejecuta:

```bash
python main.py
```

## Flujo de Uso

Una vez iniciado el programa este realizará el preprocesamiento de manera automática. Luego sigue estos pasos:

1. Seleccionar modelo: Escribe `jaccard`, `tfidf` o `bm25`
2. Ingresar Consulta: Escribe los términos que deseas buscar
3. Definir Top K: Indica cuántos resultados quieres ver

## Ejemplo de Salida

```text
Sistema IR
Modelo [jaccard | tfidf | bm25] or eval: bm25
Consulta: economy growth
Top K: 3

--- Top 3 resultados ---
 Index    Score                                                    Documento                                                          URL
 34176 1.000000 Adrian Ramsay says economic growth statistics do not capt...               https://www.bbc.com/news/articles/c511lz64rrpo
 14763 0.923243 Jeremy Hunt tells the BBC his plans will kick start growt... https://www.bbc.co.uk/news/uk-politics-64964911?at_medium...
 34755 0.648271 Labour has launched its manifesto, focusing on economic g...                 https://www.bbc.com/news/videos/c4nn8e70z5no
¿Otra consulta? (S/N):

```
