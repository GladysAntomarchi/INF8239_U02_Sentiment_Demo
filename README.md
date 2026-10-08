# INF-8239 · Demostración de Análisis de Sentimientos

**Asignatura:** Ciencia de Datos II (INF-8239)  
**Unidad:** Procesamiento de Lenguaje Natural (NLP)

Este repositorio contiene una demostración práctica de **análisis de sentimientos** utilizando el corpus público **TweetEval**.

La actividad corresponde a una extensión práctica vinculada a LAB05 y tiene como propósito aplicar un pipeline de clasificación de texto a tres clases de sentimiento:

- `negative`
- `neutral`
- `positive`

La demostración no constituye un laboratorio independiente ni genera una puntuación adicional.

---

## Objetivo

Construir una línea base reproducible para clasificar tweets según su sentimiento utilizando:

- TF-IDF
- Logistic Regression
- evaluación con F1 macro
- métricas por clase
- matriz de confusión
- análisis de errores

Además, se presta especial atención a dificultades relacionadas con:

- desbalance de clases;
- negación;
- ironía;
- falta de contexto;
- lenguaje figurativo;
- diferencias culturales y lingüísticas.

---

## Dataset

Se utilizó el dataset público **TweetEval**, específicamente la tarea:

```text
sentiment
```

Repositorio utilizado:

```python
load_dataset("cardiffnlp/tweet_eval", "sentiment")
```

El dataset contiene las columnas:

```text
text
label
```

y utiliza tres clases:

| Etiqueta | Clase |
|---:|---|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

---

## Distribución de los datos

Los conjuntos utilizados fueron:

| Conjunto | Registros |
|---|---:|
| Train | 45,615 |
| Validation | 2,000 |
| Test | 12,284 |

La distribución aproximada del conjunto de entrenamiento fue:

| Clase | Proporción |
|---|---:|
| negative | 15.55 % |
| neutral | 45.32 % |
| positive | 39.13 % |

La clase negativa está menos representada, por lo que se utilizó:

```python
class_weight="balanced"
```

en la regresión logística.

---

## Pipeline utilizado

El modelo combina TF-IDF y regresión logística dentro de un mismo pipeline:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=25_000
    )),
    ("classifier", LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        random_state=42
    ))
])
```

TF-IDF se ajusta únicamente con los datos de entrenamiento, reduciendo el riesgo de fuga de información.

---

## Resultados

### Validación

El F1 macro obtenido en validación fue aproximadamente:

```text
0.641
```

### Prueba

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.556 | 0.676 | 0.610 |
| neutral | 0.654 | 0.536 | 0.589 |
| positive | 0.546 | 0.594 | 0.569 |

Resultados globales:

- **Accuracy:** 0.593
- **F1 macro:** 0.589

La clase con menor recall fue:

```text
neutral = 0.536
```

---

## Matriz de confusión

La matriz obtenida fue:

```text
                 Predicción
              neg   neu   pos
Real neg     2687  1027   258
Real neu     1842  3182   913
Real pos      305   660  1410
```

El error más frecuente fue:

```text
neutral → negative = 1,842 casos
```

Esto indica que el modelo tiene dificultades para distinguir mensajes neutrales que contienen vocabulario asociado con situaciones negativas.

---

## Análisis de errores

Se revisó una muestra reproducible de 20 errores de clasificación.

Los principales tipos de dificultad identificados fueron:

- falta de contexto;
- lenguaje político o informativo con carga negativa;
- negación;
- ironía;
- humor;
- referencias culturales;
- hashtags;
- sentimiento implícito;
- ambigüedad entre neutral y negativo.

Por ejemplo, algunos tweets neutrales contienen palabras como:

```text
crisis
war
sanctions
oppression
apartheid
```

Aunque el mensaje sea informativo, TF-IDF puede asociar esas palabras con sentimiento negativo.

También se observaron ejemplos donde el significado depende de ironía o referencias culturales, lo cual representa una limitación importante para un modelo basado principalmente en asociaciones léxicas.

---

## Discusión

### ¿Qué errores parecen relacionados con negación, ironía o falta de contexto?

La revisión de 20 errores mostró que la **falta de contexto** es la dificultad predominante.

También se identificaron casos de:

- negación explícita;
- humor;
- ironía;
- lenguaje figurativo;
- referencias culturales.

Estos ejemplos muestran que TF-IDF puede capturar patrones léxicos, pero no siempre comprende adecuadamente intención, contexto o composición semántica.

### ¿Qué clase obtiene menor recall?

La clase **neutral** obtuvo el menor recall:

```text
0.536
```

Esto significa que el modelo reconoce correctamente aproximadamente el 53.6 % de los tweets realmente neutrales.

### ¿Cambiaría la decisión si los falsos negativos tuvieran mayor costo?

Sí.

Si los falsos negativos tuvieran mayor costo, sería necesario priorizar el recall de la clase de interés y no seleccionar el modelo únicamente por F1 macro o accuracy global.

Dependiendo del contexto, podrían evaluarse:

- ajustes de pesos de clase;
- modificación de umbrales;
- métricas ponderadas;
- modelos alternativos.

### ¿Qué limitaciones tendría aplicar el modelo a mensajes dominicanos en español?

El modelo fue entrenado con tweets en inglés, por lo que tendría limitaciones importantes al aplicarlo directamente a mensajes dominicanos en español.

Entre las principales diferencias se encuentran:

- idioma;
- vocabulario;
- modismos;
- abreviaciones;
- errores ortográficos;
- expresiones coloquiales;
- emojis;
- referencias culturales.

Además, la **jerga dominicana** representa una dificultad adicional. Muchas expresiones, dobles sentidos y formas coloquiales solo pueden interpretarse correctamente dentro del contexto sociocultural dominicano.

Una expresión podría parecer neutral para un modelo extranjero y, sin embargo, tener una connotación negativa, positiva, irónica o sarcástica para un hablante dominicano.

Por estas razones, el modelo no debería utilizarse directamente sobre mensajes dominicanos en español sin adaptación o reentrenamiento con datos representativos del idioma, la jerga y el contexto cultural local.

---

## Verificaciones

Se ejecutaron las siguientes verificaciones mínimas:

```python
assert len(test_pred) == len(test_df)
assert set(np.unique(test_pred)).issubset({0, 1, 2})
assert "tfidf" in model.named_steps
assert "classifier" in model.named_steps
```

Resultado:

```text
Verificaciones superadas
```

---

## Notebook

El notebook ejecutado se encuentra en:

```text
Demostración_._Análisis_de_sentimientos.ipynb
```

El archivo incluye:

- carga del dataset;
- análisis de distribución;
- entrenamiento;
- evaluación;
- matriz de confusión;
- muestra de errores;
- verificaciones;
- discusión interpretativa.

---

## Fuente

Barbieri, F. et al. (2020).  
*TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.*  
Findings of EMNLP 2020.  
DOI: 10.18653/v1/2020.findings-emnlp.148

---

## Uso responsable

Esta demostración tiene fines académicos.

Las predicciones del modelo no deben interpretarse como una lectura completa del estado emocional de una persona.

El sentimiento detectado corresponde únicamente a patrones presentes en el texto y puede verse afectado por contexto, ironía, cultura, idioma y ambigüedad.
