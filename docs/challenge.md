# Decisiones técnicas

## Parte I — Modelo

### Qué modelo elegí y por qué

Después de revisar el notebook, vi que el DS probó 6 combinaciones: XGBoost y Logistic Regression, cada uno con/sin top 10 features y con/sin balanceo de clases. Los resultados entre XGBoost y LR son bastante parecidos en f1-score general, así que no es una diferencia dramática.

Me quedé con **XGBoost + top 10 features + balanceo** por lo siguiente:

- El `feature_importance` de XGBoost me dio las 10 features más relevantes directamente, sin tener que hacer un análisis aparte. Con LR habría tenido que hacer algo como permutation importance o mirar los coeficientes, que es más indirecto.
- Reducir a 10 features no baja la performance (lo probé corriendo el notebook), pero simplifica mucho la API: el endpoint solo necesita OPERA, TIPOVUELO y MES. Si usara todas las features, necesitaría pedir SIGLADES, DIANOM, etc., que complican el contrato.
- El balanceo con `scale_pos_weight` es clave. Sin él, el modelo tiene un recall de ~0.03 en clase 1 (delays), o sea que casi nunca detecta un delay. Con balanceo el recall de delays sube a ~0.69, que tiene mucho más sentido operacionalmente. El aeropuerto prefiere que le digan "este vuelo podría atrasarse" y que a veces sea falsa alarma, a que se le escape un delay real.

### Bug que encontré

El esqueleto de `model.py` tenía `Union(Tuple[...])` con paréntesis. `Union` es un generic type, se usa con corchetes: `Union[Tuple[...]]`. Con paréntesis intenta llamar a Union como función y tira TypeError.

### Decisiones de implementación

- En `preprocess`, cuando calculo el target hago `data.copy()` antes de agregar `min_diff` y `delay`. Si no, el DataFrame original del caller queda modificado, y eso rompe si alguien reutiliza el mismo DataFrame (que es exactamente lo que hace el test en `setUp` con `self.data`).
- `predict` devuelve todos 0 si el modelo no está entrenado. No es ideal, pero `test_model_predict` llama a predict sin hacer fit antes. En producción esto no pasa porque el modelo se entrena en el startup.
- Los hiperparámetros (`learning_rate=0.01`, `random_state=1`) los dejé igual que en el notebook. No hice tuning adicional porque el challenge dice que no es necesario mejorar el modelo.

## Parte II — API

Implementé el POST /predict con validación usando Pydantic:

- `MES` tiene que estar entre 1 y 12
- `TIPOVUELO` solo acepta "N" o "I"
- `OPERA` se valida contra la lista de aerolíneas que aparecen en el dataset

Si alguna validación falla, la API devuelve 400. Para esto tuve que agregar un exception handler para `RequestValidationError` que retorna `JSONResponse` con status 400, porque por defecto FastAPI devuelve 422.

El modelo se entrena en el `on_event("startup")` de FastAPI, no cuando se importa el módulo. Si lo hacía al import, cada vez que algo importaba `challenge.api` se entrenaba el modelo de nuevo (incluyendo los tests), lo que era lento e innecesario.

## Parte III — Deploy

Configuré un Dockerfile con `python:3.11-slim` que levanta uvicorn en el puerto 8080. Corre con usuario no-root (buena práctica de seguridad). Cloud Run maneja el load balancing, así que no necesito nginx adelante.

## Parte IV — CI/CD

- **CI**: Corre en push a main/develop y en PRs. Ejecuta model-test y api-test.
- **CD**: Se dispara después de que CI pasa (con `workflow_run`). Buildea la imagen, la sube a GCR y deploya en Cloud Run. No quería que se pueda deployar si los tests fallaron.
