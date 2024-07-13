# Proyecto Optimizacion 

## Métodos para funciones de una variable
<details>
<summary>Mostrar más información</summary>

### Método de división de intervalos por la mitad

<details>
<summary>Mostrar más información</summary>















































## Métodos para funciones multivariadas
<details>
<summary>Mostrar más información</summary>









### Método de Newton

<details>
<summary>Mostrar más información</summary>

nombre del escrip: newton.py

```python
class Newton:
    def __init__(self, funcion, gradiente, hessiana, x0, epsilon1, epsilon2, max_iter):
        self.funcion = funcion
        self.gradiente = gradiente
        self.hessiana = hessiana
        self.x = np.array(x0)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.max_iter = max_iter

```
Implementación del Método de Cauchy Modificado para optimización con Hessiana.

Args:
- funcion (callable): Función objetivo que se desea minimizar.
- gradiente (callable): Función que calcula el gradiente de la función objetivo.
- hessiana (callable): Función que calcula la hessiana de la función objetivo.
- x0 (array-like): Punto inicial para la optimización.
- epsilon1 (float): Tolerancia para la norma del gradiente.
- epsilon2 (float): Tolerancia para la búsqueda del tamaño de paso.
- max_iter (int): Número máximo de iteraciones permitidas.

Attributes:
- funcion (callable): Función objetivo que se desea minimizar.
- gradiente (callable): Función que calcula el gradiente de la función objetivo.
- hessiana (callable): Función que calcula la hessiana de la función objetivo.
- x (numpy.ndarray): Punto actual en el proceso de optimización.
- epsilon1 (float): Tolerancia para la norma del gradiente.
- epsilon2 (float): Tolerancia para la búsqueda del tamaño de paso.
- max_iter (int): Número máximo de iteraciones permitidas.

Methods:
- optimizar():
    - Realiza el proceso de optimización y retorna el mejor punto encontrado.
- busqueda_unidireccional(f_alpha, epsilon2):
    - Realiza una búsqueda unidireccional para encontrar el tamaño de paso adecuado.

```python
def optimizar(self):
```
Realiza el proceso de optimización utilizando el Método de Cauchy Modificado.
Returns:
- numpy.ndarray: El mejor punto encontrado durante la optimización.
        
```python
def busqueda_unidireccional(self, f_alpha, epsilon2):
```
Realiza una búsqueda unidireccional para encontrar el tamaño de paso adecuado.

Args:
- f_alpha (callable): Función que evalúa la función objetivo en un punto dado alpha.
- epsilon2 (float): Tolerancia para la búsqueda del tamaño de paso.

Returns:
- float: Tamaño de paso alpha adecuado.
    
</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
import numpy as np
from multivariadas.metodos_gradiente import newton
from funcion.fun import funciones as fn

funcion = fn.f_beale

def gradiente_ejemplo(x):
    return np.array([2*x[0], 2*x[1]])  

def hessiana_ejemplo(x):
    return np.array([[2, 0], [0, 2]])  

x0 = [1, 1]  
epsilon1 = 0.001 
epsilon2 = 0.01  
max_iter = 1000  

optimizador = newton.Newton(funcion, gradiente_ejemplo, hessiana_ejemplo, x0, epsilon1, epsilon2, max_iter)
resultado = optimizador.optimizar()
```

- funcion: funcion que se quiere optimizar 
- x0: Punto inicial
- epsilon1: Primera condición de terminación
- epsilon2: Segunda condición de terminación
- max_iter: Número máximo de iteraciones
</details>
</details>


## Funciones Prueba
<details>
<summary>Mostrar más información</summary>

### Funciones una variable 

<details>
<summary>Mostrar más información</summary>

Nombre del escrip: funciones_una_variable.py
```python
def f1(x):
    return x**2 + 54/x

def f2(x):
    return x**3 + 2*x - 3

def f3(x):
    return x**4 + x**2 - 33

def f4(x):
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x
```
- f1: Esta función calcula el valor de la expresión x^2 + 54/x en un punto dado x.
- f2: Esta función calcula el valor de la expresión x^3 + 2x - 3 en un punto dado x.
- f3: Esta función calcula el valor de la expresión x^4 + x^2 - 33 en un punto dado x.
- f4: Esta función calcula el valor de la expresión 3x^4 - 8x^3 - 6x^2 + 12x en un punto dado x.
</details>

### Funciones Multivariable 

<details>
<summary>Mostrar más información</summary>

Nombre del escrip: funciones_una_variable.py
```python
def f_ackley(x):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20

def f_beale(x):
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3

def f_bukin(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)

def f_jorobas(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2

def f_cruzada_bandeja(x):
    return -0.0001 * np.power(np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2))/np.pi)) + 1, 0.1)

def f_esfera(x):
    return x[0]**2 + x[1]**2

def f_facil(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

def f_levi(x):
    return (np.sin(3*np.pi*x[0]))**2 + (x[0] - 1)**2 * (1 + (np.sin(3*np.pi*x[1]))**2) + (x[1] - 1)**2 * (1 + (np.sin(2*np.pi*x[1]))**2)

def f_matias(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def f_McCormick(x):
    return np.sin(x[0] + x[1]) + (x[0] * x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

def f_mesasoporte(x):
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))

def f_portahuevos(x):
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0]/2 + x[1] + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))

def f_goldstein(x):
    term1 = (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2))
    term2 = (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
    return term1 * term2

def f_restringida(x, A=10):
    n = len(x)
    return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))

def f_Schaffer04(x):
    return 0.5 + (np.cos(np.sin(np.abs(x[0]**2 - x[1]**2)))**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2

def f_Schaffer(x):
    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2

def f_shequel(x, a, c):
    m = len(c)
    n = len(x)
    result = 0
    for i in range(m):
        inner_sum = 0
        for j in range(n):
            inner_sum += (x[j] - a[i, j])**2
        result += 1 / (c[i] + inner_sum)
    return result

def f_stand(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def f_himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def f_rosenbrock_restringida_cubica(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def f_mishra(x):
    return np.sin(x[1]) * np.exp((1 - np.cos(x[0]))**2) + np.cos(x[0]) * np.exp((1 - np.sin(x[1]))**2) + (x[0] - x[1])**2

def f_rosenbrock_constrained(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def f_simionescu(x):
    return 0.1 * x[0] * x[1]
```
f_ackley(x)
- Descripción: Calcula el valor de la función Ackley en un punto dado x. Es una función comúnmente usada para  pruebas de optimización.

f_beale(x)
- Descripción: Calcula el valor de la función Beale en un punto dado x. Es conocida por sus múltiples mínimos locales.

f_bukin(x)
- Descripción: Calcula el valor de la función Bukin N.6 en un punto dado x. Es conocida por su estrecho valle.

f_jorobas(x)
- Descripción: Calcula el valor de la función de jorobas en un punto dado x.

f_cruzada_bandeja(x)
- Descripción: Calcula el valor de la función Cruzada de Bandeja en un punto dado x.

f_esfera(x)
- Descripción: Calcula el valor de la función Esfera en un punto dado x. Es una función simple utilizada para pruebas de optimización.

f_facil(x)
- Descripción: Calcula el valor de la función Fácil en un punto dado x.

f_levi(x)
- Descripción: Calcula el valor de la función Lévi en un punto dado x.

f_matias(x)
- Descripción: Calcula el valor de la función Matias en un punto dado x.

f_McCormick(x)
- Descripción: Calcula el valor de la función McCormick en un punto dado x.

f_mesasoporte(x)
- Descripción: Calcula el valor de la función Mesa de Soporte en un punto dado x.

f_portahuevos(x)
- Descripción: Calcula el valor de la función Porta Huevos en un punto dado x.

f_goldstein(x)
- Descripción: Calcula el valor de la función Goldstein en un punto dado x.

f_restringida(x, A=10)
- Descripción: Calcula el valor de la función Restringida en un punto dado x.

f_Schaffer04(x)
- Descripción: Calcula el valor de la función Schaffer N.4 en un punto dado x.

f_Schaffer(x)
- Descripción: Calcula el valor de la función Schaffer en un punto dado x.

f_shequel(x, a, c)
- Descripción: Calcula el valor de la función Shekel en un punto dado x.

f_stand(x)
- Descripción: Calcula el valor de la función Stand en un punto dado x.

f_himmelblau(x)
- Descripción: Calcula el valor de la función Himmelblau en un punto dado x.

f_rosenbrock_restringida_cubica(x)
- Descripción: Calcula el valor de la función Rosenbrock Restringida Cúbica en un punto dado x.

f_mishra(x)
- Descripción: Calcula el valor de la función Mishra en un punto dado x.
</details>
</details>
