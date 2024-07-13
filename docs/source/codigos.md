# Proyecto Optimizacion 

## Métodos para funciones de una variable
<details>
<summary>Mostrar más información</summary>

### Método de división de intervalos por la mitad

<details>
<summary>Mostrar más información</summary>

Nombre del escrip: intervalos_mitad.py
```python
class Optimization:
    def __init__(self, func, a, b, epsilon):
        self.func = func
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.xm = (a + b) / 2
        self.L0 = b - a
        self.L = self.L0
```

Implementación del método de optimización utilizando la técnica de interpolación cuadrática para encontrar el mínimo de una función en un intervalo dado.

Args:
- func (callable): Función objetivo que se desea minimizar.
- a (float): Extremo izquierdo del intervalo inicial.
- b (float): Extremo derecho del intervalo inicial.
- epsilon (float): Tolerancia para la longitud del intervalo donde se considera que se ha encontrado el mínimo.

Attributes:
- func (callable): Función objetivo que se desea minimizar.
- a (float): Extremo izquierdo del intervalo actual.
- b (float): Extremo derecho del intervalo actual.
- epsilon (float): Tolerancia para la longitud del intervalo donde se considera que se ha encontrado el mínimo.
- xm (float): Punto medio del intervalo [a, b].
- L0 (float): Longitud inicial del intervalo [a, b].
- L (float): Longitud actual del intervalo [a, b].

Methods:
- optimize():
- Aplica el método de optimización utilizando la técnica de interpolación cuadrática para encontrar el mínimo de la función en el intervalo [a, b].

```python
def optimize(self):

```

Aplica el método de optimización utilizando la técnica de interpolación cuadrática para encontrar el mínimo de la función en el intervalo [a, b].

Returns:
- float: El punto donde se estima que se encuentra el mínimo de la función.

</details>

<details>
<summary>Ejemplo de uso:</summary>


```python
    from una_variable.eliminacion_regiones import intervalos_mitad as im
    from funcion.fun import funciones_una_variable as fn

    funcion = fn.f1
    a = 0  
    b = 4  
    epsilon = 0.01  
    optimizador = im.Optimization(funcion, a , b, epsilon).optimize()
```
- funcion: Funcion que se quiere optimizar
- a: Límite inferior
- b: Límite superior
- epsilon: Valor pequeño para la precisión
</details>


### Búsqueda de Fibonacci

<details>
<summary>Mostrar más información</summary>

Nombre del escrip: fibonacci.py

```python
class FibonacciOptimization:
    def __init__(self, func, a, b, n):
        self.func = func
        self.a = a
        self.b = b
        self.L = b - a
        self.n = n
        self.k = 2
```

Implementación del método de optimización utilizando la sucesión de Fibonacci para encontrar el mínimo de una función en un intervalo dado.

- Args:
- func (callable): Función objetivo que se desea minimizar.
- a (float): Extremo izquierdo del intervalo inicial.
- b (float): Extremo derecho del intervalo inicial.
- n (int): Número máximo de iteraciones.

Attributes:
- func (callable): Función objetivo que se desea minimizar.
- a (float): Extremo izquierdo del intervalo actual.
- b (float): Extremo derecho del intervalo actual.
- L (float): Longitud actual del intervalo [a, b].
- n (int): Número máximo de iteraciones.
- k (int): Contador de iteraciones.

Methods:
- fibonacci(n):
    - Calcula el n-ésimo número de la sucesión de Fibonacci.
- optimize():
    - Aplica el método de optimización utilizando la sucesión de Fibonacci para encontrar el mínimo de la función en el intervalo [a, b].


```python
def fibonacci(self, n):
```

Calcula el n-ésimo número de la sucesión de Fibonacci.

Args:
- n (int): Índice del número de Fibonacci que se desea calcular.

Returns:
- int: El valor del n-ésimo número de Fibonacci.


```python
def optimize(self):
```
Aplica el método de optimización utilizando la sucesión de Fibonacci para encontrar el mínimo de la función en el intervalo [a, b].

Returns:
- float: El punto donde se estima que se encuentra el mínimo de la función.

</details>

<details>
<summary>Ejemplo de uso:</summary>


```python
    from una_variable.eliminacion_regiones import fibonacci as fib
    from funcion.fun import funciones_una_variable as fn

    funcion = fn.f1
    a = 0  
    b = 4  
    n = 10

    optimizador = fib.FibonacciOptimization(funcion, a , b, n).optimize()
```
- funcion: funcion que se quiere optimizar 
- a: Límite inferior
- b: Límite superior
- n: Número de evaluaciones de la función
</details>





### Método de bisección

<details>
<summary>Mostrar más información</summary>

Nombre del escrip: metodo_biseccion.py
```python
class OptimizacionBusqueda:
    def __init__(self, funcion, derivada, a, b, epsilon):
        self.funcion = funcion
        self.derivada = derivada
        self.a = a
        self.b = b
        self.epsilon = epsilon
```
Implementación del método de optimización de búsqueda por bisección para encontrar el punto donde la derivada es cero.

Args:
- funcion (callable): Función a optimizar.
- derivada (callable): Función que calcula la derivada de la función objetivo.
- a (float): Extremo izquierdo del intervalo inicial.
- b (float): Extremo derecho del intervalo inicial.
- epsilon (float): Tolerancia para la magnitud de la derivada cercana a cero.

Attributes:
- funcion (callable): Función a optimizar.
- derivada (callable): Función que calcula la derivada de la función objetivo.
- a (float): Extremo izquierdo del intervalo inicial.
- b (float): Extremo derecho del intervalo inicial.
- epsilon (float): Tolerancia para la magnitud de la derivada cercana a cero.

Methods:
- optimizar():
    - Aplica el método de bisección para encontrar el punto donde la derivada de la función es cercana a cero.


```python
def optimizar(self):
    x1 = self.a
        x2 = self.b
        
        while True:
            z = (x2 + x1) / 2
            f_prime_z = self.derivada(z)
            
            if abs(f_prime_z) <= self.epsilon:
                return z
            elif f_prime_z < 0:
                x1 = z
            else:
                x2 = z
```
Aplica el método de bisección para encontrar el punto donde la derivada de la función es cercana a cero.

Returns:
- float: El punto donde se estima que la derivada es cercana a cero.


</details>

<details>
<summary>Ejemplo de uso:</summary>


```python
    from una_variable.basado_derivada import metodo_biseccion as mb
    from funcion.fun import funciones_una_variable as fn

    funcion = fn.f1
    a = 0  
    b = 4  
    epsilon = 0.001

    optimizador = mb.OptimizacionBusqueda(funcion , a , b , epsilon).optimizar()
```
- funcion: funcion que se quiere optimizar 
- a: Límite inferior
- b: Límite superior
- epsilon: Valor pequeño para la precisión
</details>



### Método de Newton-Raphson

<details>
<summary>Mostrar más información</summary>

nombre del escrip: newton_Raphson
```python
class OptimizacionNewton:
    def __init__(self, func, x0, epsilon):
        self.func = func
        self.x = x0
        self.epsilon = epsilon
        self.h = 1e-5
```

Implementación del método de optimización de Newton-Raphson para encontrar el punto donde la derivada es cero.

Args:
- func (callable): Función objetivo que se desea minimizar.
- x0 (float): Punto inicial para la optimización.
- epsilon (float): Tolerancia para la magnitud de la derivada cercana a cero.

Attributes:
- func (callable): Función objetivo que se desea minimizar.
- x (float): Punto actual en el proceso de optimización.
- epsilon (float): Tolerancia para la magnitud de la derivada cercana a cero.
- h (float): Pequeño incremento para calcular las derivadas usando diferencias finitas.

Methods:
- dfunc(x):
    - Calcula la primera derivada de la función objetivo utilizando diferencias finitas.
- ddfunc(x):
    - Calcula la segunda derivada de la función objetivo utilizando diferencias finitas.
- optimizar():
    - Aplica el método de Newton-Raphson para encontrar el punto donde la derivada de la función es cercana a cero.

```python
def dfunc(self, x):
    return (self.func(x + self.h) - self.func(x - self.h)) / (2 * self.h)
```
Calcula la primera derivada de la función objetivo utilizando diferencias finitas.
Args:
- x (float): Punto en el que se calcula la derivada.
Returns:
- float: El valor de la primera derivada en el punto dado.

```python
def ddfunc(self, x):
    return (self.func(x + self.h) - 2 * self.func(x) + self.func(x - self.h)) / (self.h ** 2)
```
Calcula la segunda derivada de la función objetivo utilizando diferencias finitas.
Args:
- x (float): Punto en el que se calcula la derivada.
Returns:
- float: El valor de la segunda derivada en el punto dado.

```python
def optimizar(self):
```
Aplica el método de Newton-Raphson para encontrar el punto donde la derivada de la función es cercana a cero.

Returns:
- float: El punto donde se estima que la derivada es cercana a cero.

</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
    from una_variable.basado_derivada import newton_Raphson as nr
    from funcion.fun import funciones_una_variable as fn

    funcion = fn.f1
    x0 = 2
    epsilon = 0.001

optimizador = nr.OptimizacionNewton(funcion , x0 , epsilon).optimizar()
```
- funcion: funcion que se quiere optimizar 
- x0: Punto inicial y nunca debe de ser cero 
- epsilon: Valor pequeño para la precisión
</details>



### Método de la secante

<details>
<summary>Mostrar más información</summary>

nombre del escrip: metodo_secante

```python
class OptimizacionSecante:
    def __init__(self, funcion, derivada, a, b, epsilon):
        self.funcion = funcion
        self.derivada = derivada
        self.a = a
        self.b = b
        self.epsilon = epsilon
```

Implementación del método de optimización de la secante para encontrar el punto donde la derivada es cero.

Args:
- funcion (callable): Función a optimizar.
- derivada (callable): Función que calcula la derivada de la función objetivo.
- a (float): Primer punto inicial para la secante.
- b (float): Segundo punto inicial para la secante.
- epsilon (float): Tolerancia para la magnitud de la derivada cercana a cero.

Attributes:
- funcion (callable): Función a optimizar.
- derivada (callable): Función que calcula la derivada de la función objetivo.
- a (float): Primer punto inicial para la secante.
- b (float): Segundo punto inicial para la secante.
- epsilon (float): Tolerancia para la magnitud de la derivada cercana a cero.

Methods:
- optimizar():
    - Aplica el método de la secante para encontrar el punto donde la derivada de la función es cercana a cero.

```python
def optimizar(self):
```
Aplica el método de la secante para encontrar el punto donde la derivada de la función es cercana a cero.

Returns:
- float: El punto donde se estima que la derivada es cercana a cero.

</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
from una_variable.basado_derivada import metodo_secante as ms
from funcion.fun import funciones_una_variable as fn

funcion = fn.f1
a = 2
b = 3
epsilon = 0.001

optimizador = ms.OptimizacionSecante(funcion , a , b , epsilon).optimizar()
```
- funcion: funcion que se quiere optimizar 
- a: Límite inferior
- b: Límite superior 
- epsilon: Valor pequeño para la precisión
</details>
</details>









































## Métodos para funciones multivariadas
<details>
<summary>Mostrar más información</summary>

### Caminata aleatoria

<details>
<summary>Mostrar más información</summary>

nombre del escrip: caminata_aleatoria.py

```python
class OptimizadorRandomWalk:
    pass
    def __init__(self, funcion, x0, epsilon, max_iter):
        self.funcion = funcion
        self.x0 = np.array(x0)
        self.epsilon = epsilon
        self.max_iter = max_iter
```

Implementación de un optimizador utilizando Random Walk.

Args:
- funcion (callable): Función objetivo que se desea minimizar.
- x0 (array-like): Punto inicial para la optimización.
- epsilon (float): Tamaño del vecindario para generar puntos aleatorios.
- max_iter (int): Número máximo de iteraciones permitidas.

Attributes:
- funcion (callable): Función objetivo que se desea minimizar.
- x0 (numpy.ndarray): Punto inicial para la optimización.
- epsilon (float): Tamaño del vecindario para generar puntos aleatorios.
- max_iter (int): Número máximo de iteraciones permitidas.

Methods:
- generacion_aleatoria(xk):
    - Genera un nuevo punto aleatorio en el vecindario de xk.
- optimizar():
    - Realiza el proceso de optimización y retorna el mejor punto encontrado.


```python
def generacion_aleatoria(self, xk):
     return xk + np.random.uniform(-self.epsilon, self.epsilon, size=xk.shape)
```
Genera un nuevo punto aleatorio en el vecindario de xk.

Args:
- xk (numpy.ndarray): Punto actual en el que se genera el nuevo punto.
Returns:
- numpy.ndarray: Nuevo punto generado aleatoriamente dentro del vecindario de xk.

```python
 def optimizar1(self):
```
Realiza el proceso de optimización utilizando el método de Random Walk.
Returns:
- numpy.ndarray: El mejor punto encontrado durante la optimización.

</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
from multivariadas.metodo_directos import caminata_aleatoria as ca
from funcion.fun import funciones as fn

funcion = fn.f_beale
x0 = [1, 1]  
epsilon = 0.1  
max_iter = 1000  

optimizador = ca.OptimizadorRandomWalk(funcion , x0 , epsilon , max_iter ).optimizar1()
```
- funcion: funcion que se quiere optimizar 
- x0: Punto inicial
- max_iter: Número máximo de iteraciones
- epsilon: Tolerancia para la generación aleatoria
</details>


### Método de Nelder y Mead (Simplex)

<details>
<summary>Mostrar más información</summary>

nombre del escrip: nelder_Mead.py

```python
class OptimizacionNelder:
    
    def __init__(self, funcion, x0, alpha, gamma, beta, epsilon):
        self.funcion = funcion
        self.x0 = np.array(x0)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.N = len(x0)
        self.simplex = self.crear_simplex_inicial()
```

Implementación del método de optimización Nelder-Mead (Simplex).

Args:
- funcion (callable): Función objetivo que se desea minimizar.
- x0 (array-like): Punto inicial para la optimización.
- alpha (float): Parámetro de expansión del simplex.
- gamma (float): Parámetro de contracción del simplex.
- beta (float): Parámetro de reflexión del simplex.
- epsilon (float): Tolerancia para la convergencia.

Attributes:
- funcion (callable): Función objetivo que se desea minimizar.
- x0 (numpy.ndarray): Punto inicial para la optimización.
- alpha (float): Parámetro de expansión del simplex.
- gamma (float): Parámetro de contracción del simplex.
- beta (float): Parámetro de reflexión del simplex.
- epsilon (float): Tolerancia para la convergencia.
- N (int): Número de dimensiones del espacio de búsqueda.
- simplex (numpy.ndarray): Simplex utilizado en el proceso de optimización.

Methods:
- crear_simplex_inicial():
    - Genera el simplex inicial basado en el punto inicial x0.
- optimizar():
    - Realiza el proceso de optimización y retorna el mejor punto encontrado.

```python
def crear_simplex_inicial(self):
```
Genera el simplex inicial basado en el punto inicial x0.

Returns:
- numpy.ndarray: Simplex inicial generado.


```python
 def optimizar(self):

```
Realiza el proceso de optimización utilizando el método Nelder-Mead (Simplex).
Returns:
- numpy.ndarray: El mejor punto encontrado durante la optimización.

</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
from multivariadas.metodo_directos import nelder_Mead as nm
from funcion.fun import funciones as fn

funcion = fn.f_beale
x0 = [1, 1]  
alpha = 5.0  
gamma = 2.0  
beta = 0.5   
epsilon = 0.001  

optimizador = nm.OptimizacionNelder(funcion , x0 ,alpha,gamma,beta, epsilon ).optimizar()
```

- funcion: funcion que se quiere optimizar 
- x0: Punto inicial
- alpha: Factor de escala
- gamma: Factor de expansión
- beta: Factor de contracción
- epsilon: Tolerancia para la generación aleatoria
</details>


### Método de Hooke-Jeeves

<details>
<summary>Mostrar más información</summary>

nombre del escrip: nhooke_jeeves.py

```python
class BusquedaPorPatrones:
    
    def __init__(self, funcion, x0, deltas, alpha, epsilon):
        self.funcion = funcion
        self.x = np.array(x0)
        self.deltas = np.array(deltas)
        self.alpha = alpha
        self.epsilon = epsilon
        self.N = len(x0)
        self.k = 0
```

Implementación de búsqueda por patrones para optimización heurística.

Args:
- funcion (callable): Función objetivo que se desea minimizar.
- x0 (array-like): Punto inicial para la búsqueda.
- deltas (array-like): Tamaños de los pasos para cada dimensión.
- alpha (float): Factor de reducción para los tamaños de paso.
- epsilon (float): Tolerancia para la convergencia.

Attributes:
- funcion (callable): Función objetivo que se desea minimizar.
- x (numpy.ndarray): Punto actual en el proceso de búsqueda.
- deltas (numpy.ndarray): Tamaños de los pasos para cada dimensión.
- alpha (float): Factor de reducción para los tamaños de paso.
- epsilon (float): Tolerancia para la convergencia.
- N (int): Número de dimensiones del espacio de búsqueda.
- k (int): Contador de iteraciones realizadas.

Methods:
- movimiento_exploratorio():
    - Realiza un movimiento exploratorio y actualiza el punto actual si encuentra una mejor solución.
- movimiento_patron(x_prev):
    - Genera un nuevo punto de patrón basado en el punto anterior.
- optimizar():
    - Realiza el proceso de optimización y retorna el mejor punto encontrado.

```python
def movimiento_exploratorio(self):
```

Realiza un movimiento exploratorio y actualiza el punto actual si encuentra una mejor solución.

Returns:
- bool: True si se realizó un movimiento que mejoró el punto actual, False en caso contrario.

```python
def movimiento_patron(self, x_prev):
```
Genera un nuevo punto de patrón basado en el punto anterior.

Args:
- x_prev (numpy.ndarray): Punto anterior en el proceso de optimización.

Returns:
- numpy.ndarray: Nuevo punto de patrón generado.

```python
def optimizar(self):
```
Realiza el proceso de optimización utilizando búsqueda por patrones.

Returns:
- numpy.ndarray: El mejor punto encontrado durante la optimización.


</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
from multivariadas.metodo_directos import hooke_Jeeves as hj
from funcion.fun import funciones as fn

funcion = fn.f_beale
x0 = [5, 1]  
deltas = [0.5, 0.5]  
alpha = 2.0  
epsilon = 0.1  
optimizador = hj.BusquedaPorPatrones(funcion , x0 ,deltas, alpha, epsilon ).optimizar()
```

- funcion: funcion que se quiere optimizar 
- x0: Punto inicial
- deltas: Incrementos de variables
- alpha: Factor de escala
- epsilon: Tolerancia para la generación aleatoria
</details>





### Método de Cauchy

<details>
<summary>Mostrar más información</summary>

nombre del escrip: cauchy.py

```python
class Cauchy:
    def __init__(self, funcion, gradiente, x0, epsilon1, epsilon2, max_iter):
        self.funcion = funcion
        self.gradiente = gradiente
        self.x0 = np.array(x0)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.max_iter = max_iter
```

Implementación del método de Cauchy para optimización con gradiente.

Args:
- funcion (callable): Función objetivo que se desea minimizar.
- gradiente (callable): Función que calcula el gradiente de la función objetivo.
- x0 (array-like): Punto inicial para la optimización.
- epsilon1 (float): Tolerancia para la norma del gradiente.
- epsilon2 (float): Tolerancia para la convergencia del tamaño de paso.
- max_iter (int): Número máximo de iteraciones permitidas.

Attributes:
- funcion (callable): Función objetivo que se desea minimizar.
- gradiente (callable): Función que calcula el gradiente de la función objetivo.
- x0 (numpy.ndarray): Punto inicial para la optimización.
- epsilon1 (float): Tolerancia para la norma del gradiente.
- epsilon2 (float): Tolerancia para la convergencia del tamaño de paso.
- max_iter (int): Número máximo de iteraciones permitidas.

Methods:
- buscar_alpha(xk, gradiente_xk):
    - Busca el tamaño de paso alpha adecuado que satisfaga la condición de terminación del gradiente.
- optimizar():
    - Realiza el proceso de optimización y retorna el mejor punto encontrado.

```python
 def aproximar_gradiente(self, xk):

```
Aproxima el gradiente de la función objetivo en el punto dado xk utilizando diferencias finitas.

Args:
- xk (np.ndarray): Punto en el cual se aproxima el gradiente.

Returns:
- np.ndarray: Aproximación del gradiente en el punto xk utilizando diferencias finitas.


```python
def buscar_alpha(self, xk, gradiente_xk):

```
Busca el tamaño de paso alpha adecuado que satisfaga la condición de terminación del gradiente.

Args:
- xk (numpy.ndarray): Punto actual en el proceso de optimización.
- gradiente_xk (numpy.ndarray): Gradiente en el punto actual xk.

Returns:
- float: Tamaño de paso alpha adecuado.

```python
def optimizar(self):

```
Realiza el proceso de optimización utilizando el método de Cauchy.
Returns:
- numpy.ndarray: El mejor punto encontrado durante la optimización.
        

</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
from multivariadas.metodos_gradiente import cauchy as cu
from funcion.fun import funciones as fn

funcion = fn.f_beale
x0 = [1, 1]  
epsilon1 = 0.01  
epsilon2 = 0.01  
max_iter = 1000  
optimizador = cu.Cauchy(funcion , x0 ,epsilon1 , epsilon2 , max_iter ).optimizar()
```

- funcion: funcion que se quiere optimizar 
- x0: Punto inicial
- epsilon1: Primera condición de terminación
- epsilon2: Segunda condición de terminación
- max_iter: Número máximo de iteraciones
</details>


### Método de Fletcher-Reeves

<details>
<summary>Mostrar más información</summary>

nombre del escrip: fletcher_Reeves.py

```python
class OptimizadorGradienteConjugado:
    
    def __init__(self, funcion, gradiente, x0, epsilon1, epsilon2, epsilon3, max_iter):
        self.funcion = funcion
        self.gradiente = gradiente
        self.x0 = np.array(x0)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.max_iter = max_iter
```
Implementación del método de optimización de Gradiente Conjugado.

Args:
- funcion (callable): Función objetivo que se desea minimizar.
- gradiente (callable): Función que calcula el gradiente de la función objetivo.
- x0 (array-like): Punto inicial para la optimización.
- epsilon1 (float): Tolerancia para la búsqueda del tamaño de paso.
- epsilon2 (float): Tolerancia para la norma relativa del cambio en x.
- epsilon3 (float): Tolerancia para la norma del gradiente.
- max_iter (int): Número máximo de iteraciones permitidas.

Attributes:
- funcion (callable): Función objetivo que se desea minimizar.
- gradiente (callable): Función que calcula el gradiente de la función objetivo.
- x0 (numpy.ndarray): Punto inicial para la optimización.
- epsilon1 (float): Tolerancia para la búsqueda del tamaño de paso.
- epsilon2 (float): Tolerancia para la norma relativa del cambio en x.
- epsilon3 (float): Tolerancia para la norma del gradiente.
- max_iter (int): Número máximo de iteraciones permitidas.

Methods:
- buscar_lambda(xk, sk):
    - Busca el tamaño de paso lambda adecuado usando la regla de Armijo.
- optimizar():
    - Realiza el proceso de optimización y retorna el mejor punto encontrado.


```python
def buscar_lambda(self, xk, sk):
        lambda_ = 1.0
        while True:
            xk1 = xk + lambda_ * sk
            if self.funcion(xk1) < self.funcion(xk) - self.epsilon1 * lambda_ * np.dot(self.gradiente(xk), sk):
                break
            lambda_ *= 0.5  
        return lambda_
```
Busca el tamaño de paso lambda adecuado usando la regla de Armijo.
Args:
- xk (numpy.ndarray): Punto actual en el proceso de optimización.
- sk (numpy.ndarray): Dirección de búsqueda (usualmente el gradiente negativo).

Returns:
- float: Tamaño de paso lambda adecuado.

```python
def optimizar(self):
```
Realiza el proceso de optimización utilizando el método de Gradiente Conjugado.
Returns:
- numpy.ndarray: El mejor punto encontrado durante la optimización.

</details>

<details>
<summary>Ejemplo de uso:</summary>

```python
from multivariadas.metodos_gradiente import fletcher_Reeves as fr
from funcion.fun import funciones as fn

funcion = fn.f_beale
x0 = [1, 1]  
epsilon1 = 0.001 
epsilon2 = 0.001  
epsilon3 = 0.001  
max_iter = 1000 
optimizador = fr.OptimizadorGradienteConjugado(funcion , x0 ,epsilon1 , epsilon2,epsilon3 , max_iter ).optimizar()

```

- funcion: funcion que se quiere optimizar 
- x0: Punto inicial
- epsilon1: Primera condición de terminación
- epsilon2: Segunda condición de terminación
- epsilon3: Tercera condición de terminación
- max_iter: Número máximo de iteraciones
</details>


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
