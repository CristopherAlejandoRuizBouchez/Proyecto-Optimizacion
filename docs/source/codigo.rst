codigo
=====
.. _codigo:
Métodos para funciones de una variable
====================================
Métodos de eliminación de regiones
----------------
Método de división de intervalos por la mitad
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    class Optimization:
    def __init__(self, func, a, b, epsilon):
        self.func = func
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.xm = (a + b) / 2
        self.L0 = b - a
        self.L = self.L0
Implementación del método de optimización utilizando la técnica de interpolación cuadrática para encontrar el mínimo de una función en un intervalo dado.


Args:
o. func (callable): Función objetivo que se desea minimizar.
o. a (float): Extremo izquierdo del intervalo inicial.
o. b (float): Extremo derecho del intervalo inicial.
o. epsilon (float): Tolerancia para la longitud del intervalo donde se considera que se ha encontrado el mínimo.

Attributes:
o. func (callable): Función objetivo que se desea minimizar.
o. a (float): Extremo izquierdo del intervalo actual.
o. b (float): Extremo derecho del intervalo actual.
o. epsilon (float): Tolerancia para la longitud del intervalo donde se considera que se ha encontrado el mínimo.
o. xm (float): Punto medio del intervalo [a, b].
o. L0 (float): Longitud inicial del intervalo [a, b].
o. L (float): Longitud actual del intervalo [a, b].

Methods:
o. optimize():
o. Aplica el método de optimización utilizando la técnica de interpolación cuadrática para encontrar el mínimo de la función en el intervalo [a, b].



.. code-block::python
    def optimize(self):

Aplica el método de optimización utilizando la técnica de interpolación cuadrática para encontrar el mínimo de la función en el intervalo [a, b].

Returns:
o. float: El punto donde se estima que se encuentra el mínimo de la función.

**Ejemplo de uso**

.. code-block::python
    from una_variable.eliminacion_regiones import intervalos_mitad as im
    from funcion.fun import funciones_una_variable as fn

    funcion = fn.f1
    a = 0  
    b = 4  
    epsilon = 0.01  
    optimizador = im.Optimization(funcion, a , b, epsilon).optimize()

o. funcion: Funcion que se quiere optimizar
o. a: Límite inferior
o. b: Límite superior
o. epsilon: Valor pequeño para la precisión
