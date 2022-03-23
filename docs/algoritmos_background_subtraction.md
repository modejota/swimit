La eliminación de fondos es un enfoque muy utilizado para detectar objetos en movimiento en una secuencia de frames tomados por una cámara estática.
Este enfoque se basa en hallar las diferencias entre el frame actual y un frame de referencia, anterior o posterior temporalmente. Los objetos en movimiento serán aquellos que provoquen variaciones en los valores de los píxeles, mientras que el fondo permanecerá sin alteraciones. 

[https://analyticsindiamag.com/using-background-subtraction-methods-in-image-processing/]

# Diferencia de frames

Nos limitamos a calcular las diferencias entre el frame actual y el frame inmediatamente anterior. Consideraremos como parte del movimiento aquellos píxeles cuyo valor diferencia se encuentre por encima de un determinado umbral.

Este enfoque es muy sensible al valor umbral seleccionado, así como la estructura del objeto, su velocidad y el frame rate de la secuencia de vídeo a analizar. 

[HSMD_An_Object_Motion_....]


# Gaussian Mixture Models (GMM, de manera general)

Stauffer and Grimson, así como KaewTraKulPong y Borden sugieren modelizar cada píxel como una mezcla de k distribuciones gaussianas (con k entre 3 y 5 normalmente). 
Se asume que las diferentes distribuciones representan de forma distinta los colores correspondientes al fondo estático y a la parte en movimiento de la imagen.
El peso de cada distribución es proporcional al tiempo en que un mismo color se mantiene para un determinado píxel. Así, cuando el peso de la distribución para el píxel es pequeño se considera parte de un objeto en movimiento.

{De aquí se derivan MOG y MOG2. En algunos sitios GMM == MOG, pero se supone que no es así.}

# KNN (K Nearest Neighbor)

# LSBP (Local SVD Binnary Patern)
Artículo de 2016 que me queda pendiente por leer

# GSOC (Google Summer of Code)
[https://learnopencv.com/background-subtraction-with-opencv-and-bgs-libraries/#gsoc-algorithm]
[https://github.com/opencv/opencv_contrib/blob/master/modules/bgsegm/src/bgfg_gsoc.cpp]

Durante el Google Summer of Code (GSoC) de 2017 se proporcionó un nuevo algoritmo para la eliminación de fondos, el cual ha recibido como nombre las siglas de dicho programa de formación. Este algoritmo es una evolución del ya conocido LSBP, para que fuera más robusto y rápido. El método GSoC se basa en valores de color RGB en lugar de descriptores LSBP. 

Es el segundo algoritmo, de entre aquellos disponibles en la biblioteca OpenCV, que mejor resultado proporciona en la práctica sobre los bancos de pruebas CDnet2012 y CDnet2014. Sólo le supera en rendimiento SubSENSE. (Dada la aplicación, podemos tolerar un menor rendimiento si implica tiempos de ejecucion menores, investigar mejor el Subsense, o utilizarlo para futuras comparativas)

La implementación de este algoritmo GSoC no surge ni está relacionada con ningún artículo académico, por lo que para conocer su manera de operar deberemos consultar y analizar el fichero de código fuente correspondiente. El fichero al que hacemos alusión es "bgfg_gsoc.cpp", del módulo "bgsegm"; el cual no se encuentra incluido con la instalación básica por defecto de OpenCV mediante el gestor de paquetes.

""" ESTO SE PUEDE IGNORAR AL PRINCIPIO Y MENCIONAR DE SER NECESARIO, PROBABLEMENTE SOLO EL LEARNING RATE"
Comenzaremos enumerando y comentando los parámetros que recibe el constructor de la clase BackgroundSubtractorGSOC:
- mc: bandera de compensación de movimiento de la cámara
- nSamples: número de muestras a mantener en cada punto del cuadro.
- replaceRate: probabilidad de reemplazar la muestra antigua, es decir, la rapidez con la que se actualizará el modelo.
- propagationRate: probabilidad de propagación a los vecinos.
- hitsThreshold: cuántos positivos debe obtener la muestra antes de ser considerada como un posible reemplazo.
- alpha: coeficiente de escala para el umbral.
- beta: coeficiente de sesgo para el umbral.
- blinkingSupressionDecay: factor de decaimiento de la supresión del parpadeo.
- blinkingSupressionMultiplier: multiplicador de supresión de parpadeo.
- noiseRemovalThresholdFacBG: intensidad de la supresión del ruido de fondo.
- noiseRemovalThresholdFacFG: intensidad de la eliminación del ruido en el primer plano.


El cálculo se encuentra reflejado en el método apply().

Como mencionamos anteriormente, el método se basa en el uso de valores de color RGB. Así, lo primero que hace la función es comprobar si el frame posee tres canales, y si no es así realiza la conversión desde grises. 
Intencionalmente, seleccionamos la banda de crominancia roja de cada uno de los frames, que se encuentra representada en escala de grises. Cuando el método toma dicha banda, la triplica para convertirla en un valor RGB, siendo los valores de las bandas idénticos.
De esta manera, obtenemos los beneficios de utilizar el espacio de color YCbCr (resistencia a cambios de iluminación, y seleccionar por cromnancia según interés), y los beneficios de este algoritmo sobre LSBP. 

Los píxeles que cambian frecuentemente entre el primer plano y el fondo se definen como parpadeantes. El enfoque GSoC BS aplica una heurística especial para la detección de píxeles parpadeantes:

Aquí la supresión del parpadeo puede definirse como un mapa de píxeles parpadeantes obtenido mediante el XOR de la máscara actual y la anterior. A continuación, los valores obtenidos con los coeficientes de supresión del parpadeo se eligen al azar para clasificar los píxeles adecuados como fondo.

El valor del umbral para la eliminación del ruido se produce con la multiplicación  de los valores noiseRemovalThresholdFacBG y noiseRemovalThresholdFacFG en el área de la máscara. Además, los valores de la máscara se actualizan de acuerdo con el umbral obtenido.

Por último, la máscara producida sufre un procesamiento adicional, consiste en la eliminación de ruido y la aplicación de un filtro de desenfoque gaussiano.

(Se podría mencionar que es un algoritmo bastante costoso en tiempo de cómputo, al tener una complejidad algorítmica de orden cuadrático, debe recorrer varias veces todos los píxeles del frame para realizar distintas operaciones. Por lo menos realiza una pasada en tiempo cúbico para construir el modelo del fondo inicial.)

Se requieren unos doscientos frames más o menos para entrenar el modelo y que se produzca la eliminación del fondo de manera correcta. En el dominio de nuestro problema, esto pueden ser aproximadamente 2 o 6 segundos; en función del framerate (y resolución) de nuestro vídeo. Teniendo en cuenta el tiempo que toma iniciar la grabación y avisar al entrenador de que puede indicar la salida del nadador, la preparación de los natadores, y el inicio de la prueba, el entrenamiento del modelo se realiza a tiempo. (REDACTAR ALGO MEJOR)

[HSMD_An_Object_Motion_....]
