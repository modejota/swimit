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

{De aquí se derivan MOG y MOG2. En algunos sitios GMM == MOG, pero no es así.}

# KNN (K Nearest Neighbor)

# LSBP (Local Singular Binnary Patern)

# GSOC (Google Summer of Code)

En 2017, OpenCV lanzó una versión mejorada del algoritmo LSBP, desarrollada durante el Google Summer of Code de dicho año, y que recibe como nombre las siglas de dicho programa de formarción, GSOC. 
Este algoritmo mejora a LSBP al utilizar descriptores de color y varias ¿heurísticas de estabilización?.
Es el algoritmo, de entre aquellos disponibles en la biblioteca OpenCV, que mejor resultado proporciona en la práctica sobre los bancos de pruebas CDnet2012 y CDnet2014. 