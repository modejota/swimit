# RGB

RGB es un modelo de color aditivo, es decir, un modelo en el que cada color se representa mediante la mezcla por adición de otros colores. En concreto, se utilizan los colores primarios aditivos; rojo (R), verde (G) y azul (B).

Este modelo de color se basa en el tri-cromatismo de los receptores de nuestro sistema de visión, los conos L, M y S. 

El número de colores que nos permite componer depende del número de bits que utilicemos para representar los valores de los colores primarios. Si utilizamos 8 bits por color podemos representar 16.777.216 colores diferentes, mientras que si usamos 16 bits por color podremos represtanr cerca de 281 billones de colores.

Podemos representar las combinaciones de colores mediante un cubo, tal que rojo, verde y azul no estén en vértices adyacentes. {Adjuntar imagen}

Resulta de especial relevancia en el mundo de la electrónica, y, por ejemplo, es el espacio de color del que hacen uso las pantallas LED.

[https://lledoenergia.es/colorimetria-iii-espacios-de-color-hsl-hsv-y-rgb/]
[https://es.sttmedia.com/modelo-de-color-rgb]

# HSV

Es un modelo de color alternativo a RGB, creado por A.R. Smith en 1978. Su principal objetivo es representar con mayor exactitud la forma en la que la visión humana percibe los atributos que componen cada color que los modelos aditivos (o sustractivos). 

Utiliza tres componentes:
- H (Tono): representa el atributo de una sensación visual según el cual un área parece similar al color rojo, amarillo, verde, azul o a una combinación de dos de los colores anteriores. La gama cromática se representa en una rueda circular, de manera que indicamos el color mediante un ángulo entre los 0 y 360 grados. 
- S (Saturacion): indica la cantidad de color que tenemos en un área en proporción a su brillo. Suele representarse en porcentajes.
- V (Valor): representa el atributo de una sensación visual según el cual un área parece emitir más o menos luz. Suele representarse en porcentajes.

[https://medium.com/programming-fever/how-to-find-hsv-range-of-an-object-for-computer-vision-applications-254a8eb039fc] 
[https://es.sttmedia.com/modelo-de-color-hsv]  
[https://www.comunicacion-multimedia.info/2010/05/modos-o-modelos-de-color-hsb-o-hsv-y.html]
{Apuntes propios de la asignatura}

Los colores se suelen representar mediante un cono. {Adjuntar imagen}

# ¿Por qué elegimos HSV?

El espacio de color HSV es utilizado en visión por computador debido a su "rendimiento superior" frente a RGB al representar variaciones de los niveles de iluminación. 

Supongamos que tenemos dos colores similares, que varían ligeramente en iluminación/brillo. Su representación RGB tendrá dos tripletas distintas, en las que pueden variar los tres valores; sin embargo, las representaciones HSV diferirán solamente en tono.

En visión por computador es frecuente que querramos realizar filtrados en funció n del color. El espacio de color HSV es más robusto frente a pequeños cambios de iluminación de la escena, lo que nos permite realizar el filtrado de forma más precisa y teniendo en cuenta menos parámetros.

[https://dsp.stackexchange.com/questions/2687/why-do-we-use-the-hsv-colour-space-so-often-in-vision-and-image-processing]  
[https://www.quora.com/Why-use-an-HSV-image-for-color-detection-rather-than-an-RGB-image#:~:text=The%20reason%20we%20use%20HSV,relatively%20lesser%20than%20RGB%20values.]