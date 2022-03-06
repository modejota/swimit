Los aquí presentados, son estadísticos utilizados para comparar la similitud entre dos muestras.
[https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/]

# Índice de Jaccard (Intersection Over Union, IoU)
[https://es.wikipedia.org/wiki/%C3%8Dndice_de_Jaccard]{Sé que no queda bien, buscar otra página}

La fórmula se encuentra en la web. Se calcula como la división entre la cardinalidad de la interseccion de ambos conjuntos y la cardinalidad de la unión de los conjuntos. El indice de Jaccard tiene otra expresion en funcion de TP (true possitives), FP (false possitives), FN (false negatives).
Toma valores entre 0 y 1, de manera que el 0 representa desigualdad total entre los conjutnos, y 1 representa igualdad total.
[https://statologos.jaol.net/jaccard-similarity-python/]

# Coeficiente de Sorensen-Dice (F1Score)
(Con el segundo nombre si que lo he visto en algún paper)

[https://es.wikipedia.org/wiki/Coeficiente_de_Sorensen-Dice]{Sé que no queda bien, buscar otra página}

La fórmula se encuentra en la web, y en esta, 
|A| y |B| son el número de muestras de los conjuntos de mismo nombre, respectivamente, mientras que C es el número de especies compartidas por las dos muestras.

El coeficiente de similitud varía entre 0 y 1.
Es una métrica similar a Jaccard, pero con algunas diferencias. (INVESTIGAR MÁS)
Parece ser más apropiado para distribuciones muy desequilibradas

[https://towardsdatascience.com/the-f1-score-bec2bbc38aa6]

## Link
[https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2](Para ambos indices)