
## Primera parte: algoritmos no supervisados (sin etiquetas):   
Estudiar los resultados y ver que tal.  
Además hablar de la reduccion de la dimensionalidad con las caracteristicas importantes, hacerlos de las dos maneras y hablar de la maldicion de la dimensionalidad.
**Algoritmos a probar y comentar:**  
> Meanshift  
> Probamos el DBSCAN  
> Gaussian Mixture (EM)  
> Clustering jerarquico.   

En el enunciado de dice que no usemos las etiquetas para crear el algoritmo, pero las podemos usar para evaluar el algortimo.

## Segunda parte: algortimos supervisados:  
Primero reducimos el conjunto con algun algortimo de clustering (habría que discutir si esto es lo más recomendable para MLP etc):  
> KMedioids.  
> KMeans: parece que va mejor pero este no nos selecciona puntos, tenemos un pasa intermedio para la seleccion de estos.  
> Gaussianas  

Algortimos para aplicar:  
> Trees  
> SVM  
> 

## Tercer parte: Importancia de caracteristicas  
Buscar las caracteristicas más relevantes tanto por arboles como por el método de barajar una características y evaluar los errores (crear función).   
Al final definir las más importantes y volver a construir y evaluar el modelo.   

Si sobrara tiempo, estudiar los algoritmos de reducción de características etc.  
> PCA  
> etc.  

## Cuarta parte: Ensembles  

Algoritmos a probar:  
> Random forest.  
> Bagging:
>> Trees  
>> SVM  
>> MLP o Perceptrón  
>> Clasificador lineal (=perceptron??)   
>> _alguna más_   
>  
> AdaBoost o GradientBoosting    
>> Trees  
>> SVM  
>> MLP o Perceptrón  
>> Clasificador lineal (=perceptron??)  
>> _alguna más_    
>  
> ¿¿Mixtura de expertos y stacked generalizado??  

Probarlo si se puede, y mencionar en el trabajo que se probó pero no suponía una mejora. 


## Preguntas:  
> ¿A qué se refiere con lo de validación usando los datos no etiquetados?  
> ¿A qué se refiere con lo de **los diferentes conjuntos de entrenamiento**?  
> ¿Usar el 50% de representantes igual no es lo mejor para algunos algoritmos, para MLP usar todo es mejor que usar la mitad, pero para Kernels si es mejor?   
> 
