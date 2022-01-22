# Trabajo_Aprendizaje_Maquina


# TRABAJO FINAL DE LABORATORIO 
## TRABAJO A1 
### Arturo Sirvent Fresneda y Ángel Guevara Ros

## OBJETIVO 

El propósito del trabajo es crear una aplicación de clasificación de eventos sonoros  partiendo de un conjunto de muestras etiquetadas con sus clases correspondientes. 

Este tipo de problemas se caracteriza por la alta dimensionalidad y variabilidad presente en las clases, así como en la dificultad de disponer de muestras convenientemente etiquetadas.

Intentaremos abordar el problema tanto de forma supervisada como no supervisada, simulando que no se conocen algunas o todas las etiquetas del conjunto de datos.

## MÓDULOS


```python
# módulos básicos
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import sys, os, warnings

# módulos para manejar las muestras
import librosa, librosa.display
import IPython.display as ipd

import sklearn 

#datasets
from sklearn.datasets import make_blobs

# clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# clasificación
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,LinearSVC

from sklearn.neighbors import   NearestCentroid, \
                                KNeighborsClassifier, \
                                KernelDensity


# métricas
from sklearn.metrics import silhouette_samples, adjusted_rand_score,accuracy_score,f1_score, \
                            mutual_info_score, normalized_mutual_info_score,precision_score, \
                            adjusted_mutual_info_score,recall_score,cohen_kappa_score,  \
                            homogeneity_score, completeness_score, v_measure_score, \
                            fowlkes_mallows_score,silhouette_score, calinski_harabasz_score,\
                            make_scorer,roc_auc_score, confusion_matrix
                            
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler

#crossval y parameter estimation
from sklearn.model_selection import KFold,cross_validate, train_test_split, cross_val_score, \
                                    GridSearchCV

# importancia de características
from sklearn.inspection import permutation_importance



```


```python
#añadimos esta cell para eliminar del informe final algunas advertencias que lanza SKlearn,
#esto no es recomendable hacerlo mientras se trabaja en el notebook, solo para obtener un 
#resultado final libre de advertencias que se han decidido ignorar.
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
```

## FUNCIONES 

En este apartado, definiremos diferentes funciones que vamos a usar a lo largo del trabajo.


```python
def transformacion_pca(data, srate, ncomp):
    Xf = librosa.feature.mfcc(y=data[0,:],
                        sr=srate,n_mfcc=20).flatten()    
    for s in range(1,n):
        Mfcc = librosa.feature.mfcc(y=data[s,:],
                        sr=srate,n_mfcc=20).flatten()    
        Xf = np.vstack([Xf,Mfcc])

    pca = sklearn.decomposition.PCA(n_components=ncomp)    
    pca.fit(Xf)
    X = pca.transform(Xf)
    return X,Xf

```


```python
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
```


```python
def get_closer_points(medias,data,n):
    # a esta funcion le pasamos las posiciones de los puntos y de las medias,
    # y nos devuelve un vector logico diciendo que n puntos son los mas cercanos a las medias
    #las distancias entre las medias y los puntos son 
    distancias=cdist(medias,data)

    #y queremos solo nos n puntos mas cercanos a cualquiera de estas
    #primero los juntamos pero guardando donde deben ser separados
    leng=distancias.shape[1]
    aux_dists=np.concatenate(distancias,axis=0)
    indices_aux_dist=np.argsort(aux_dists)
    #ahora queremos un vector de boleanos que nos diga cuales son los que vamos a considerar
    aux_bool=np.full(( indices_aux_dist.shape[-1]), False, dtype=bool)
    aux_bool[indices_aux_dist[:n]]=True
    # y le volvemos a dar la forma
    aux_bool=aux_bool.reshape(-1,leng)

    #por ultimo lo coloreamos escogemos si esta cerca de cualquiera de las medias, 
    #entonces hacermo sentancia or/any para la dimension 1
    points_represent=np.any(aux_bool,axis=0) #el axis este no lo entiendo...
    return points_represent


def train_and_get_stats(model,X,y,cv=5,train=True):
    #vamos a hacer una funcion que nos reciba el modelo, los datos y algunos parametros más, y nos evalue todo esto
    #asi nos lo devuelve y ya lo tenemos para meterlo en el dataframe

    #esta funcion va a tomar el modelo, lo va a entrenar y va a darnos las metricas,
    #pero no va a hacer nada sobre la busqueda de parametros optimos
        
    
    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
        
    #calculamos en train y test de forma normal con los conjuntos train y test
    if train:
        model.fit(X_train,y_train)
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    
    #calculamos las metricas que nos hacen falta
    aux1=pd.Series({"Accuracy_train":accuracy_score(y_train,y_pred_train),
                    "Accuracy_test":accuracy_score(y_test,y_pred_test),
                    "cohen-kappa_train":cohen_kappa_score(y_train,y_pred_train),
                    "cohen-kappa_test":cohen_kappa_score(y_test,y_pred_test),
                    "confusion_matrix":confusion_matrix(y_test,y_pred_test),
                    "auc_roc":roc_auc_score(y_test,model.predict_proba(X_test),multi_class="ovr")})
                    

    #Tambien vamos a usar cross validation para obtener una estimacion más correcta de los parametros
    #prec = make_scorer(precision_score,greater_is_better=True, average="micro")
    #recall = make_scorer(recall_score,greater_is_better=True, average="micro")
    #f1 = make_scorer(f1_score,greater_is_better=True, average="micro")


    metrics_aux=cross_validate(model, X,y,cv=cv,
                               scoring={"accuracy":"accuracy"},#,"precision":prec,"recall":recall,"f1":f1},
                               return_train_score=False)
    #obtenemos un diccionario y listas de las estadisticas para cada una de las evaluaciones en los k folds
    #ahora para guardarlos hacemos la media de estas
    
    aux2=pd.Series({"Accuracy_kfold":np.mean(metrics_aux["test_accuracy"])})

    return pd.concat([aux1,aux2])

def print_resultados(df):
    for i in df.index:
        aux=resultados.iloc[i].to_dict()
        print(f"Para el algoritmo {aux['Algoritmo']} tenemos: ")
        print(f"OA train: {round(aux['Accuracy_train'],3)}")
        print(f"OA test: {round(aux['Accuracy_test'],3)}")
        print(f"Kappa Cohen train: {round(aux['cohen-kappa_train'],3)}")
        print(f"Kappa Cohen test: {round(aux['cohen-kappa_test'],3)}")
        print(f"auc roc: {round(aux['auc_roc'],3)}")
        print(f"Y la matriz de confusion: \n {aux['confusion_matrix']}")
        print("\n ========================================= \n")
        
```

## PREPROCESADO DE DATOS

Aquí, importaremos y prepararemos los datos correspondientes a la tarea A1. Nuestros datos son señales de audio que contienen sonidos de animales. En particular, hay seis tipos de animales posibles y tendremos una cantidad balanceada de muestras de cada clase. 

A continuación, cargamos los datos y echamos un vistazo a las dimensiones de los mismos.


```python
class_names = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat']
srate = 22050        # frecuencia de muestreo
c = len(class_names) # número de clases

# cargamos los datos
data = np.load('./Datasets_A/adata1.npy')

n,d = data.shape  # size and (raw) dimension
nc = n//c         # samples per class

print('%d samples from %d classes'%(n,c))
print('Raw dimension is %d corresponding to %.1f seconds of audio'%(d,d/srate))

# creamos las etiquetas
labs = np.int16(np.kron(np.arange(c),np.ones(nc)))

```

    720 samples from 6 classes
    Raw dimension is 66150 corresponding to 3.0 seconds of audio


Observamos que tenemos 720 muestras de cada una de las seis clases. Además cada muestra tienen un total de 66150 características que se obtienen de los tres segundos de duración de la señal de audio. 

Podemos escuchar una de las muestras para comprobar que existe cierto ruido en ellas que dificultará la clasificación. 


```python
arow = np.random.randint(n)

sig = data[arow,:]

ipd.Audio(sig, rate=srate)
```





<audio  controls="controls" >
    <source src="data:audio/wav;base64,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" type="audio/wav" />
    Your browser does not support the audio element.
</audio>




A la hora de trabajar con los datos, tenemos varias opciones: 

* Trabajar en el dominio temporal.

* Usar Espectrogramas de Mel.  

* Usar los Coeﬁcientes Cepstrales en las Frecuencias de Mel. 

Una vez elegido el espacio en el que queremos trabajar, es conveniente hacer una reducción de la dimensionalidad por PCA. 

En particular, nosotros hemos elegido trabajar con los Coeficientes Cepstrales en las Frecuencias de Mel. Una vez que hemos transformado nuestros datos a dicho espacio, llevamos a cabo una reducción de la dimensionalidad mediante PCA quedándonos con el número de componentes necesario para explicar el 99% de la varianza. 

La elección de trabajar en este espacio se debe a que la reducción de características es notable y además no estamos perdiendo información importante. Este último aspecto se puede comprobar si tomamos por ejemplo un clasificador lineal y calculamos el acierto usando todas las características y luego usando las características reducidas. 



```python
wc = LinearDiscriminantAnalysis()

X, Xf = transformacion_pca(data,srate,ncomp=.99)   # Transformamos los datos al espacio nuevo y reducimos con PCA
y = labs


print('Dimension: %d -> %d'%(Xf.shape[1],X.shape[1])) 
print('Resub. acc of a linear class.: %.2f -> %.2f'%(wc.fit(Xf,y).score(Xf,y),wc.fit(X,y).score(X,y)))
```

    Dimension: 2600 -> 464
    Resub. acc of a linear class.: 0.93 -> 0.94


Como se puede observar, hemos reducido el número de características hasta 464 e incluso hemos mejorado el acierto utilizando un clasificador lineal. Por tanto, de aquí en adelante, trabajaremos con los datos transformados a dicho espacio.

## CLUSTERING

En este apartado, nos planteamos un primer análisis de los datos. No usaremos las etiquetas y el objetivo será encontrar posibles patrones en nuestros datos y ver la compatibilidad con las etiquetas. 

Usaremos varios algoritmos de clustering para tener distintos puntos de vista. Para ver si los clusters son compatibles con las etiquetas reales del problema (que las conocemos), calcularemos algunas métricas como por ejemplo: 

* **Índice de Rand ajustado:** tiene en cuenta en lo que están de acuerdo y en lo que no están de acuerdo dos agrupamientos. Varía entre -1 y 1. Índices cercanos a cero quieren decir que son agrupamientos aleatorios.

* **Medida V:** compara la homogeneidad y la completitud entre dos agrupamientos. Varía entre 0 y 1, siendo 1 un parecido idéntico.

* **Índice de Silhouette:** nos da información sobre la dispersión de los clusters. Varía entre -1 y 1, y cuánto más grande sea, mejor será el agrupamiento hecho. Valores cercanos a cero indican solapamiento entre clusters.

### Clustering Jerárquico Aglomerativo

En este tipo de algoritmos, comenzamos con un cluster para cada muestra y vamos agrupando los dos clusters más parecidos en cada iteración hasta que llegamos a tener un único cluster para todo el conjunto de datos. Para caracterizar el parecido entre clusters, necesitamos la noción de distancia. 

El clustering aglomerativo es más popular que el divisivo debido a su eficiencia. Aún así, es un algoritmo costoso ya que en cada iteración tenemos que calcular la distancia con el nuevo cluster formado. 

La representación típica del clustering jerárquico es mediante dendrogramas, en los que en el eje Y medimos la distancia entre clusters.

A continuación, aplicamos un algoritmo de clustering aglomerativo a nuestro conjunto de datos. No le fijaremos el número de clusters para que me haga el proceso hasta el final. 

Una vez hecho el clustering, dibujaremos su dendrograma y lo comentaremos.


```python
# Fijamos `distances_threshold=0` para que se calcule todo el árbol
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward' )   

# Lo entrenamos con nuestros datos
model = model.fit(X)

plt.title("Dendrograma del Clustering Jerárquico")
# fijando p=3 mostramos los tres primeros níveles del dendrograma.
plot_dendrogram(model, truncate_mode="level", p=3)  
plt.axhline(y=5000,linestyle="--",color="k",linewidth=1)
plt.axhline(y=4000,linestyle="--",color="k",linewidth=1)
plt.axhline(y=3000,linestyle="--",color="k",linewidth=1)
plt.xlabel("Número de puntos por nodo (o índice del punto si no hay paréntesis)")
plt.xticks(rotation = 45)
plt.show()
```


    
![png](imagenes_md/output_24_0.png)
    


Como hemos dicho antes, en el eje Y está representada la distancia entre clusters. A nosotros nos va a interesar que está distancia sea suficientemente grande ya que eso indicará que la partición hecha es estable. En cambio, cuando veamos un salto pequeño en el eje Y, eso indicará que la partición no es tan estable. 

Así, observando el dendrograma podríamos tomar varias decisiones: 

* Por un lado, cortar por la línea horizontal de 5000. Esto nos daría tres clusters, dos de ellos balanceados y uno con menor número de muestras. Estos clusters están lo suficientemente alejados para afirmar que no tienen patrones en común con el resto. 

* Por otro lado, podríamos decidir cortar por la línea horizontal de 4000. Esto supondría que ahora tendríamos cinco clusters con menos muestras cada uno.

* Por último, podríamos cortar por la línea horizontal de 3000, lo que nos daría una división en seis clusters.

Cualquiera de las tres decisiones puede ser buena, sin embargo hay factores que nos hace decidirnos por una. Si nos fijamos en el final del dendrograma, tenemos que la unión entre los grupos (105) y (131) se hace muy arriba. Esto quiere decir que la distancia entre ambos grupos es grande y, por tanto podría tener sentido que fueran lo suficientemente diferentes. Así, podríamos descartar el agrupamiento en tres clusters. 

Ya solo nos quedarían el agrupamiento en cinco y en seis clusters. El factor decisivo en este caso es el número de elementos en cada grupo. Si nos quedamos con el de seis clusters, entonces tendríamos dos clusters (los de la izquierda) con muy pocos elementos en comparación con el resto de grupos. 

Por tanto, viendo el dendrograma nosotros diríamos que a priori puede haber cinco grupos distintos en nuestros datos. 


Para comparar la información obtenida con las etiquetas reales del problema, lo que haremos será volver a hacer un clustering jerárquico pero ahora fijando el número de clusters a seis. Echaremos un vistazo a las etiquetas predichas y usaremos las métricas definidas para obtener algunos resultados cuantitativos. 


```python
model = AgglomerativeClustering(n_clusters=6, linkage='ward' )   

model = model.fit(X)
y_Aggl = model.labels_

print('Indice de Rand Ajustado: %.2f '%adjusted_rand_score(y,y_Aggl))
print('Medida V: %.2f '% v_measure_score(y,y_Aggl))
print('Indice de Silhouette: %.2f '%silhouette_score(X, y_Aggl))
```

    Indice de Rand Ajustado: 0.15 
    Medida V: 0.26 
    Indice de Silhouette: 0.13 


Observamos que los resultados obtenidos no son muy buenos. El índice de Rand ajustado junto con la medida V nos dicen que no hay mucho parecido entre las etiquetas predichas y las reales. 

El índice de Silhouette confirma lo mismo que las otras medidas y además como es un valor cercano a cero, nos indica que puede existir solapamiento entre los diferentes clusters. 

El solapamiento entre clusters hace más complicado este primer análisis sin las etiquetas.

### Clustering basado en densidad. MeanShift

En este tipo de métodos nos fijamos en la acumulación de puntos. Lo que buscamos son zonas densas dónde haya muchas muestras juntas y, dentro de las densas, nos interesarán las más densas posibles. 

Así, estos métodos consisten en elegir una región del espacio, calcular la media (o la moda) de los puntos que caigan dentro y moverme hacia ella. Con esto conseguimos movernos hacia una región más densa. Si repetimos esto de forma iterativa, acabaremos alcanzando la zona más densa de mi nube de puntos. 

Para seleccionar la región del espacio, se usan lo que se conoce como ventanas. Al mirar por la ventana lo que estamos haciendo es calcular la media ponderada de los puntos de esa región.

Precisamente en esto consiste el algoritmo **Meanshift**. Este método nos asegura la convergencia hacia un punto estacionario. Los puntos del espacio que converjan al mismo centro, pertenecerán al mismo cluster. 

Cabe destacar, que el algoritmo **Meanshift** supone que hay una gradación de la densidad de nuestros puntos. Si esto no ocurre, el algoritmo no funcionará correctamente. 

Uno de los hiperparámetros del algoritmo **Meanshift** es la anchura de la ventana. Si cogemos una ventana muy grande, puede que estemos teniendo en cuenta puntos muy alejados a la hora de calcular la media y entonces ésta nos salga desvíada hacia una región poco densa. Si cogemos una anchura de ventana muy pequeña, el tiempo de computación puede aumentar considerablemente y además obtener demasiados clusters. 

Existe una función que nos permite estimar este parámetro y será la que utilicemos. 


```python
# Estimación de la anchura de ventana

bandwidth = estimate_bandwidth(X, quantile=0.3,random_state=0)

print("Estimated bandwidth=%.2f"%bandwidth)
```

    Estimated bandwidth=687.78


Procedemos a utilizar el algoritmo **MeanShift**.


```python
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, 
               cluster_all=True).fit(X)

y_ms = ms.labels_
cluster_centers = ms.cluster_centers_
y_ms_unique = np.unique(y_ms)
n_clusters_ = len(y_ms_unique ) - (1 if -1 in y_ms else 0)

print('Bandwidth: ' , bandwidth)
print("number of estimated clusters : %d" % n_clusters_)
print('Labels: ' , set(y_ms))
```

    Bandwidth:  687.7780598958333
    number of estimated clusters : 3
    Labels:  {0, 1, 2}


Observamos que con la anchura de ventana estimada y nuestros datos, el resultado del MeanShift es que hay tres grupos solamente en nuestros datos. Sin embargo, nosotros sabemos que en realidad hay seis clusters correspondientes a cada una de las clases del problema, por lo que el resultado obtenido no es muy bueno. 

La respuesta que nos da el algoritmo MeanShift se puede deber a que existe solapamiento entre los clusters. Por tanto, no es capaz de distinguir seis regiones densas distintas, sino que esas zonas se entremezclan dando lugar a solo tres regiones. 

Para poder comparar de forma cuantitativa con las etiquetas reales del problema, vamos a jugar con el ancho de ventana para que el resultado del clustering sea de seis grupos. Por ejemplo, fijando `bandwidth=650` obtenemos dicho resultado.


```python
bandwidth=650
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, 
               cluster_all=True).fit(X)

y_ms2 = ms.labels_
cluster_centers2 = ms.cluster_centers_
y_ms_unique2 = np.unique(y_ms2)
n_clusters_2 = len(y_ms_unique2 ) - (1 if -1 in y_ms2 else 0)

print('Bandwidth: ' , bandwidth)
print("number of estimated clusters : %d" % n_clusters_2)
print('Labels: ' , set(y_ms2))
```

    Bandwidth:  650
    number of estimated clusters : 6
    Labels:  {0, 1, 2, 3, 4, 5}


Para comparar con las etiquetas reales, usamos las mismas métricas que hemos presentado al inicio del apartado. 


```python
print('Indice de Rand Ajustado: %.2f '%adjusted_rand_score(y,y_ms2))
print('Medida V: %.2f '% v_measure_score(y,y_ms2))
print('Indice de Silhouette: %.2f '%silhouette_score(X, y_ms2))
```

    Indice de Rand Ajustado: 0.01 
    Medida V: 0.11 
    Indice de Silhouette: 0.31 


El índice de Rand ajustado y la medida V nos indican que hay poca similitud en general y en cuanto a homogeneidad y completitud. 

El índice de Silhouette sin embargo, es algo mayor pero tampoco es muy bueno. Se aleja un poco del cero, lo que quiere decir que no existe tanto solapamiento entre los clusters calculados.  

Viendo las tres métricas calculadas, concluimos que tampoco es un clustering que sea compatible con las etiquetas reales.

### Conclusiones

Viendo los resultados de los dos algoritmos de clustering llevados a cabo, podemos concluir que éstos no son compatibles con las etiquetas reales en las que existen seis grupos de datos. Esto se puede deber a diversos factores: 

* Existencia de ruido en las muestras que dificulta la labor del agrupamiento. 

* Muestras que contengan atributos erróneos. 

* Solapamiento entre clusters. Probablemente producido por la cantidad de características de cada muestra que tenemos.  

* Falta de estructura en los clusteres debido a la alta dimensionalidad de los datos.  

Estos últimos puntos lo abordaremos más adelante en el trabajo. Por eso, cuando hagamos una selección de las características más importantes, reduciendo así la dimensión, volveremos a probar los algoritmos de clustering para ver si mejoran los resultados. 

## ALGORITMOS SUPERVISADOS

### Primera parte: Estudio de los métodos para la reducción de datos.  

En el enunciado se pide que reduzcamos la cantidad de datos disponibles mediante alguna técnica de aprendizaje no supervisado, para así obtener un 50% compuesto de respresentantes de cada clase.   
Así pues, en este apartado vamos a estudiar como se puede realizar la reducción del conjunto de entrenamiento al 50% de la mejor manera posible.  
Las opciones que se barajan son:  
* Uso de KMeans para localizar representantes de cada clase mediante cercania a los centroides.  
* Uso de KMedoids para lo mismo, pero ahora los representantes son directamente elementos del conjunto.  
* Uso de gausianas, buscando los puntos más cercanos a los centros de las gausianas ajustadas.  

También contamos con dos estrategias:  
1. Podemos seleccionar tantos centroides (o medioides o gaussianas) como representantes queremos obtener (el 50% del total).  
2. Podemos intenter ajustar los datos a un número óptimo de clusters, y seleccionar los puntos con menos distancia a estas posiciones centrales.  

Primero investigaremos cual de estas dos estrategias parece tener mejores resultados, usando unos datos simulados. 


```python
#vamos a hacer un ejemplo sencillo con ambas estrategias 
#para los tres métodos (KMeans, Kmedoids, Gaussians)

#primero creamos los datos 
n=600
data=make_blobs(n,centers=1,cluster_std=1.5)


#primero para KMeans
fig=plt.figure(figsize=(24,6))
n_clusters=[300,1]
titles=["Tantos centros como representantes.","Numero correcto de centros y \n representantes por ceracanía."]

for j,i in enumerate(n_clusters):
    plt.subplot(1,4,j+1)
    modelmeans=KMeans(n_clusters=i)
    modelmeans.fit(data[0])

    points_represent=get_closer_points(modelmeans.cluster_centers_,data[0],n=300)
    plt.scatter(data[0][:,0],data[0][:,1],c=points_represent,s=8)
    plt.plot(modelmeans.cluster_centers_[:,0],modelmeans.cluster_centers_[:,1],"+k")
    plt.title(titles[j],fontsize=14)
    plt.xticks([]);
    plt.yticks([]);
plt.suptitle("Usando KMeans",fontsize=16,x=0.25)
plt.tight_layout()

#segundo para las Gaussianas
fig=plt.figure(figsize=(24,6))
n_clusters=[300,1]
titles=["Tantos centros como representantes.","Numero correcto de centros y \n representantes por ceracanía."]

for j,i in enumerate(n_clusters):
    plt.subplot(1,4,j+1)
    modelgaus=GaussianMixture(n_components=i)
    modelgaus.fit(data[0])
    points_represent=get_closer_points(modelgaus.means_,data[0],n=300)
    plt.scatter(data[0][:,0],data[0][:,1],c=points_represent,s=8)
    plt.plot(modelgaus.means_[:,0],modelgaus.means_[:,1],"+k")
    plt.title(titles[j],fontsize=14)
    plt.xticks([]);
    plt.yticks([]);
plt.suptitle("Usando Gaussianas",fontsize=16,x=0.25)
plt.tight_layout()

#Tercero para Kmedoids
fig=plt.figure(figsize=(24,6))
n_clusters=[300,1]
titles=["Tantos centros como representantes.","Numero correcto de centros y \n representantes por ceracanía."]
#primer subplot
plt.subplot(1,4,1)
modelmedoids=KMedoids(n_clusters=300)
modelmedoids.fit(data[0])
color_aux=np.full(data[0].shape[0],False,dtype=bool)
color_aux[modelmedoids.medoid_indices_]=True
plt.scatter(data[0][:,0],data[0][:,1],c=color_aux,s=8)
plt.plot(modelmedoids.cluster_centers_[:,0],modelmedoids.cluster_centers_[:,1],"+k")
plt.title(titles[0],fontsize=14)
plt.xticks([]);
plt.yticks([]);
#segundo subplot
plt.subplot(1,4,2)
modelmedoids=KMedoids(n_clusters=1)
modelmedoids.fit(data[0])
points_represent=get_closer_points(modelmedoids.cluster_centers_,data[0],n=300)
plt.scatter(data[0][:,0],data[0][:,1],c=points_represent,s=8)
plt.plot(modelmedoids.cluster_centers_[:,0],modelmedoids.cluster_centers_[:,1],"+k")
plt.title(titles[1],fontsize=14)
plt.xticks([]);
plt.yticks([]);

plt.suptitle("Usando Kmedoids",fontsize=16,x=0.25)
plt.tight_layout()
```


    
![png](imagenes_md/output_45_0.png)
    



    
![png](imagenes_md/output_45_1.png)
    



    
![png](imagenes_md/output_45_2.png)
    


Las gráficas anteriores ejemplifican perfectamente el problema de intentar ajustar tantas gaussianas o centroides como representantes se desea obtener. El problema es que dichos centros se separarán entre si para no solapar, y esto hace que no obtengamos representantes sino que obtengamos un muestreo de los datos que solo atiende a pequeñas agrupaciones de puntos o quizá ni eso en muchos casos. Los representantes que obtenemos cuando escogemos de forma acertada el número de clusters, son muchos más adecuados, a excepción de los medoides cuyo resultado final es bastantes similar.  

Sin embargo nuestro problema cuenta con una dimensionalidad altísima y no podemos realizar un estudio visual como el anterior, para determinar el número correcto de clusters. Además los clusters debido a la llamada _"maldición de la dimensionalidad"_ estarán muy repartidos y posiblemente no tengan una forma muy compacta.   
A este respecto valoramos dos opciones:  
1. Hacer pruebas sobre el conjunto completo de datos y probar distintos valores para el número de clusters, y medir la eficiencia con los estadísticos y coeficientes adecuados.  
2. Para cada clase realizar clustering para los valores de esa etiqueta, encontrar el centro (centroide, medoide o media según el algoritmo) y seleccionar la mitad de las muestras más cercanas a dicho centro para cada clase individualmente.  

Nos parece que la segunda opción es la que puede dar mejores resultados (usando KMeans), de modo que procedemos con ella.   




```python
#elección de representantes, la eleccion la haremos mediante kmeans
clases=np.unique(y).shape[0]

X_reduc=[]
y_reduc=[]

for i in range(clases):
    #la eleccion de los puntos mas cercanos al centroide la hacemos para cada clase
    len_aux=len(y[y==i])
    model=KMeans(n_clusters=1, random_state=0)
    model.fit(X[y==i])  
    points_represent=get_closer_points(model.cluster_centers_,X[y==i],n=len_aux//2)
    X_reduc.append(X[y==i][points_represent])
    y_reduc.extend(np.ones(sum(points_represent))*i)
X_reduc=np.array(X_reduc)
y_reduc=np.array(y_reduc)
X_reduc2=X_reduc.reshape(-1,464)


#separamos los datos en conjuntos para entrenamiento y para test
#lo hacemos tanto para el conjunto completo como  para el reducido por representantes
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_reduc_train,X_reduc_test,y_reduc_train,y_reduc_test=train_test_split(X_reduc2,y_reduc,
                                                                       test_size=0.2,random_state=0)
  
```

### Segunda parte: Construccion de los algoritmos.  
Vamos a intentar resolver el problema mediante algoritmos de aprendizaje supervisado.  
A primera vista, el algoritmo que parece más indicado para la tarea es el de **máquinas de vector soporte** para clasificación, pues contamos con una dimensionalidad muy alta y (tras la reducción al 50%) la cantidad de ejemplo para entrenar el modelo no es muy grande.  
Los modelos que vamos a probar son:  
- Arboles  
- MLP  
- SVM  
- Vecinos próximos.  
 

**Nota:** Vamos a hacer uso de validación cruzada para la selección de hiperparámetros, más concretamente del algoritmo k-folds con $k=5$. Durante esta selección de hiperparámetros se calculará solo la _accuracy_, y al final, de los mejores modelos, calcularemos _auc roc_, _kappa_cohen_, _confusion matrix_ y _overall accuracy_.

#### Arboles 

Para el caso de los árboles además de hacer tuning de los hiperparámetros, también tenemos la opción de la **post-poda** lo cual nos podría resultar en un arbol simplificado que de mejores resultados.  
Para proceder primero escogeremos los hiperparámetros que funcionen mejor con **pre-poda** y como paso final, probaremos si la **post-poda** mejora los resultados.


```python
%%time
#gridsearch
tree_params={"criterion":["gini","entropy"],"max_depth":np.arange(2,50,2)}

#lo evaluamos 

model_tree=DecisionTreeClassifier(random_state=0)

tree_best=GridSearchCV(model_tree,tree_params)
tree_best.fit(X_reduc_train,y_reduc_train);
print("La mejor precision (",tree_best.best_score_,") ha sido obtenida con los parametros: "
      ,tree_best.best_params_)
```

    La mejor precision ( 0.46890502117362365 ) ha sido obtenida con los parametros:  {'criterion': 'gini', 'max_depth': 4}
    CPU times: user 24.7 s, sys: 35.7 ms, total: 24.8 s
    Wall time: 24.9 s


Podemos visualizar el aprendizaje de los arboles ploteando su precisión en la predicción en función de la profundidad del arbol.  
Y también vamos a estudiar como mejora (o empeora) el modelo conforme hacemos la **post-poda**.  


```python
plt.figure(figsize=(13,8))

#primero los entrenamos en pre-poda

acc_train_list=[]
acc_test_list=[]
depths=list(range(1,13))
for i in depths:
    model=DecisionTreeClassifier(criterion="gini",max_depth=i,random_state=0)
    model.fit(X_reduc_train,y_reduc_train)
    y_pred_1=model.predict(X_reduc_train)
    acc_train_list.append(accuracy_score(y_reduc_train,y_pred_1))
    y_pred_2=model.predict(X_reduc_test)
    acc_test_list.append(accuracy_score(y_reduc_test,y_pred_2))

plt.title("Arbol entrenado con pre- y post- poda y diferentes profundidades (criterio: gini).",fontsize=14)
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.plot(depths,acc_train_list,"r",label="Pre-prunning accuracy on train")
plt.plot(depths,acc_test_list,"b",label="Pre-prunning accuracy on test")

#ahora hacemos la post-poda
aux=model.cost_complexity_pruning_path(X_reduc_train,y_reduc_train)
ccp,imps=aux.ccp_alphas, aux.impurities
acc_train_list_pruned=[]
acc_test_list_pruned=[]
node_counts=[]
models=[]
for i in ccp:
    model=DecisionTreeClassifier(ccp_alpha=i,random_state=0)
    model.fit(X_reduc_train,y_reduc_train)
    models.append(model)
    y_pred_1=model.predict(X_reduc_train)
    node_counts.append(model.get_depth())
    acc_train_list_pruned.append(accuracy_score(y_reduc_train,y_pred_1))
    y_pred_2=model.predict(X_reduc_test)
    acc_test_list_pruned.append(accuracy_score(y_reduc_test,y_pred_2))

plt.plot(node_counts,acc_test_list_pruned,"b--",label="Post-prunning accuracy on test")
plt.plot(node_counts,acc_train_list_pruned,"r--",label="Post-prunning accuracy on train")
plt.legend()
plt.grid()

print("Max acc on test data pre-prunning: %.3f " % max(acc_test_list))
print("Max acc on test data post-prunning: %.3f" % max(acc_test_list_pruned))



```

    Max acc on test data pre-prunning: 0.611 
    Max acc on test data post-prunning: 0.625



    
![png](imagenes_md/output_52_1.png)
    


En la gráfica anterior vemos como el mejor resultado (en el conjunto de test) se da con una profundidad de 7, y tras aplicar la **post-poda**. Por tanto, vemos que aplicar post-poda resulta en una mejora del modelo en este caso.


```python
indice = np.where(acc_test_list_pruned==max(acc_test_list_pruned))
indice[0]    
# Mejores aciertos en los índices 46 y 47
```




    array([46, 47])




```python
tree_best_prunned = models[46]
tree_best_prunned
```




    DecisionTreeClassifier(ccp_alpha=0.01785078347578348, random_state=0)



#### MLP 
Si usamos un perceptron multicapa, este debería ser entrenado con todos los datos disponibles, y no vemos motivo alguno a limitarnos usando solo unos cuantos representantes de cada clase. Esto es discutido más adelante, donde ambos cojuntos son usados (reducido y sin reducir) para el entrenamiento. 


```python
%%time
#usando el conjunto reducido de datos


X_reduc_train_norm=(X_reduc_train-X_reduc_train.mean())/np.std(X_reduc_train)
X_reduc_test_norm=(X_reduc_test-X_reduc_test.mean())/np.std(X_reduc_test)

#normalizando los datos de entrada, pero con el set reducido
mlp_params={"hidden_layer_sizes":[(4,5),(2,3)],
            "alpha":np.geomspace(25,45,5),"solver":["lbfgs","adam"]}


model_mlp=MLPClassifier(max_iter=2000,verbose=False,random_state=0)
mlp_best=GridSearchCV(model_mlp,mlp_params)
mlp_best.fit(X_reduc_train_norm,y_reduc_train)
print("La mejor precision (",mlp_best.best_score_,") ha sido obtenida con los parametros: "
      ,mlp_best.best_params_)
print("La precisión del mejor modelo en el conjunto de test es: ",
      accuracy_score(y_reduc_test,mlp_best.best_estimator_.predict(X_reduc_test_norm)))
```

    La mejor precision ( 0.5765275257108288 ) ha sido obtenida con los parametros:  {'alpha': 33.54101966249685, 'hidden_layer_sizes': (4, 5), 'solver': 'lbfgs'}
    La precisión del mejor modelo en el conjunto de test es:  0.5416666666666666
    CPU times: user 4min 55s, sys: 5.06 s, total: 5min
    Wall time: 2min 35s



```python
%%time 
# sin usar la reducción de datos


X_train_norm=(X_train-X_train.mean())/np.std(X_train)
X_test_norm=(X_test-X_test.mean())/np.std(X_test)

#normalizando los datos de entrada, y con el set completo
mlp_params={"hidden_layer_sizes":[(4,5),(2,3)],
            "alpha":np.geomspace(25,45,5),"solver":["lbfgs","adam"]}


model_mlp=MLPClassifier(max_iter=2000,verbose=False,random_state=0)
mlp_best=GridSearchCV(model_mlp,mlp_params)
mlp_best.fit(X_train_norm,y_train)
print("La mejor precision (",mlp_best.best_score_,") ha sido obtenida con los parametros: "
      ,mlp_best.best_params_)
print("La precisión del mejor modelo en el conjunto de test es: ",
      accuracy_score(y_test,mlp_best.best_estimator_.predict(X_test_norm)))
```

    La mejor precision ( 0.6231184407796102 ) ha sido obtenida con los parametros:  {'alpha': 38.850300961670285, 'hidden_layer_sizes': (4, 5), 'solver': 'lbfgs'}
    La precisión del mejor modelo en el conjunto de test es:  0.625
    CPU times: user 10min 37s, sys: 8.17 s, total: 10min 45s
    Wall time: 5min 44s


Podemos ver que efectivamente, el modelo entrenado con mayor cantidad de datos a resultado tener una precisión más alta. Limitar el conjunto de entrenamiento a solo unos pocos representantes no ha sido de ayuda en este caso, en el caso de las redes es mejor tener cuantos más datos mejor. También supuso una ventaja normalizar los datos de entrada de la red.

#### SVM  
Tal como mencionábamos al comienzo de este apartado de aprendizaje supervisado, este algoritmo a primera vista parece ser el más adecuado para el problema que se nos plantea (por la alta dimensionalidad y la relativa escasez de datos). 


```python
%%time
#gridsearch
gamma_scale=1/(X_reduc_train.shape[-1]*np.var(X_reduc_train.flatten()))
svm_params={"C":np.linspace(1,1.4,30),"gamma":np.linspace(gamma_scale*0.5,gamma_scale*2,30)}


model_svm=SVC(random_state=0,probability=True)
svm_best=GridSearchCV(model_svm,svm_params)
svm_best.fit(X_reduc_train,y_reduc_train)
print("La mejor precision (",svm_best.best_score_,") ha sido obtenida con los parametros: "
      ,svm_best.best_params_)
print("Los extremos para el rango de gamma son: ",gamma_scale*0.5,gamma_scale*2)
```

    La mejor precision ( 0.6666666666666666 ) ha sido obtenida con los parametros:  {'C': 1.2206896551724138, 'gamma': 7.947431338301957e-06}
    Los extremos para el rango de gamma son:  2.4260579874816497e-06 9.704231949926599e-06
    CPU times: user 7min 20s, sys: 185 ms, total: 7min 20s
    Wall time: 7min 22s


#### Vecinos próximos  
Este algoritmo si bien es de clustering, lo hemos incluido en este apartado pues requiere de un conjunto de entrenamiento para la clasificación del resto de elementos según los k vecinos más próximos a cada punto del que se desea predecir la clase.   


```python
%%time
nearestn_params={"n_neighbors":[1,2,3],"weights":["uniform", "distance"],
                 "algorithm":["ball_tree", "kd_tree", "brute"]} #metric:??


model_nearestn=KNeighborsClassifier()
nearestn_best=GridSearchCV(model_nearestn,nearestn_params)
nearestn_best.fit(X_reduc_train,y_reduc_train)
print("La mejor precision (",nearestn_best.best_score_,") ha sido obtenida con los parametros: "
      ,nearestn_best.best_params_)
```

    La mejor precision ( 0.48608590441621297 ) ha sido obtenida con los parametros:  {'algorithm': 'ball_tree', 'n_neighbors': 2, 'weights': 'uniform'}
    CPU times: user 1.21 s, sys: 4.06 ms, total: 1.22 s
    Wall time: 1.09 s


**Conclusiones:**  
En lo relativo a algortimos supervisados, hemos observado que los mejores resultados vienen dados por las **máquinas de vector soporte** y los **MLP** entrenados con un conjunto no reducido de datos. 

**Nota:** Todos los hiperparametros probados no son los que se muestran aquí, ha habido pruebas previas que se han descartado por falta de eficiencia, como por ejemplo probar un kernel polinómico, una vez quedó claro que el "rbf" daba mejores resultados, el polinómico ni lo teniamos en cuenta a la hora de ajustar el resto de hiperparámetros.

## IMPORTANCIA DE CARACTERÍSTICAS

En este apartado, vamos a utilizar clasificadores del tipo árboles de decisión para intentar predecir las etiquetas de nuestro conjunto de datos. Usaremos este tipo de clasificadores porque entre sus atributos se encuentra la importancia de características. 

La importancia de las características es una información muy útil porque nos permite saber qué atributos no son relevantes para nuestro modelo. 

Entonces haremos lo siguiente: 

* Primero usaremos el árbol de decisión con los mejores parámetros calculados en el apartado anterior. 

* Depués veremos cuáles son las características más importantes con el atributo `feature_importances_`. 

* Finalmente, eliminaremos todas aquellas características que tengan una importancia de cero. Nos quedaremos con el resto y volveremos a entrenar el árbol de decisión para ver si mejoran (o no empeoran) los resultados. 

Cabe destacar que, en el caso de los árboles de decisión, la importancia de una caracaterística se calcula como la reducción total (normalizada) del criterio de división (normalemente la impureza de Gini) al dividir por dicha característica.

**Nota:** 

Si el clasificador que estamos usando no tiene directamente un atributo que nos de la importancia de las características, podemos usar la función `permutation_importance`. Esta implementación lo que hace es barajar una de las características y volver a entrenar el modelo con dicha característica permutada. Después comprueba si el rendimiento cambia mucho o no. Si cambia mucho, entonces diremos que esa variable es importante, y si cambia poco diremos que no lo es. Esto lo repite para cada una de las características y así nos ofrece un valor cuantitativo de la importancia de cada característica.

También lo probaremos en este apartado usando máquinas de vector soporte.

### Árboles de decisión

Así, vamos a calcular un "ranking" de características para nuestro árbol de decisión. 

Para ello, cogemos el árbol de decisión calculado en el apartado anterior y usamos el atributo de `feature_importances_` para ver la importancia de las características. Nos quedaremos con aquellas cuya importancia sea no nula.


```python
# El árbol que mejores resultados nos daba con nuestros datos.
clf_tree = tree_best_prunned
clf_tree.fit(X_reduc_train,y_reduc_train)
```




    DecisionTreeClassifier(ccp_alpha=0.01785078347578348, random_state=0)



Calculamos una estimación del acierto en entrenamiento y en test mediante cross-validation. 


```python
print('OA train %0.2f' % np.mean(cross_val_score(clf_tree, X_reduc_train,y_reduc_train, scoring='accuracy', cv=5)))
print('OA test %0.2f' % np.mean(cross_val_score(clf_tree, X_reduc_test,y_reduc_test, scoring='accuracy', cv=5)))
```

    OA train 0.42
    OA test 0.36


Cálculo de las importancias.


```python
importancias=clf_tree.feature_importances_
# print(importancias)           descomentar para ver el resultado exacto de la importancia de cada característica.      
```

Selección de características.


```python
indices=np.where(importancias!=0)
X_reduc2_2=X_reduc2[:,indices]
print(X_reduc2_2.shape)
X_reduc2_2= X_reduc2_2.reshape((360,len(indices[0])))
print(X_reduc2_2.shape)
```

    (360, 1, 9)
    (360, 9)



```python
indices  # Características seleccionadas
```




    (array([  0,   1,   2,   4,   5,  10,  33, 116, 221]),)



Tras calcular la importancia de las características, observamos que tan solo hay nueve cuya importancia sea no nula. Por tanto, nos quedamos solo con esas características y volvemos a evaluar el rendimiento del clasificador.


```python
Xtr, Xts, ytr, yts = train_test_split(X_reduc2_2, y_reduc, test_size=0.2, shuffle=True, random_state=0)

# El árbol que mejores resultados nos daba con nuestros datos.
clf_tree = tree_best_prunned
clf_tree.fit(Xtr,ytr);
```


```python
print('OA train %0.2f' % np.mean(cross_val_score(clf_tree, Xtr, ytr, scoring='accuracy', cv=3)))
print('OA test %0.2f' % np.mean(cross_val_score(clf_tree, Xts, yts, scoring='accuracy', cv=3)))
```

    OA train 0.56
    OA test 0.57


Observamos que tras eliminar la gran mayoría de las características, los aciertos en test y en entrenamiento han mejorado. Esto se debe a que tan solo había unas pocas características relevantes para nuestro árbol de clasificación que son las que estaban tomando realmente las decisiones, y el resto de atributos no participaba.

### SVM

Vamos a hacer también una selección de características para el caso en el que estamos clasificando con una máquina de vector soporte. Lo separamos del caso anterior porque las características que son importantes para un método puede que no lo sean para otro. 

De hecho, el clasificador de las máquinas de vector soporte (SVC) no tiene un atributo como tal que nos dé la importancia de las características, sino que tenemos que usar una función auxiliar (`permutation_importance`). 

Así, cogemos de nuevo el SVC con los parámetros óptimos calculado en el apartado anterior y trabajamos sobre él.


```python
clf_svc = svm_best.best_estimator_
clf_svc.fit(X_reduc_train,y_reduc_train)


print('OA train %0.2f' % np.mean(cross_val_score(clf_svc,X_reduc_train,y_reduc_train, scoring='accuracy', cv=5)))
print('OA test %0.2f' % np.mean(cross_val_score(clf_svc, X_reduc_test,y_reduc_test, scoring='accuracy', cv=5)))
```

    OA train 0.67
    OA test 0.47


Los resultados no son malos. Calculamos la importancia de las características usando  `permutation_importance` y nos quedamos solo con las relevantes. Volvemos a evaluar el modelo para ver si mejora (o no empeora).


```python
resultado = permutation_importance(clf_svc, X_reduc2, y_reduc,scoring='accuracy', n_repeats=5, random_state=0)
```


```python
importancias_svc = resultado.importances_mean
#importancias_svc  #descomentar para ver los valores exactos de importancia de cada característica.
```

Selección de características.


```python
indices2=np.where(importancias_svc > abs(1e-2))
X_reduc3=X_reduc2[:,indices2]
print(X_reduc3.shape)
X_reduc3= X_reduc3.reshape((360,len(indices2[0])))
print(X_reduc3.shape)
```

    (360, 1, 4)
    (360, 4)



```python
indices2  
```




    (array([0, 1, 2, 3]),)



Una vez hecha la selección de las características más importantes, obtenemos cuatro (tres de las cuales coinciden con las anteriores obtenidas por los árboles), volvemos a entrenar el modelo con dichas características. Una vez entrenado, calculamos la estimación de aciertos para ver si no empeora con respecto al modelo con todas las características.


```python
Xtr, Xts, ytr, yts = train_test_split(X_reduc3, y_reduc, test_size=0.2, shuffle=True, random_state=0)

clf_svc = svm_best.best_estimator_
clf_svc.fit(Xtr,ytr)


print('OA train %0.2f' % np.mean(cross_val_score(clf_svc, Xtr, ytr, scoring='accuracy', cv=5)))
print('OA test %0.2f' % np.mean(cross_val_score(clf_svc, Xts, yts, scoring='accuracy', cv=5)))

```

    OA train 0.52
    OA test 0.58


Es sorprendente que al eliminar características (dejando solo cuatro), los resultados en test mejoran y el modelo se simplifica significativamente. Es cierto que en el entrenamiento baja un poco el acierto, pero no es preocupante. Además, a nosotros nos interesa que en test vaya bien para que el modelo sea capaz de generalizar. 

El hecho de que el acierto en test no varie mucho al eliminar gran cantidad de las características, se debe a que el resto de características no contenían informacion relevante, y el modelo generaliza bien.

Cabe destacar también que los aciertos están bastante balanceados, por lo que no debemos preocuparnos por el sobreajuste.

### Clustering

Como comentamos en el apartado de clustering, una vez hecha la reducción de características puede ser interesante volver a realizar algún algoritmo de agrupamiento para ver si los resultados mejoran al haber menos dimensiones. 

Aquí simplemente repetimos lo hecho anteriormente pero sobre el conjunto que contiene solo las características importantes.

#### Clustering Jerárquico Aglomerativo

Volvemos a hacer clustering aglomerativo imponiendo que me calcule seis clusters. Además, calculamos los mismo índices que en el primer apartado para comparar con las etiquetas reales y ver si ha habido una mejora significativa.


```python
model = AgglomerativeClustering(n_clusters=6, linkage='ward' )   

model = model.fit(X_reduc2_2)
y_Aggl2 = model.labels_

print('Indice de Rand Ajustado: %.2f '%adjusted_rand_score(y_reduc,y_Aggl2))
print('Medida V: %.2f '% v_measure_score(y_reduc,y_Aggl2))
print('Indice de Silhouette: %.2f '%silhouette_score(X_reduc2, y_Aggl2))
```

    Indice de Rand Ajustado: 0.15 
    Medida V: 0.30 
    Indice de Silhouette: 0.16 


A la vista de los resultados, no parece que haya mejorado mucho el resultado del clustering. Por tanto, parece que la dimensionalidad en este caso no era tan importante. 

Que los resultados no hayan cambiado mucho puede deberse a que las características más importantes siguen estando solapadas entre sí, lo que dificulta mucho la labor de buscar patrones y establecer grupos.

#### MeanShift

Comprobamos si con el clustering basado en densidad conseguimos mejorar los resultados.


```python
# Estimación de la anchura de ventana

bandwidth = estimate_bandwidth(X_reduc2_2, quantile=0.3,random_state=0)

print("Estimated bandwidth=%.2f"%bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, 
               cluster_all=True).fit(X_reduc2_2)

y_ms = ms.labels_
cluster_centers = ms.cluster_centers_
y_ms_unique = np.unique(y_ms)
n_clusters_ = len(y_ms_unique ) - (1 if -1 in y_ms else 0)

print('Bandwidth: ' , bandwidth)
print("number of estimated clusters : %d" % n_clusters_)
print('Labels: ' , set(y_ms))
```

    Estimated bandwidth=246.48
    Bandwidth:  246.47625403697955
    number of estimated clusters : 7
    Labels:  {0, 1, 2, 3, 4, 5, 6}


En este caso, vemos como el agrupamiento propuesto con los parámetros estimados tampoco parece acertado en cuánto al número de grupos. Sin embargo, se acerca más a nuestras etiquetas, lo que es una mejora con respecto al resultado que obteníamos en el primer apartado, donde el algoritmo MeanShift nos daba una predicción de tres grupos. 

Calculemos algunos de los índices para ver si acertamos asignando las etiquetas a los puntos. Para ello, hacemos un MeanShift con el ancho de banda adecuado para que nos salga un agrupamiento en seis clusters.  



```python
bandwidth = 250
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, 
               cluster_all=True).fit(X_reduc2_2)

y_ms = ms.labels_
cluster_centers = ms.cluster_centers_
y_ms_unique = np.unique(y_ms)
n_clusters_ = len(y_ms_unique ) - (1 if -1 in y_ms else 0)

print('Bandwidth: ' , bandwidth)
print("number of estimated clusters : %d" % n_clusters_)
print('Labels: ' , set(y_ms))
```

    Bandwidth:  250
    number of estimated clusters : 6
    Labels:  {0, 1, 2, 3, 4, 5}



```python
print('Indice de Rand Ajustado: %.2f '%adjusted_rand_score(y_reduc,y_ms))
print('Medida V: %.2f '% v_measure_score(y_reduc,y_ms))
print('Indice de Silhouette: %.2f '%silhouette_score(X_reduc2, y_ms))
```

    Indice de Rand Ajustado: 0.10 
    Medida V: 0.29 
    Indice de Silhouette: 0.28 


Además, tanto el índice de Rand como la medida V han sufrido un incremento en comparación al clustering que hacíamos con todas las características. No son resultados buenos tampoco, pero mejores que los que ya teníamos. 

El índice de Silhouette se mantiene más o menos igual. 


Por tanto, en el caso del clustering basado en densidad hemos comprobado que una reducción de la dimensionalidad sí que mejora los resultados. Sobre todo en cuánto a la organización interna de los datos, ya que es capaz de detectar que existen siete grupos distintos entre nuestras muestras, cosa que se acerca a ser compatible con nuestras etiquetas.


### Algoritmos de clasificación que hacen uso de Ensembles  
En este apartado se probarán los algoritmos que tienen como piezas primarias _"weak learners"_, pero es el conjunto lo que permite que se contruya un predictor completo y funcional.  
Los modelos que se probarán son:  
- Random Forests. 
- Bagging.   

Para entrenar ambos modelos no solo usaremos el conjunto reducido de datos (50% representantes de cada clase), sino que solo usaremos las características más relevantes.

Como se indica en el enunciado de la tarea, para el entrenamiento de los modelos que hacen uso de ensembles, usaremos solo los cojuntos reducidos tanto en características (las nueve características más relevantes según el ranking de importancia dado por los árboles) como en número datos.


```python
Xtr, Xts, ytr, yts = train_test_split(X_reduc2_2, y_reduc, test_size=0.2, shuffle=True, random_state=0)
```

#### Random Forests


```python
#gridsearch

forest_params={"n_estimators":np.rint(np.linspace(270,290,10)).astype(int),
               "max_depth":[2,3,4,5,6]}#"criterion":["gini","entropy"]

model_forest=RandomForestClassifier(random_state=0)
forest_best=GridSearchCV(model_forest,forest_params)
forest_best.fit(Xtr,ytr)
print("La mejor precision (",forest_best.best_score_,") ha sido obtenida con los parametros: "
      ,forest_best.best_params_)
```

    La mejor precision ( 0.5901996370235935 ) ha sido obtenida con los parametros:  {'max_depth': 5, 'n_estimators': 283}


#### Bagging


```python
bagging_params={"base_estimator":[SVC(random_state=0),DecisionTreeClassifier(random_state=0)],#MLPClassifier(40),
                "n_estimators":[4,12,16,26,40]}#:np.rint(np.geomspace(10,30,5)).astype(int)},
                #"max_features":[0.3,0.6,1.0]}
model_bagging=BaggingClassifier()
bagging_best=GridSearchCV(model_bagging,bagging_params)
bagging_best.fit(Xtr,ytr)
print("La mejor precision (",bagging_best.best_score_,") ha sido obtenida con los parametros: "
      ,bagging_best.best_params_)
```

    La mejor precision ( 0.6282516636418632 ) ha sido obtenida con los parametros:  {'base_estimator': DecisionTreeClassifier(random_state=0), 'n_estimators': 26}


Recordemos que la diferencia entre bagging con árboles como predictores débiles y los random forest, es que los random forest solo consideran un subconjunto aleatorio de características en cada _split_ del los árboles, mientras que bagging considera todas ellas en cada elección.

Para el uso de ensembles hemos obtenido unos mejores resultados en el caso del Bagging, usando como weak learner los árboles de decisión. En este caso además hemos necesitado una cantidad muchísimo menor de estimadores (componentes del ensemble).

### Estadísticos y coeficientes para los mejores ajustes de cada modelo.   
Ahora vamos a sacar varios estadisticos de cada uno de los mejores modelos ajustados anteriormente. 


```python
resultados=pd.DataFrame(columns=["Algoritmo","Accuracy_train","Accuracy_test",
                                 "Accuracy_kfold","cohen-kappa_train","cohen-kappa_test","confusion_matrix",
                                 "auc_roc"])



names=["Tree","SVC","MLP","KNN","Forest","Bagging"]

for i,j in enumerate([tree_best_prunned,svm_best.best_estimator_,
                      mlp_best.best_estimator_,nearestn_best.best_estimator_,forest_best.best_estimator_,
                      bagging_best.best_estimator_]):
    aux_series=train_and_get_stats(j,X_reduc2_2,y_reduc,cv=5,train=True)
    aux_series=aux_series.append(pd.Series({"Algoritmo":names[i]}))
    resultados=resultados.append(aux_series,ignore_index=True)
resultados
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Accuracy_train</th>
      <th>Accuracy_test</th>
      <th>Accuracy_kfold</th>
      <th>cohen-kappa_train</th>
      <th>cohen-kappa_test</th>
      <th>confusion_matrix</th>
      <th>auc_roc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tree</td>
      <td>0.711806</td>
      <td>0.625000</td>
      <td>0.530556</td>
      <td>0.652931</td>
      <td>0.548747</td>
      <td>[[3, 1, 1, 1, 3, 3], [1, 13, 0, 0, 0, 1], [2, ...</td>
      <td>0.836262</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVC</td>
      <td>0.656250</td>
      <td>0.638889</td>
      <td>0.591667</td>
      <td>0.584815</td>
      <td>0.570839</td>
      <td>[[6, 0, 0, 0, 2, 4], [1, 10, 0, 0, 0, 4], [2, ...</td>
      <td>0.913487</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MLP</td>
      <td>0.628472</td>
      <td>0.597222</td>
      <td>0.583333</td>
      <td>0.553178</td>
      <td>0.513287</td>
      <td>[[4, 1, 0, 1, 3, 3], [0, 12, 1, 0, 0, 2], [1, ...</td>
      <td>0.870686</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNN</td>
      <td>0.781250</td>
      <td>0.611111</td>
      <td>0.541667</td>
      <td>0.737819</td>
      <td>0.529082</td>
      <td>[[10, 0, 0, 0, 0, 2], [5, 10, 0, 0, 0, 0], [2,...</td>
      <td>0.845822</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Forest</td>
      <td>0.819444</td>
      <td>0.625000</td>
      <td>0.597222</td>
      <td>0.782322</td>
      <td>0.551557</td>
      <td>[[5, 0, 0, 1, 3, 3], [2, 10, 0, 0, 0, 3], [3, ...</td>
      <td>0.905194</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bagging</td>
      <td>1.000000</td>
      <td>0.583333</td>
      <td>0.597222</td>
      <td>1.000000</td>
      <td>0.496855</td>
      <td>[[4, 2, 0, 2, 3, 1], [3, 11, 0, 0, 0, 1], [2, ...</td>
      <td>0.876996</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_resultados(resultados)
```

    Para el algoritmo Tree tenemos: 
    OA train: 0.712
    OA test: 0.625
    Kappa Cohen train: 0.653
    Kappa Cohen test: 0.549
    auc roc: 0.836
    Y la matriz de confusion: 
     [[ 3  1  1  1  3  3]
     [ 1 13  0  0  0  1]
     [ 2  0  9  1  0  0]
     [ 2  0  4  8  1  0]
     [ 2  1  0  0  7  2]
     [ 0  1  0  0  0  5]]
    
     ========================================= 
    
    Para el algoritmo SVC tenemos: 
    OA train: 0.656
    OA test: 0.639
    Kappa Cohen train: 0.585
    Kappa Cohen test: 0.571
    auc roc: 0.913
    Y la matriz de confusion: 
     [[ 6  0  0  0  2  4]
     [ 1 10  0  0  0  4]
     [ 2  0  7  3  0  0]
     [ 2  0  3  9  0  1]
     [ 1  0  0  0  8  3]
     [ 0  0  0  0  0  6]]
    
     ========================================= 
    
    Para el algoritmo MLP tenemos: 
    OA train: 0.628
    OA test: 0.597
    Kappa Cohen train: 0.553
    Kappa Cohen test: 0.513
    auc roc: 0.871
    Y la matriz de confusion: 
     [[ 4  1  0  1  3  3]
     [ 0 12  1  0  0  2]
     [ 1  0  4  7  0  0]
     [ 2  0  2 11  0  0]
     [ 1  1  1  0  7  2]
     [ 0  1  0  0  0  5]]
    
     ========================================= 
    
    Para el algoritmo KNN tenemos: 
    OA train: 0.781
    OA test: 0.611
    Kappa Cohen train: 0.738
    Kappa Cohen test: 0.529
    auc roc: 0.846
    Y la matriz de confusion: 
     [[10  0  0  0  0  2]
     [ 5 10  0  0  0  0]
     [ 2  1  8  1  0  0]
     [ 1  1  5  7  0  1]
     [ 2  0  0  0  9  1]
     [ 3  1  0  0  2  0]]
    
     ========================================= 
    
    Para el algoritmo Forest tenemos: 
    OA train: 0.819
    OA test: 0.625
    Kappa Cohen train: 0.782
    Kappa Cohen test: 0.552
    auc roc: 0.905
    Y la matriz de confusion: 
     [[ 5  0  0  1  3  3]
     [ 2 10  0  0  0  3]
     [ 3  0  8  1  0  0]
     [ 0  0  5  9  0  1]
     [ 1  1  0  0  8  2]
     [ 0  1  0  0  0  5]]
    
     ========================================= 
    
    Para el algoritmo Bagging tenemos: 
    OA train: 1.0
    OA test: 0.583
    Kappa Cohen train: 1.0
    Kappa Cohen test: 0.497
    auc roc: 0.877
    Y la matriz de confusion: 
     [[ 4  2  0  2  3  1]
     [ 3 11  0  0  0  1]
     [ 2  0  8  2  0  0]
     [ 2  0  4  7  1  1]
     [ 1  1  0  0  8  2]
     [ 0  2  0  0  0  4]]
    
     ========================================= 
    


## Conclusiones

Tras abordar el problema de clasificación de señales de audio tanto de forma supervisada como no supervisada, y probando diferentes algoritmos, llegamos a las siguientes conclusiones. 

* Las técnicas de clustering utilizadas sobre el conjunto inicial de datos sin las etiquetas no son de gran ayuda. Esto se debe a la gran dimensionalidad de las características y a que existe cierto solapamiento entre grupos según hemos visto con algunos índices como el de Silhouette. 

* Para aplicar algoritmos supervisados, hemos reducido previamente el conjunto de datos al 50%. Esto se ha hecho utilizando el algoritmo K-Means con un número de clusters fijado previamente, y después cogiendo los puntos cercanos a los centroides de cada cluster. 

* Una vez reducido el conjunto, decidimos aplicar árboles de decisión (con poda y sin poda), Máquinas de Vector Soporte, Redes Neuronales y Vecinos Próximos. Para la elección de los hiperparámetros de cada clasficador, hemos hecho una búsqueda en un posible rango de valores para cada uno, y hemos determinado cuál de las posibles combinaciones de parámetros nos daba el mejor resultado para quedarnos con ella. Tras varios experimentos, nos quedamos con dos clasificadores: 

    **1)** Un árbol de decisión resultante de aplicar la post-poda. Se ha elegido dicho árbol porque se ha visto que era el que daba el mejor rendimiento durante la fase de test. 
    
    **2)** Un clasificador tipo SVM que también nos daba buenos resultados en comparación con el resto. Además, teníamos cierta predilección por este tipo de clasificadores porque sabíamos que suelen funcionar bien cuando se trata de datos con una gran cantidad de características. 
    
* Hemos decidido Normalizar los datos solo en el caso de las Redes Neuronales ya que en estos casos es epecialmente importante para que haya estabilidad en los pesos. Cuando usamos Redes Neuronales, si no normalizamos los pesos puede ocurrir que estos se disparen y crezcan mucho o disminuyan mucho. De hecho, cuando tenemos muchas capas, es común introducir una capa solamente de normalización.

* Con los clasificadores escogidos del apartado anterior, hemos hecho un estudio de la importancia de las características para ver si realmente nos hacen falta todas o si podemos eliminar algunas y quedarnos solo con las más relevantes. Esto puede hacer que disminuya la dimensionalidad del problema y, por tanto, hacerlo más sencillo. Este ha sido nuestro caso, en el que hemos conseguido eliminar gran parte de las características que no estaban aportando nada a los modelos. 
    
    **1)** En el caso de los árboles de decisión, éstos tienen implementado directamente un atributo que me calcula la importancia de las características. Haciendo uso de él, reducimos las variables hasta quedarnos solo con nueve. 
    
    **2)** Para las SVM, utilizamos una función auxiliar para calcular la importancia. En este caso, fijando un umbral elegido por nosotros, nos quedamos solo con cuatro características. 

* Una vez hecha la reducción de las características, hemos llevado a cabo de nuevo técnicas de clustering para ver si mejoraban los resultados obtenidos al inicio del trabajo. Hemos comprobado que no hay una gran mejora en cuánto al acierto al asignar las etiquetas (índices de comparación de clusters), sin embargo sí que parece haber un progreso en el hecho de predecir el número de grupos que hay en nuestros datos. Aunque seguimos sin acertar, por lo menos somos capaces de distinguir un número de grupos cercano al real. 

* Finalmente, se han probado diferentes técnicas de ensembles con el conjunto de datos con las características reducidas. Estabamos en buen lugar para aplicar este tipo de algoritmos, ya que los clasificadores obtenidos no nos daban grandes rendimientos (*weak-learners*), y es en esos casos cuando juntar varios predictores débiles puede resultar en uno mejor. En particular, hemos utilizado Random Forest y Bagging (tanto con árboles como con SVM), dando el segundo de ellos mejores resultados. 

* Observando el cuadro resumen de métricas, si tuvieramos que elegir un modelo para predecir, escogeriamos las **máquinas de vector soporte** por los buenos resultados mostrados en todas la métricas de evaluación usadas.  

* De las matrices de confusión podemos ver como en la mayoría de casos, los errores vienen de la primera clase, o bien porque no se clasifican bien los audios de la primera clase, o bien porque audios que no son de esta, son clasificados como de la primera clase. Como trabajo futuro para la mejora de estos modelos, propondríamos que se realice un filtrado y preprocesado mas exaustivo y controlado de las señales "raw", antes de realizar la primera extracción de características, y también podría mejorar los resultados estudiar más en profundidad los errores observados en la matriz de confusión, y en caso de existir mucha ambigüedad entre clases, proceder con una "fusión" de clases, en la que dos audios que no son distinguibles por nuestros métodos, terminan clasificándose en una sola clase.
