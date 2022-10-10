# EntrenamientoCancerDannyAvalos
#IDENTIFICAR SI UN PACIENTE TIENO O NO TIENE CANCER

Objetivo:
El proyecto tiene como objetivo crear un clasificador de aprendizaje automático que, utilizando los parámetros dados del conjunto de datos, 
prediga si la célula es maligna (cancerosa) o benigna (no cancerosa).

Descripción del conjunto de datos
data set utiilizado: wdbc.data

El conjunto de datos contiene 569 filas y 32 columnas. Algunas de las columnas se describen a continuación.
	*id — Este número de identificación. se asigna a cada paciente y es único.
	*diagnóstico: esta sería nuestra variable objetivo, 'M' significa tumor maligno (canceroso) y 'B' significa tumor benigno (no canceroso)
	*radio — La distancia desde el centro hasta el perímetro de la celda
	*textura — La desviación estándar de los valores de la escala de grises**
	*perímetro_mean — Media del perímetro
	*area_mean — Media del área de la celda
	*smoothness — La variación local en las longitudes de los radios
	*concavity — La severidad de los patrones cóncavos en el contorno
	*simetría
	*dimensión fractal

Librerias usadas:
	import numpy as np # linear algebra
	import pandas as pd # data processing, CSV file
	import matplotlib.pyplot as plt # Se usa para trazar el grafico 
	import seaborn as sns # utilizado para trazar gráfico interactivo
	%matplotlib inline
	from sklearn.linear_model import LogisticRegression # para usar el modelo de regresión
	from sklearn.model_selection import train_test_split # para dividir los datos en entrenamieto y pruebas
	from sklearn.model_selection import GridSearchCV# parametro de ajuste
	from sklearn.ensemble import RandomForestClassifier # para genrear modelo de clasificación
	from sklearn.naive_bayes import GaussianNB # para genrear modelo de GaussianNB
	from sklearn.neighbors import KNeighborsClassifier # para genrear modelo de KNN
	from sklearn.tree import DecisionTreeClassifier # para genrear modelo de decisión
	from sklearn import svm # vectores de soporte
	from sklearn import metrics # comprobar el error y precision del modelo
	from sklearn.preprocessing import StandardScaler #estándariza los datos
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import confusion_matrix#mattriz de confusión
	from sklearn.svm import SVC # vectores de soporte
	from sklearn.decomposition import PCA
	from dtreeviz.trees import dtreeviz # visualizar arbolde nos ayuda a la grafica
	from xgboost import XGBClassifier # para genrear modelo XGBClassifier
	from sklearn.metrics import classification_report

Proecedimiento:

	* Lestura de archivo con el cual realizaremos el aprendizaje
	* El archivo no contiene cabecera se procede con la asignación de los nombres para cada columnas
	* Lectura de archivo
	* Información de cada columna
	* validación si existe NULL en alguna columana
	* validación de duplicados
	* lecuta de campo diagnosis
		*Se puede observar que el 62,7% (357 de 569) de las personas presentaban tumores Benignos (no cancerosos) y el 37,3% (212 de 569) presentaban tumores Malignos (Cancerosos).
	* transformación de la columna diagnosis de String a Int M(maligno)=1 y B(Benignos)=1
	* realizamos un drop(borrado) de la columna ID ya que esta no sirve para realizar el estudio
	* se realiza un describe() para identificar por columana:
		* count	
		* mean	
		* std
		* min
		* 25%
		* 50%
		* 75%
		* max
	* Se segmenta el dataset en base al tipo (mean, se, worst) para poder trabajar con ciertas varaiables

GRAFICA DE DATOS:
	* Diagrama de barras que identifica la correlacipon entre diagnosis con cada uno de las colomunas
	* Diagrama de barras que identifica la correlacipon entre diagnosis con cada uno de las colomunas que sobrepasan el 0.6 de afectación
	* Diagrama de matriz de correlación donde  muestra la correlación entre las diferentes variables y el objetivo.
	
USA DE UN UNIVERSO REDUCIDO:
	*	Se asigna una variable con los campos que se quiere trabajar en este caso las columnas de media(mean)
	*	Identificamos el valor del Array
GRAFICA DE DATOS UNIVERSO REDUCIDO:
	* Gráfico de conteo en relación (B o M)	
	* Gráfico de pares de todas algunas de las características relevantes que visualiza la relación entre ellas
	* Diagrama de matriz de correlación donde  muestra la correlación entre las diferentes variables y el objetivo.
	* Diagrama de distribución conjunta entre dos variables x="radius_mean",y="texture_mean"
	* Diagrama de distribución conjunta entre dos variables x="radius_mean",y="perimeter_mean"
	* Diagrama de las características categóricas, es decir, B (benigno) y M (canceroso) se asignan de 0 a 1 respectivamente.

DECLARACIÓN DE VARIABLES DE ENTRENAMIENTO:
	* Declaración de de variable Objetivo(X), variables de entrada(Y)
	* Declaración de de variables para Train y Test

ENTRENAMIENTO DE MODELOS
	La estandarización transforma los datos para que tengan una media de cero y una desviación estándar de 1, lo que mejora el rendimiento del modelo lineal. 
	Modelos como Regresión logística, Máquina de vectores de soporte, K Vecinos más cercanos muestran rendimientos mejorados del modelo, mientras que el modelo 
	basado en árbol y los métodos de conjunto no requieren que se realice el escalado de características, ya que no son sensibles a la variación en los datos.
	
	* Modelo de regresión: Exactitud de modelo de Regression: 92.98245614035088
		* Mapa de matriz de confusión en base al modelo de regresión
	* Modelo con maquina de vectores:Precisión utilizando la máquina de vectores de soporte: 92.10526315789474
	* Modelo de KNN: Precisión utilizando modelo KNN: 92.10526315789474
	* Modelo de arbol de decisión: Precisión utilizando modelo Decision Tree: 92.98245614035088
		*	Gráfico de arbol de decisión con el uso de dtreeviz
	* Modelo de random Forest: Precisión de modelo Random Forest: 94.73684210526315
	* Modelo de arbol de XGBoost: Precisión de modelo XGBoost Classifier: 97.36842105263158
	
Analisis de resultados:
	* usando un Marco de datos identificamos de mayor a menor el mejor modelo
	* Reporte de clasificación

#IDENTIFY IF A PATIENT HAS CANCER OR NOT

Objective:
The project aims to create a machine learning classifier that, using the given parameters of the dataset,
predict whether the cell is malignant (cancerous) or benign (not cancerous).

Description of the data set
data set used: wdbc.data

The data set contains 569 rows and 32 columns. Some of the columns are described below.
*id — This identification number. it is assigned to each patient and is unique.
*diagnosis: this would be our target variable, 'M' means malignant (cancerous) tumor and 'B' means benign (non-cancerous) tumor
*radius — The distance from the center to the perimeter of the cell
*texture — The standard deviation of grayscale values**
*perimeter_mean — Mean of perimeter
*area_mean — Mean of cell area
*smoothness — The local variation in spoke lengths
*concavity — The severity of the concave patterns in the contour
*symmetry
*fractal dimension

Used libraries:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file
import matplotlib.pyplot as plt # Used to plot the plot
import seaborn as sns # used to plot interactive graph
%matplotlib inline
from sklearn.linear_model import LogisticRegression # to use the regression model
from sklearn.model_selection import train_test_split # to split the data into training and testing
from sklearn.model_selection import GridSearchCV# setting parameter
from sklearn.ensemble import RandomForestClassifier # to generate classification model
from sklearn.naive_bayes import GaussianNB # to generate GaussianNB model
from sklearn.neighbors import KNeighborsClassifier # to generate KNN model
from sklearn.tree import DecisionTreeClassifier # to generate decision model
from sklearn import svm # support vectors
from sklearn import metrics # check model error and accuracy
from sklearn.preprocessing import StandardScaler #standardizes the data
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix#confusion matrix
from sklearn.svm import SVC # support vectors
from sklearn.decomposition import PCA
from dtreeviz.trees import dtreeviz # visualize tree helps us to graph
from xgboost import XGBClassifier # to generate XGBClassifier model
from sklearn.metrics import classification_report

Procedure:

* File reading with which we will carry out the learning
* The file does not contain a header, we proceed with the assignment of the names for each column
* Read file
* Information for each column
* validation if there is NULL in any column
* duplicate validation
* diagnostic field reading
*It can be seen that 62.7% (357 of 569) of people had Benign (non-cancerous) tumors and 37.3% (212 of 569) had Malignant (Cancerous) tumors.
* transformation of the diagnosis column from String to Int M(malignant)=1 and B(Benign)=1
* we make a drop (deletion) of the ID column since it is not useful to carry out the study
* a describe() is performed to identify by column:
* count
* mean
*std
* minutes
* 25%
* fifty%
* 75%
*max
* The dataset is segmented based on the type (mean, se, worst) to be able to work with certain variables

DATA GRAPH:
* Bar chart that identifies the correlation between diagnoses with each of the columns
* Bar chart that identifies the correlation between diagnoses with each of the colons that exceed 0.6 of involvement
* Correlation matrix diagram where it shows the correlation between the different variables and the objective.

USE OF A REDUCED UNIVERSE:
* A variable is assigned with the fields that you want to work with, in this case the mean columns (mean)
* We identify the value of the Array
REDUCED UNIVERSE DATA GRAPH:
* Graph of count in relation (B or M)
* Pair chart of all some of the relevant features visualizing the relationship between them
* Correlation matrix diagram where it shows the correlation between the different variables and the objective.
* Joint distribution diagram between two variables x="radius_mean",y="texture_mean"
* Joint distribution diagram between two variables x="radius_mean",y="perimeter_mean"
* Plot of categorical features, ie B (benign) and M (cancerous) are assigned 0 to 1 respectively.

STATEMENT OF TRAINING VARIABLES:
* Declaration of variable Target(X), input variables(Y)
* Declaration of variables for Train and Test
MODEL TRAINING
Standardization transforms the data to have a mean of zero and a standard deviation of 1, which improves the performance of the linear model.
Models like Logistic Regression, Support Vector Machine, K Nearest Neighbors show improved performances of the model, while the model
Tree-based and ensemble methods do not require feature scaling to be performed, as they are not sensitive to variation in the data.

* Regression model: Regression model accuracy: 92.98245614035088
* Confusion matrix map based on the regression model
* Model with vector machine: Precision using support vector machine: 92.10526315789474
* KNN model: Precision using KNN model: 92.10526315789474
* Decision tree model: Accuracy using Decision Tree model: 92.98245614035088
* Decision tree plot with the use of dtreeviz
* Random Forest model: Random Forest model precision: 94.73684210526315
* XGBoost Tree Model: XGBoost Classifier Model Accuracy: 97.36842105263158

Analysis of results:
* using a Data Frame we identify from highest to lowest the best model
* Classification report
