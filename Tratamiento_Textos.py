from bs4 import BeautifulSoup
from functions import *
from Clustering import *
import re, os
import nltk


'''
########  PRUEBA PARA EL FUNCIONAMIENTO DE EXTRACCION, TRADUCCION Y OBTENCION DE EN DE UNA PAGINA ################
soup = read_file("Corpus/All 66 on board plane that crashed in Iranian mountains 'believed dead'.html")
extraction = extract_content(soup)


######## Traduccion de una pagina  #############
traduction = translate_goslate(extraction['title'], extraction['body'])
print(' La traduccion es: ', traduction)

########## Obtencion de EN de una pagina  ##############
print(extraction[3]["body"])
tags_sents = extract_tags(extraction['body'])
entities = extract_entities(tags_sents)
print('Longitud entities: ', len(entities))
print(entities)
print('Longitud set entities: ', len(set(entities)))
print(set(entities))

##################  FIN DEL ESTUDIO DE UNA PAGINA ########################
'''


################# Estudio del corpus ###################
# La ejecucion del script proporciona la lectura de los archivos, la extraccion de su contenido,
# la traduccion de cada uno de ellos, la extraccion de Entidades Nombradas y la realizacion del cluster usando como
# vocabulario las EN unicamente.
# Ademas, se proporciona codigo comentado donde se muestran diversas pruebas para la realizacion del cluster que se
# explican en cada uno de ellos. Se aconseja al lector que si se desea descomentar parte del codigo vaya comentado
# el resto. Solo coincide la parte primera de lectura del fichero donde se encuentran los textos, y su extraccion.


######### LECTURA DE CORPUS Y EXTRACCION DE SU CONTENIDO ###########

#read_folder te devuelve un objeto de tipo diccionario donde la clave es el nombre de cada fichero y el valor
#es un objeto BeautifulSoup que nos permitira manejar su contenido posteriormente
texts = read_folder("Corpus")

extraction = []
print('Extrayendo titulos, cuerpo y tags.....')
#A continuacion extramenos los titutlos, el cuerpo y los tags. Esto es asi pues en un primer momento queriamos realizar
#diversas pruebas diferenciando entre estos campos. Sin embargo, no se tendra en cuenta los tags que obtiene pero
#lo hemos mantenido ya que no ocupa mucho espacio y se podria dar un futuro uso
for pag,title in zip(texts.values(), texts):
    title = title.replace('.html', '')
    #extract_content devuelve un diccionario con claves: title, body y tags, y el valor lo obtenido al manipular los
    #articulos con BeautifulSoup. ( La extraccion proporcionara ruido o, en algunos casos, sera vacio ).
    dicc = extract_content(pag)
    #Puesto que no podemos asegurar que la extraccion del titulo sea correcta y ya que sabemos que el titulo es el
    #nombre del fichero, se lo cambiamos por este.
    dicc['title'] = title
    extraction.append(dicc)
    print('.', end=' ')



################# ESTUDIO SOLO CON EN  ####################


#Previo a la extraccion de EN, translate_goslate se encargara de traducir el texto en caso de que este en otro idioma
#distinto al español. Para comprobar el idioma, TextBlob proporciona una funcion para saber en que idioma esta el texto
# que se le pasa. Para ahorrar ejecucion, le metemos el texto a traducir y el titulo para que compruebe con este
# en que idioma esta.
#Finalmente, usamos la funcion extract_entities que se encarga de extraer las EN de tipo 'ORGANIZATION',
# 'PERSON', 'LOCATION', 'GPE' que encuentre en los textos traducidos. Las demas EN las categoriza como 'MISCELANEA'.

entities = []
for file in extraction:
    #traducciendo
    file['body'] = translate_goslate(file['title'], file['body'])
    #Obteniendo tags y entidades nombradas
    en = extract_entities(extract_tags(file['body']))
    #Añadiendo a vector entities
    entities.append(en)

#Calculo de matriz con frecuencia de cada EN en un texto.
vectores = vector_cluster(entities)

#Uso la funcion cluster_text para realizar la clusterizacion. Proporciona el indice de aciertos obtenido. Mas certero
#conforme mas cercano al 1 se encuentre.
rand = cluster_texts(vectores, 5)

print("Rand score: ", rand)
#Podemos ver que se equivoca en 3

#################  FIN DE ESTUDIO ###################


'''
#################   ESTUDIO CON EN SIN TRADUCIR #######################

#Se hace lo mismo que en el procedimiento anterior solo que no se lleva a cabo la traduccion del texto. 
#Esto podría resultar efectivo igualmente ya que las EN en su mayoria no varian (en nuestro caso) de un idioma a otro.
#Sin embargo, hemos encontrado un problema con el corpus que permite obtener las EN en espanol. 

entities = []
for file in extraction:
    #Obteniendo tags y entidades nombradas
    enti = extract_entities(extract_tags(file['body']))
    #Añadiendo
    entities.append(enti)


#Calculo de matriz con frecuencia de cada EN en un texto.
vectores = vector_cluster(entities)

rand = cluster_texts(vectores,5)

print("Rand score: ", rand)

#################  FIN DE ESTUDIO ###################
'''

'''
################  ESTUDIO CON LEMATIZACION DE PALABRAS SIN EN  #########################

#Aqui obtenemos los tags de todas las palabras y las lematizamos previa traduccion.

all_word = []
for file in extraction:
    #traducciendo
    file['body'] = translate_goslate(file['title'], file['body'])
    #Obteniendo tags
    tags = extract_tags(file['body'])
    lem = []
    for tuples in tags:
        #Lematizando
        lem.extend(lemmatize(tuples))
    #Añadiendo
    all_word.append(filter_stopwords(remove_punctuation(lem)))



#Calculo de matriz con frecuencia de cada EN en un texto.
vectores = vector_cluster(all_word)

rand = cluster_texts(vectores,5)

print("Rand score: ", rand)

#################  FIN DE ESTUDIO ###################
'''

'''
################  ESTUDIO CON LEMATIZACION DE PALABRAS CON EN   #########################

#Unimos la obtencion de palabras lematizadas con las EN

all_word = []
for file in extraction:
    #traducciendo
    file['body'] = translate_goslate(file['title'], file['body'])
    #Obteniendo tags y entidades nombradas
    tags = extract_tags(file['body'])
    lem = []
    for tuples in tags:
        lem.extend(lemmatize(tuples))

    clean_word = filter_stopwords(remove_punctuation(lem))
    entities = extract_entities(tags)
    clean_word.extend(entities)
    all_word.append(clean_word)


#Calculo de matriz con frecuencia de cada EN en un texto.
vectores = vector_cluster(all_word)

rand = cluster_texts(vectores,5)

print("Rand score: ", rand)

#################  FIN DE ESTUDIO ###################

'''

'''
######### ESTUDIO DE CLUSTER PROPIPO: COMPARACION ENTRE TITULOS #############

#A continuacion, se realiza un cluster que tiene en cuenta la semajeanza entre dos titulos, comprobando uno con
#el resto de los ya intoducidos al cluster. El mas parecido entra en su grupo. Si no se parece a ninguno crea un
#nuevo grupo.
#Para ello traducimos primero los titulos, y extraemos las EN. Si no encuentra ninguna le dejamos el titulo limpio. 
#(sin stopword, signos de putacion, etc.

titles = []
for file in extraction:
    title = translate_title(file['title'])
    print(title)
    EN = extract_EN(extract_tags([title]))
    clean_title = remove_punctuation(filter_stopwords(nltk.word_tokenize(title)))
    if EN != []:
        title = EN
        title.extend(clean_title)
    else:
        title = clean_title
    print('EN de titles: ', title)
    titles.append(title)

test = cluster_title(titles)
print(test)

reference = [0, 5, 0, 0, 0, 2, 2, 2, 3, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 0, 2, 5]
print('Indices originales: ')
print(reference)

print("rand_score cluster with EN: ", adjusted_rand_score(reference,test))

##################  FIN DEL ESTUDIO  ######################
'''




#Comparacion entre dos títulos PRUEBA
#A continuacion se presenta el codigo para la comparacion de dos titulos, medida para el cluster propio que relaciona
#dos articulos en funcion de la semanjanza entre sus titulos. Esta comparacion a su vez se mide con el porcentaje
#de similaridad (palabra por palabra)

'''
t1 = 'Iran Plane Crash Leads to Search-and-Rescue Effort at 14,500 Feet - The New York Times'
token1 = remove_punctuation(filter_stopwords(nltk.word_tokenize(t1)))
print(token1)

t2 = 'Iran plane crash_ Agonising wait continues for relatives - BBC News'
token2 = remove_punctuation(filter_stopwords(nltk.word_tokenize(t2)))
print(token2)

t3 = 'The far-left separatists who took Catalonia to the brink'
token3 = remove_punctuation(filter_stopwords(nltk.word_tokenize(t3)))
print(token3)


comparation(token1,token2)
comparation(token1,token3)
comparation(token2,token3)
#print('Y la comparacion es: ', d, '\n De longitud: ', len(d))
'''

'''
print(titles[0],'+++++++')
print(titles[1],'+++++++')

print(comparation(titles[0],titles[1]))
print(titles[2],'+++++++')
print(comparation(titles[0],titles[2]))
print(titles[3],'+++++++')
print(comparation(titles[0],titles[3]))
print(comparation(titles[2],titles[3]))
print(comparation(titles[0],titles[1]))
print(comparation(titles[1],titles[11]))
'''