from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from textblob import TextBlob
from bs4 import BeautifulSoup
import re, os
import nltk
import string


def read_file(file):
    print("......")
    try:
        with open(file,'r', encoding="utf-8") as f:
            pag = f.read()
        f.close()
    except UnicodeDecodeError:
        with open(file,'r', encoding="latin-1") as f:
            pag = f.read()
        f.close()
    bsObj = BeautifulSoup(pag, "html.parser")
    return(bsObj)

def read_folder(folder):
    # Empty list to hold text documents.
    texts = {}

    listing = os.listdir(folder)
    for file in listing:

        if file.endswith(".html"):
            #print("File: ",file)
            print("Leyendo ", file)
            url = folder + "/" + file
            text = read_file(url)
            texts[file] = text

    print("Preparados ", len(texts), " documentos...")
    return(texts)


def extract_content(pag):
    attribs = {}
    '''
    title = ""
    try:
        #tokens = nltk.word_tokenize(pag.h1.get_text())
        #title = nltk.Text(tokens)
        title = pag.h1.get_text()
    except AttributeError:
        pass

    attribs['title'] = title
    print("Titulo extraido: ", attribs['title'])
    #Esta extraccion no recoge todos los titulos, por eso lo comentamos
    '''
    body = []
    try:
        texto = []
        parrafo = pag.find_all('p')
        for p in parrafo:
            sentence = p.get_text()
            texto.append(sentence)
        for sent in texto:
            try:
                #Toqueniza por frases frases
                token_sent = nltk.sent_tokenize(sent)
                body.extend(token_sent)
            except TypeError:
                pass

    except AttributeError:
        pass

    attribs['body'] = body

    #print("Texto extraido: ", attribs['body'])

    etiquetas = []

    try:
        tags = pag.find_all(class_=re.compile("tags|temas"))

        for t in tags:
            #Si tiene etiqueta ul cojo sus li, de ahi extraigo los tags

            li = t.find_all('li')
            if li is not None:
                for href in li:
                    if href.find(href=True) is not None:
                        etiquetas.append(href.find(href=True).get_text())

        etiquetas = list(set(etiquetas))
        try:
            etiquetas.remove("Show more")
        except ValueError:
            pass

    except AttributeError:
        pass

    attribs['tags'] = etiquetas
    #print("\nLas etiquetas son ", attribs['tags'])
    return attribs




#Hay que tener en cuenta que la API google no permite la traduccion de textos largos (habria que separarlo por lineas)
#Detecta el idioma y si no es ingles lo traduce
def translate_goslate(title, body):
    #Recibe el titulo y el body, asi primero comprueba si el titulo no esta en ingles y en tal caso ya traduce
    t = TextBlob(title)
    traduction_sentences = []
    language_id = t.detect_language()
    if language_id != 'en':
        print('Traducciendo al ingles.....')
        for text in body:
            #Se mete en el array tokenizado de body y traduce una a una las frases
            try:
                texto = TextBlob(text)
                translate = texto.translate(to='en')
                traduction_sentences.append(translate)
            except:
                pass

    else:
        traduction_sentences = body
    return traduction_sentences

def translate_title(title):
    #Recibe el titulo
    t = TextBlob(title)
    language_id = t.detect_language()
    if language_id != 'en':
        try:
            traduccion = t.translate(to='en')
        except:
            pass

    else:
        traduccion = title
    return str(traduccion)



def extract_tags(sentences):
    tag_sents = []
    #Obteniendo tags
    for sentence in sentences:
        try:
            sentence = str(sentence)#Para que no lo tome como objeto TextBlob
            tokenized_sentence = nltk.word_tokenize(sentence)
            tagged_sentence = nltk.pos_tag(tokenized_sentence)
            tag_sents.append(tagged_sentence)
        except TypeError:
            pass

    return tag_sents

def filter_stopwords(token_list):
    print('Filtrando stopwords.....')
    #Token list es lista de pares donde ya a detectado
    #las categorias gramaticales de cada palabra. (palabra - categoria gramatical)
    # que le ha dato nltk
    stop = set(stopwords.words('english'))
    clean_tokens = []
    for token in token_list:
        if token not in stop:
            clean_tokens.append(token)
    return clean_tokens

    # Se eliminan los símbolos de puntuación.
def remove_punctuation(token_list):
    result = []
    for token in token_list:
        punct_removed = ''.join([letter for letter in token if letter in string.ascii_letters])
        if punct_removed != '':
            result.append(punct_removed)
    return result

# Se genera la equivalencia para que el lematizador entienda los tags de la NLTK.
    # Primero usamos pos_tag para que gramatice, pero da una gramatizacion propia, nosotros
    # usamos el wordNetLematizer para lematizar, este lo entiende de otra forma, asi la funcion a
    # continuacion los convierte a entendible.
def wordnet_value(value):
    result = ''
    if value.startswith('J'):
        return wordnet.ADJ
    elif value.startswith('V'):
        return wordnet.VERB
    elif value.startswith('N'):
        return wordnet.NOUN
    elif value.startswith('R'):
        return wordnet.ADV
    return result

# Se realiza la lematización de los tokens conforme a su tag modificado.
def lemmatize(tag_sents):
    lemmatizer = WordNetLemmatizer()
    result = []
    for token in tag_sents:
        if len(token) > 0:
            pos = wordnet_value(token[1])
            if pos != '': #Porque wordnet_value si no reconoce devuelve vacío
                result.append(lemmatizer.lemmatize(str(token[0]).lower(), pos=pos))
                #para lematizar la funcion necesita la parabla en minuscula con la categoria gramatical
    return result


def extract_entities(tag_sents):
    print("Extrayendo EN...... ")
    # Las frases chunked devuelven la estructura de cada frase en forma de árboles.
    chunked_sentences = nltk.ne_chunk_sents(tag_sents, binary=False)

    # Función recursiva que recorre el arbol.

    types = {'ORGANIZATION', 'PERSON', 'LOCATION', 'GPE', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'FACILITY', 'GSP'}

    def extract_entity_names(t):
        entity_names = []
        # Se comprueba que el token tenga etiqueta.
        if hasattr(t, 'label') and t.label:
            # t.label = <bound method Tree.label of Tree('S', [Tree('GPE ........
            # Si es un entity name entonces lo agregamos con los que ya hemos identificado.
            if t.label() in types:
                if t.label() in {'ORGANIZATION', 'PERSON', 'LOCATION', 'GPE'}:
                    entity_names.append(t.label() + ': ' + ' '.join([child[0] for child in t]))
                else:
                    entity_names.append('MISCELANEA' + ': ' + ' '.join([child[0] for child in t]))
            # En caso contrario obtenemos todos los hijos del token para continuar con la búsqueda.
            else:
                for child in t:
                    entity_names.extend(extract_entity_names(child))
        return entity_names

    # Inicializamos el resultado.
    entity_names = []

    # Recorremos cada árbol correspondiente a cada frase.
    for tree in chunked_sentences:
        # print('Arbol: ', tree)
        entity_names.extend(extract_entity_names(tree))

    # Devolvemos
    return entity_names

def extract_EN(tag_sents):
    # Las frases chunked devuelven la estructura de cada frase en forma de árboles.
    chunked_sentences = nltk.ne_chunk_sents(tag_sents, binary=True)

    # Función recursiva que recorre el arbol.

    def extract_entity_names(t):
        entity_names = []
        # Se comprueba que el token tenga etiqueta.
        if hasattr(t, 'label') and t.label:
            # t.label = <bound method Tree.label of Tree('S', [Tree('GPE ........
            # Si es un entity name entonces lo agregamos con los que ya hemos identificado.
            if t.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in t]))

            # En caso contrario obtenemos todos los hijos del token para continuar con la búsqueda.
            else:
                for child in t:
                    entity_names.extend(extract_entity_names(child))
        return entity_names

    # Inicializamos el resultado.
    entity_names = []

    # Recorremos cada árbol correspondiente a cada frase.
    for tree in chunked_sentences:
        #print('Arbol: ', tree)
        entity_names.extend(extract_entity_names(tree))

    # Devolvemos
    return entity_names
