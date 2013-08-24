# -*- encoding: utf-8 -*-



import matplotlib.pylab as plt
import networkx as nx
# import pandas
import csv
import numpy as np
import zipfile, cStringIO
import networkx as nx
import matplotlib.pylab as plt
# import pylab as P
from matplotlib.transforms import offset_copy
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey, Boolean, Unicode, Sequence, Text, Table
from sqlalchemy import func, distinct
from sqlalchemy.sql import select
from sqlalchemy.orm import relationship, sessionmaker, backref
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql
from collections import defaultdict, Counter
from operator import itemgetter

engine = create_engine('sqlite:///datadb.sqlite', echo=True)

Session = sessionmaker(bind=engine)

session = Session()

Base = declarative_base()

class Influences(Base):
    """
    Tabela de influenciador
    """
    __tablename__ = 'influences'
    id = Column(Integer, primary_key=True)
    name = Column(Unicode)
    # influenced = relationship("Influenced", order_by="Influenced.id", backref="influences")

    def influences(id, name):
        self.id = id
        self.name = name

class Influenced(Base):
    """
    Tabela de influenciado
    """
    __tablename__ = 'influenced'
    id = Column(Integer, primary_key=True)
    name = Column(Unicode)
    # influences_id = Column(Integer, ForeignKey('influences.id'))
    # influences = relationship("Influences", backref=backref("influenced", order_by=id))

    def influenced(id, name):
        self.id = id
        self.name = name
        # self.infleunces_id = influences_id

metadata = MetaData()

Base.metadata.create_all(engine)


path = "datasets.csv" # arquivo github


ifile  = open(path, "rb")
reader = csv.reader(ifile)

k = [(k,v) for k,v in reader]

data_dict = {}
data_list = []

# itera um dicionario com influenciador (chave) e influenciado (valor) e
# adiciona no banco de dado mantendo o arquivo original
for i in range(len(k)):
    data_dict[k[i][0]] = k[i][1]
    infs = Influences(name=unicode(k[i][0], errors='ignore'))
    # conn.execute(infs)
    session.add(infs)
    infd = Influenced(name=unicode(k[i][1], errors='ignore'))
    # conn.execute(infd)
    session.add(infd)

def run_query(smtt):
    rs = smtt.execute()
    for row in rs:
        print row


# query para agrupar os influenciadores
C = session.query(Influences.name, func.count(Influences.id)).\
                                    group_by(Influences.name).all()

for x in range(len(C)):
    A = C

# itera um dicionario com influenciador e sua frequencia
infs_counter = {x:i for x,i in A}
order_dic_influences=sorted(infs_counter.items(),key=itemgetter(1),reverse=True)
# lista os 10 maiores influenciadores
#####################################
order_dic_influences[:10]           #
#####################################

# query para agrupar os influenciados
D = session.query(Influenced.name,func.count(Influenced.id)).group_by(Influenced.name).all()

for x in range(len(D)):
    H = D

# itera um dicionario com influenciados e sua frequencia
infd_counter = {x:i for x,i in H}
order_dic_influenced=sorted(infd_counter.items(),key=itemgetter(1),reverse=True)
# lista os 10 maiores influenciados
###################################
order_dic_influenced[:10]         #
###################################
unicos = [x for x in set(A+H)]
dict_unicos ={}
for i in range(len(unicos)):
    dict_unicos[(i+1)]=unicos[i]


# Lista todos com a quantidade de vezes em que aparecem
todos_frequencia = session.query(Influences.name, func.count(Influences.name),
                                Influenced.name, func.count(Influenced.name)).\
                                                            group_by(Influences.name)

result = todos_frequencia.filter(Influences.id == Influenced.id).all()

li_todos=[]
for x in range(len(H)):
    for i in range(len(A)):
        li_todos.append([A[i][0],A[i][1], H[x][0], H[x][1]])

tu_todos=[]
tu_todos=[(y,-k) for x,y,z,k in li_todos]


session.commit()

lista_semelhange=[]
def show_graph_features(G,circular=False):
    for node,adj_list in G.adjacency_iter():
        print node, ': ',adj_list   
        
    print  nx.adjacency_matrix(G)
    
    #N\u00f3s:
    print 'N\u00famero de n\u00f3s: ',G.number_of_nodes()
    print 'N\u00f3s: \\n','\\t',G.nodes()  
    #Arestas:
    print 'N\u00famero de arestas: ',G.number_of_edges()
    print 'Arestas: \\n','\\t',G.edges(data=True)

def plot_graph(G,circular=False):    
    #Desenha grafo e mostra na tela
    if not circular:
        nx.draw(G)
    else:
        nx.draw_circular(G)
    plt.show()

    
def separaGrauSemelhante(G):           
    #Dicion\u00e1rio dos semelhantes
    dic_grau_semelh={}
    for node,adj_list in G.adjacency_iter():
        if node<0 and len(adj_list)>=2:             
            dic_grau_semelh[node]=adj_list
    return dic_grau_semelh
    
    
def geraListaSemelhante(dic_grau_semelh):    
 
    controle_lista=[]
    dic_no={}
    li_semelhante=[]
    for node,dicionario in dic_grau_semelh.items():
        if node not in controle_lista:                              
            dic_no={}                                               
            qtd=0                                                   
            for chave,valor in dic_grau_semelh.items():             
                if sorted(dicionario.keys())==sorted(valor.keys()): 
                   qtd+=1                                           
                   controle_lista.append(chave)                     
                   dic_no[chave] =valor                             
            li_semelhante.append([dic_no.keys(),dicionario,qtd])    
    return sorted(li_semelhante,key=lambda semelhante: semelhante[2],reverse=True)

def printListaSemelhante(lista_semelhante):  
    print 'Terceira resposta trabalho, os 5 maiores grupos de \
            influenciados que tiveram um grupo de influenciadores semelhantes'
    for i in range(len(lista_semelhante)):
        if i >4: return
        print '===========  GRUPO   = ' + str(i+1)+ ' =============='
        print '       INFLUENCIADOS   = ' + str(lista_semelhante[i][2])
        for x in lista_semelhante[i][0]:
            print imprimeNome(x)
        print '       INFLUENCIADORES SEMELHANTES = '  +\
                 str(len(lista_semelhante[i][1].keys()))
        for x in lista_semelhante[i][1].keys():
            print imprimeNome(x)
           
    
def imprimeNome(i):
    if i<0:
        return dict_unicos[-(i)]  #Recupera o nome
    return dict_unicos[(i)]  


# Código do André Eduardo Bento Garcia
from rdflib import Graph, URIRef
from xml.dom.minidom import parseString as xml_parse
import requests
from SPARQLWrapper import SPARQLWrapper, SPARQLExceptions, JSON
import urllib2 
def dbpedia_query(resource, desired, what='property'):
    original_query = u'''select ?{} where {{
<{}>
<http://dbpedia.org/{}/{}> ?{} . }}'''
    query = original_query.format(desired, resource.replace(' ', '_'), what,
                                  desired, desired)
    #print query
    response = requests.get('http://dbpedia.org/sparql',
                            params={'query': query})
    #print response
    xml = xml_parse(response.content)
    results = xml.getElementsByTagName('binding')
    return [result.childNodes[0].childNodes[0].toxml() for result in results]

def get_country_of_birth(name):
    uri = 'http://dbpedia.org/resource/'
    birth_places = dbpedia_query(uri + name, 'birthPlace')
    #print birth_places
    country = []
    for place in birth_places:
        if place.startswith(uri):
            place = place.replace(uri, '')
        if ',' in place:
            place = place.split(',')[-1].strip()
        country.append(get_country(place))
    return country

def get_country(place):
    uri = 'http://dbpedia.org/page/'
    place = uri + place
    response = requests.get(place.replace(' ','_'))
    if 'http://schema.org/Country' in response.content:
        result = place.replace(uri, '')
    elif ':country' not in response.content:
        result = ''
    else:
        result = response.content.split(':country')[1].split('\n')[0]\
                         .split('">')[0].split('/')[-1]
    result = urllib2.unquote(result).replace('_', ' ')
    return result

def get_short_description(name):
    uri = 'http://dbpedia.org/resource/'
    result = dbpedia_query(uri + name, 'wordnet_type')[0]
    return result.replace('http://www.w3.org/2006/03/wn/wn20/instances/synset-', '')\
                 .replace('-noun-1', '')


# Resposta do exercício 7
def draw_graph(graph, labels=None, graph_layout='spectral',
               node_size=1600, node_color='red', node_alpha=0.3,
               node_text_size=10,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    # G=nx.Graph()

    # add edges
    # for edge in graph:
    #     G.add_edge(edge[0], edge[1], color='green')

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = zip(graph, labels)

    # show graph
    plt.show()
        
if __name__=='__main__':
    dic_grau_semelh={}
    lista_semelhante=[]
    G = nx.Graph(data=None, name = 'Grafico')
    G.add_nodes_from(dict_unicos.keys())
    G.add_edges_from(tu_todos)
    
    dic_grau_semelh=separaGrauSemelhante(G)
    
    lista_semelhante=geraListaSemelhante(dic_grau_semelh)
    
    printListaSemelhante(lista_semelhante)

    lista_dez_mai= order_dic_influences[:10]

    print 'Países dos 10 Maiores Influenciadores\n'
    print 'Nome\t'.ljust(30)+ '\t | ' +'\tPaís'
    for x in  lista_dez_mai:
        print x[0].ljust(30) + ' | ' +get_country_of_birth(x[0].replace(' ','_'))[0]
   
    print '\nAreas dos 10 Maiores Influenciadores\n'
    print 'Nome\t'.ljust(30)+ '\t | ' +'\tÁrea'
    for x in  lista_dez_mai:
        print x[0].ljust(30)+'  | ' +get_short_description(x[0].replace(' ','_'))

    draw_graph(G)
    
    #plot_graph(G)
    