import numpy as np
import random
import copy
import math

# 16x11 Matris şeklinde kapasiteleri dönderiyor
def kapasite_hesapla_cumulative(inv_genes):
    genes = copy.deepcopy(inv_genes)
    exist = copy.deepcopy(data['exists'])
    kapasiteler = []

    for i in range(11):
        exist[i] = exist[i]*data['hours'][i]

    for i in range(16):
        kapasiteler.append(kapasite_hesapla_yil_mwh(genes[0:i+1],exist))

    return kapasiteler

# 1x11 Vektör şeklinde yıllık kapasiteleri dönderiyor
def kapasite_hesapla_yil_mwh(inv_genes_v,exist):
    kapasiteler = np.empty(11)
    temp = []
    units = inv_genes_v.sum(axis=0)
    for i in range(11):
        kapasiteler[i] = units[i]*data['initial_cap'][i]*data['hours'][i]

    temp.append(kapasiteler)
    temp.append(exist)
    temp = [sum(x) for x in zip(*temp)]
    return temp

# Kaynaklar için rastgele yatırım değeri üretir integer dönderiyor
def invest_uret(climit):
    return random.randint(0,math.ceil(climit/10))

# Kaynaklar için rastgele üretim değeri üretir float dönderiyor
def generation_uret(talep,kapasite):
    if kapasite<= 0 or talep <= 0:
        return 0
    if kapasite >= talep:
        return random.uniform(talep/2,talep)
    else:
        return random.uniform(0, kapasite)

# Bireyin genlerini olusturan fonksiyon 1x352 float vektör şeklinde individual datatype dönderiyor
def birey_uret():

    investment_genes = np.zeros((16,11),dtype=int)
    generation_genes = np.zeros((16,11))
    climit = []

    for i in range(16):
        climit = copy.deepcopy(data['climit'])
        if i < 11:
            climit[5] = 0
        for j  in range(len(data['initial_cap'])):
            uretim = invest_uret(climit[j])
            investment_genes[i][j] += uretim
            climit[j]-= uretim

    kapasiteler = kapasite_hesapla_cumulative(investment_genes)