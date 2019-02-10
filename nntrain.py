# -*- coding: utf-8 -*-
"""
 prepare for  input and output data for training 
 Copyright @author: zhao shi rong   shxzhaosr@163.com

All Rights Reserved.


 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
"""

import  pricedb
import  bisect
import  numpy as np
import  random
from datetime import datetime
import tensorflow as tf
import sys

def  build_train_period_samples(loaddict,DataStartDate,DataEndDate,trainStartDate,trainEndDate,trainDays,pointpricelist):
    traindict={}
    traindict['traindata']=[]
    traindict['profit']=[]
    pointpricelist=[]
    for curcode in loaddict:
        loaddict[curcode]['splitidx']=[]
        startindex=-1
        endindex=-1
        trainstartindex=-1
        trainEndindex=-1
        if DataStartDate in  loaddict[curcode]['tradedateindex']:
            startindex=loaddict[curcode]['tradedateindex'][DataStartDate]
        if DataEndDate in  loaddict[curcode]['tradedateindex']:
            endindex=loaddict[curcode]['tradedateindex'][DataEndDate]
        
        if trainStartDate in  loaddict[curcode]['tradedateindex']:
            trainstartindex=loaddict[curcode]['tradedateindex'][trainStartDate]
            
        if trainEndDate in  loaddict[curcode]['tradedateindex']:
            trainEndindex=loaddict[curcode]['tradedateindex'][trainEndDate]    
        if(startindex==-1):
            for idx, val in enumerate(loaddict[curcode]['pricesrows']):
                if(val['tradedate']>=DataStartDate):
                    startindex=idx
                    break
        if(endindex==-1):
            for idx, val in enumerate(loaddict[curcode]['pricesrows']):
                if(val['tradedate']>=DataEndDate):
                    endindex=idx 
                    break
                
        if(trainstartindex==-1):
            for idx, val in enumerate(loaddict[curcode]['pricesrows']):
                if(val['tradedate']>=trainStartDate):
                    trainstartindex=idx 
                    break  
                
        if(trainEndindex==-1):
            for idx, val in enumerate(loaddict[curcode]['pricesrows']):
                if(val['tradedate']>=trainEndDate):
                    trainEndindex=idx 
                    break                
        if(startindex==-1) or  (endindex==-1):
            continue
        if(endindex<startindex):
            continue
        totalvolume=None
        for idx in range(startindex,endindex):
            if idx>0:
                averprice=loaddict[curcode]['pricesrows'][idx]['totalmoney']/(float)(loaddict[curcode]['pricesrows'][idx]['volume'])
                loaddict[curcode]['pricesrows'][idx]['startpercent']=loaddict[curcode]['pricesrows'][idx]['startprice']/loaddict[curcode]['pricesrows'][idx-1]['endprice']-1.0
                loaddict[curcode]['pricesrows'][idx]['endpercent']=loaddict[curcode]['pricesrows'][idx]['endprice']/loaddict[curcode]['pricesrows'][idx-1]['endprice']-1.0
                loaddict[curcode]['pricesrows'][idx]['highpercent']=loaddict[curcode]['pricesrows'][idx]['highprice']/loaddict[curcode]['pricesrows'][idx-1]['endprice']-1.0
                loaddict[curcode]['pricesrows'][idx]['lowpercent']=loaddict[curcode]['pricesrows'][idx]['lowprice']/loaddict[curcode]['pricesrows'][idx-1]['endprice']-1.0
                loaddict[curcode]['pricesrows'][idx]['averpercent']=averprice/loaddict[curcode]['pricesrows'][idx-1]['endprice']-1.0
                if(loaddict[curcode]['pricesrows'][idx]['lowpercent']<-0.12):
                    loaddict[curcode]['splitidx'].append(idx)
               
                    
                if(totalvolume==None) and (idx>=trainDays-1):
                    totalvolume=0 
                    for j in range(idx, idx-trainDays,-1):
                        totalvolume+=loaddict[curcode]['pricesrows'][j]['volume']
                elif(idx>=trainDays):
                     totalvolume-=loaddict[curcode]['pricesrows'][idx-trainDays]['volume']
                     totalvolume+=loaddict[curcode]['pricesrows'][idx]['volume']
                 
                if(idx>=trainDays-1):
                    loaddict[curcode]['pricesrows'][idx]['totalvolume']=totalvolume
                    
                
                    
        #for idx in range(trainstartindex,endindex):
        for idx in range(trainstartindex,trainEndindex):
            if not (idx-trainDays>startindex+2):
                continue
            #无足够天数历史数据t
            if(idx+1>len(loaddict[curcode]['pricesrows'])-1):
                continue
            if(loaddict[curcode]['pricesrows'][idx+1]['lowprice']/loaddict[curcode]['pricesrows'][idx]['endprice']<0.88):
                continue
            #因拆分股票计算收益不准确
            if(loaddict[curcode]['pricesrows'][idx]['startprice']/loaddict[curcode]['pricesrows'][idx-1]['endprice']>1.07):
                continue
            #今天开盘价太高，有可能买不到
            if(len(loaddict[curcode]['splitidx'])>0):
                loaddict[curcode]['splitidx'].sort()
                #print (loaddict[curcode]['splitidx'])
                u = bisect.bisect_left(loaddict[curcode]['splitidx'], idx)
                #print (u)
                #print (idx)
                if(idx-loaddict[curcode]['splitidx'][u-1]<trainDays+1):
                    continue
            #历史数据中有拆股，影响训练准确
            
            traindatalist=[]
            traindatalist.append(loaddict[curcode]['pricesrows'][idx]['startprice'])
            #今天开盘价
            traindatalist.append(loaddict[curcode]['pricesrows'][idx]['startpercent'])
            #今天开盘百分比
            for k in range(idx-1, idx-1-trainDays,-1):
                traindatalist.append(loaddict[curcode]['pricesrows'][k]['startpercent'])
                traindatalist.append(loaddict[curcode]['pricesrows'][k]['endpercent'])
                traindatalist.append(loaddict[curcode]['pricesrows'][k]['highpercent'])
                traindatalist.append(loaddict[curcode]['pricesrows'][k]['lowpercent'])
                #traindatalist.append(loaddict[curcode]['pricesrows'][k]['averpercent'])
                volumeratio=loaddict[curcode]['pricesrows'][k]['volume']*trainDays/(float)(loaddict[curcode]['pricesrows'][idx-1]['totalvolume'])
                traindatalist.append(volumeratio)
            
            #profit=loaddict[curcode]['pricesrows'][idx+1]['startprice']/loaddict[curcode]['pricesrows'][idx-1]['endprice']-1.0      
            #profit=profit*4
            profit=loaddict[curcode]['pricesrows'][idx+1]['startprice']/loaddict[curcode]['pricesrows'][idx]['startprice']-1.0 
            #下标必须相同
            traindict['traindata'].append(traindatalist)
            traindict['profit'].append(profit)
            pointpricelist.append(loaddict[curcode]['pricesrows'][idx])    
            loaddict[curcode]['pricesrows'][idx]['trainindex']=len(traindict['traindata'])-1    
    return traindict             
        

# change from  def evaluateCodeAndDate_justOneDay( pricerows, monlist,  k):
def buildOneInputAndprofit(pricerows,   k , trainDays):
    if not (k-trainDays>2):
                return None
    #无足够历史数据
    if(k+1>len(pricerows)-1):
                return None
    if(pricerows[k+1]['lowprice']/pricerows[k]['endprice']<0.88):
                return None
    #因拆分股票计算收益不准确
    if(pricerows[k]['startprice']/pricerows[k-1]['endprice']>1.07):
                return None
    #今天开盘价太高，有可能买不到
    
    for j in range(k,k-trainDays-3,-1):
         if(pricerows[j]['lowprice']/pricerows[j-1]['endprice']<0.88):
                return None
    #历史数据中有拆股，影响训练准确   

    traindatalist=[]
    traindatalist.append(pricerows[k]['startprice'])
            #今天开盘价
    traindatalist.append(pricerows[k]['startpercent'])
            #今天开盘百分比
    for j in range(k-1, k-1-trainDays,-1):
                traindatalist.append(pricerows[j]['startpercent'])
                traindatalist.append(pricerows[j]['endpercent'])
                traindatalist.append(pricerows[j]['highpercent'])
                traindatalist.append(pricerows[j]['lowpercent'])
                #traindatalist.append(loaddict[curcode]['pricesrows'][k]['averpercent'])
                volumeratio=pricerows[j]['volume']*trainDays/(float)(pricerows[k-1]['totalvolume'])
                traindatalist.append(volumeratio)     

    profit=pricerows[k+1]['startprice']/pricerows[k]['startprice']-1.0

    resultdict={}
    resultdict['traindatalist']=traindatalist
    resultdict['profit']=profit
    resultdict['stockcode']=pricerows[k]['stockcode']
    resultdict['tradedate']=pricerows[k]['tradedate']
    return  resultdict




def  computeOneDaybyNeuralNetwork( loaddict,   curtradedate):

    codeResultDict={}

    for code in list(loaddict.keys()):

        if curtradedate not in loaddict[code]['tradedateindex']:
            continue
        codedayindex= loaddict[code]['tradedateindex'][curtradedate]
        #curresult=evaluateCodeAndDate( loaddict[code]['pricesrows'], monlist, codedayindex )

        #curresult=evaluateCodeAndDate_B( loaddict[code]['pricesrows'], monlist, codedayindex )
        #curresult=evaluateCodeAndDate_justOneDay( loaddict[code]['pricesrows'], monlist, codedayindex )
        trainDays=30
        try:
            curresult=buildOneInputAndprofit(loaddict[code]['pricesrows'], codedayindex , trainDays)
        except:
            curresult=None
            continue
            
        #print ("processing")
        #print (code)
        if(curresult!=None):
            codeResultDict[code]= curresult
            curresult['code']=code
            #print ("addcode")
            #print (code)

    resultlist=[]
    stockcodelist=[]
    #print(codeResultDict)
    for key, value in codeResultDict.items():
        resultlist.append(value['traindatalist'])
        stockcodelist.append(key)
        
        
    inputlistnp=np.array(resultlist)
    
    return codeResultDict, stockcodelist,inputlistnp
    
    ##以下将添加程序  将  inputlistnp 传入神经网络， 输出预测结果

    """
    newresultlist=sorted(resultlist, key=lambda student: student['sum'], reverse=True)
    totalprofit=0.0
    for i in range(0, 5):
        totalprofit+=newresultlist[i]['profit']

    averprofit=totalprofit/5.0

    return averprofit
    """


def  putPredictProfitToCodeResultDict(codeResultDict,stockcodelist,predictlist):
      if(len(stockcodelist)!=len(predictlist)):
          print ("Heavy error length of stockcodelist and predictlist not equal")
      for idx, value in   enumerate(predictlist):
          codeResultDict[stockcodelist[idx]]['predict']=value
          



#    sqlite> SELECT * FROM stockquote where stockcode='600418' and tradedate>'2018-03-20';
def  testData():
    loaddict= pricedb.loadstockdatatodict('C://good//mypanel//turnover1200db.db',  '2018-06-01' ,  '2050-11-30')
    print ("load complted")
    pointpricelist=[]
    traindict=build_train_period_samples(loaddict,'2017-09-01','2019-01-01','2018-11-01',30,pointpricelist)
    """
    for i in range(-1,-200, -1):
        if('trainindex' not in loaddict['600418']['pricesrows'][i]): 
            continue
        print (loaddict['600418']['pricesrows'][i]['tradedate'])
        print (traindict['traindata'][loaddict['600418']['pricesrows'][i]['trainindex']])
        print (traindict['profit'][loaddict['600418']['pricesrows'][i]['trainindex']])
    
        #print (loaddict['600418']['pricesrows'])
    """    
    trainnum= np.array(traindict['traindata'])
    profitnum=np.array(traindict['profit'])
    #fullength=len()
    print (trainnum.shape)
    print (profitnum.shape)
    randomlist=generate_random_sample_list(trainnum,profitnum)
    maxbig=(int)(len(traindict['profit'])/10)
    while(1):
        bnum=random.randint(0,maxbig-1)
        x,y=generate_batch_sample(trainnum,profitnum,randomlist, 10, bnum)
        print (x)
        print (y)
        ch = sys.stdin.read(1)

 


def  testData_B():
    loaddict= pricedb.loadstockdatatodict('C://good//mypanel//turnover1200db.db',  '2017-06-01' ,  '2050-11-30')
    print ("load complted")
    pointpricelist=[]
    traindict=build_train_period_samples(loaddict,'2017-07-01','2019-01-04','2018-01-02','2018-12-29',30,pointpricelist)
    """
    for i in range(-1,-200, -1):
        if('trainindex' not in loaddict['600418']['pricesrows'][i]): 
            continue
        print (loaddict['600418']['pricesrows'][i]['tradedate'])
        print (traindict['traindata'][loaddict['600418']['pricesrows'][i]['trainindex']])
        print (traindict['profit'][loaddict['600418']['pricesrows'][i]['trainindex']])
    
        #print (loaddict['600418']['pricesrows'])
    """    
    trainnum= np.array(traindict['traindata'])
    profitnum=np.array(traindict['profit'])
    #fullength=len()
    print (trainnum.shape)
    print (profitnum.shape)
    randomlist=generate_random_sample_list(trainnum,profitnum)
    samlength=(int)(len(traindict['profit']))
    return trainnum,profitnum,randomlist,samlength,loaddict


       

def  generate_random_sample_list(trainnum,profitnum):
    if(trainnum.shape[0]!=profitnum.shape[0]):
        print ("heavy problem not equeal")
       
    
    random.seed(datetime.now())    
    randomlist=list(range((int)(trainnum.shape[0])))
    for i in range(0, trainnum.shape[0]):
        big=(int)((trainnum.shape)[0])
        #print (big)
        idxb=random.randint(0,big-1)
        #print (idxb)
        #print (i)
        temp=randomlist[i]
        randomlist[i]=randomlist[idxb]
        randomlist[idxb]=temp 

    return randomlist 


namecounter=0

def  generate_batch_sample(trainnum,profitnum,randomlist, bsize, bnum):
     sidx=bsize*bnum%len(randomlist)
     btrainlist=[]
     bprofitlist=[]
     for j in range(sidx,sidx+bsize,1 ):
         btrainlist.append(trainnum[randomlist[j]])
         templist=[]
         templist.append(profitnum[randomlist[j]])
         bprofitlist.append(templist)
         #bprofitlist.append(profitnum[randomlist[j]])
       
     nbtrainlist=np.array(btrainlist)
     nbprofitlist=np.array(bprofitlist)  
     #with tf.variable_scope("one", reuse=True):
     """
     global namecounter
     namecounter+=1
     train_batch_tf_var = tf.get_variable("trainbatch"+str(namecounter),
                                 initializer=nbtrainlist)
     #with tf.variable_scope("one", reuse=True):
     profit_batch_tf_var = tf.get_variable("profitbatch"+str(namecounter),
                                 initializer=nbprofitlist)
     
     return train_batch_tf_var,profit_batch_tf_var
     """
     return  nbtrainlist, nbprofitlist
        
if __name__ == "__main__":
    testData()        