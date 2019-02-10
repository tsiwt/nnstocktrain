# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:31:08 2019
interface with  database 

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
import sqlite3 as lite


def  loadstockdatatodict(dbname,  DataStartDate ,ProfitEndDate  ):

    loaddict={}
    con = lite.connect(dbname)
    with con:
        cur = con.cursor()
        cur.execute('SELECT SQLITE_VERSION()')
        stocknamedict={}

        #print dbname
        #raw_input()
        stockcodes=cur.execute('select * from  stocknamemap')
        for onecode  in stockcodes:
            stocknamedict[onecode[0]]=onecode[1]
        for curcode in list(stocknamedict.keys()):
            if curcode[0]=='9'  or   curcode[0]=='3' or  curcode[0]=='2':
                continue
            if stocknamedict[curcode].find('ST',0)>=0:
                continue

            manyrows=[]
            for row in cur.execute('SELECT * FROM stockquote where stockcode=:id and tradedate>:did and tradedate<:eid   ORDER BY tradedate',{"id":curcode, "did":DataStartDate, "eid":ProfitEndDate} ):
                oneItemDict={ 'stockcode':row[0],'tradedate':row[1] ,'startprice':row[2], 'highprice':row[3], 'endprice':row[4], 'lowprice':row[5],'volume':row[6], 'totalmoney':row[7],'turnover':row[8]}
                manyrows.append(oneItemDict)

            loaddict[curcode]={}
            loaddict[curcode]['pricesrows']=manyrows
            rowlength=len(manyrows)
            datemapdict={}
            for j in range(0, rowlength):
                datemapdict[loaddict[curcode]['pricesrows'][j]['tradedate']]=j

            loaddict[curcode]['tradedateindex']=datemapdict

        #pickle.dump(loaddict, open( "./outlibtest.pickle", "wb" ))

        return loaddict
    
    
    
    
def  buildtradedatelist(loaddict):
    tradedatedict={}
    tradedatelist=[]
    i=0
    for code in list(loaddict.keys( )):
        i+=1
        if(i>20):
            break
        for item in loaddict[code]['pricesrows']:
            tradedatedict[item['tradedate']]=1

    for item in  list(tradedatedict.keys( )):
        tradedatelist.append(item)

    tradedatelist.sort()
    return tradedatelist