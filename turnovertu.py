# -*- coding: utf-8 -*-
import tushare as ts
import sqlite3 as lite

import time
import datetime
import sys
#df=ts.get_hist_data('600418',start='2017-01-05',end='2017-09-09')
#print df

#df.to_csv('c:/good/600418.csv')



def  updatestocknameInDb(dbname, codedict):
    con = None
    con = lite.connect(dbname)
    rows=[]
    with con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS stocknamemap")
        cur.execute('create TABLE stocknamemap (stockcode text, stockname text)')
        for k, v in codedict.items():
            rows.append((k, v))

        cur.executemany("INSERT INTO stocknamemap VALUES(?, ?)", rows)


def   testsqlite(dbname ,manyrows, ct):
    con = None
    con = lite.connect(dbname)
    with con:
        cur = con.cursor()
        cur.execute('SELECT SQLITE_VERSION()')
        data = cur.fetchone()
        print("SQLite version: %s" % data)
        if ct==1:
            cur.execute("DROP TABLE IF EXISTS stockquote")
            cur.execute('create table stockquote (stockcode text,tradedate Date ,startprice float ,highprice float, endprice float, lowprice float, volume bigint, totalmoney bigint,turnover float,primary key(stockcode,tradedate))')
        #for i in range(len(manyrows)):
        #    cur.execute("INSERT INTO stockquote VALUES(manyrows[i].stockcode,manyrows[i].tradedate,manyrows[i].startprice,manyrows[i].highprice,manyrows[i].endprice,manyrows[i].lowprice,manyrows[i].volume,manyrows[i].money)")
        else:
            try:
                cur.executemany("INSERT INTO stockquote VALUES(?, ?, ?,?,?,?,?,?,?)", manyrows)
            except Exception as e:
                print(e)
                print("insert many error")


def  convertdfTodbrows(df,code):
    #print code
    #print df
    result=[]
    for index, row in df.iterrows():
            #print index
            #print row
            #print row['open']
            #print row['high']
            #print "\N"

        temp=[]
        temp.append(code)
        temp.append(index)
        temp.append((float)(row['open']))
        temp.append((float)(row['high']))
        temp.append((float)(row['close']))
        temp.append((float)(row['low']))
        temp.append((int)(row['volume']*100))
        temp.append((int)(0))
        temp.append((float)(row['turnover']))
        result.append(temp)

    return result


def  convertdfTodbrowsNOturnover(df,code):
    #print code
    #print df
    result=[]
    for index, row in df.iterrows():
        #print index
        #print row
        #print row['open']
        #print row['high']
        #print "\N"

        temp=[]
        temp.append(code)
        temp.append(index)
        temp.append((float)(row['open']))
        temp.append((float)(row['high']))
        temp.append((float)(row['close']))
        temp.append((float)(row['low']))
        temp.append((int)(row['volume']*100))
        temp.append((int)(0))
        #temp.append((float)(row['turnover']))
        temp.append((float)(0))

        result.append(temp)

    return result




def  convertTodaydfTorows(df,neardate):
    #print code
    #print df
    result=[]
    for index, row in df.iterrows():
        #print index
        #print row
        #print row['open']
        #print row['high']
        #print "\N"

        if row['high']==0.0:
            print("%s stop" %(row['code']))
            continue

        temp=[]
        temp.append(row['code'])
        temp.append(neardate)
        temp.append((float)(row['open']))
        temp.append((float)(row['high']))
        temp.append((float)(row['trade']))
        temp.append((float)(row['low']))
        temp.append((int)(row['volume']))
        temp.append((int)(row['amount']))
        temp.append((float)(row['turnoverratio']))

        result.append(temp)

    return result


def  getallstockcodes():

    df=ts.get_stock_basics()
    #lista=df['code']
    #lista=df.to_dict(orient='record')
    lista=df.index
    #print lista[0]
    return lista

import chardet

def get_charset(s):
    return chardet.detect(s)['encoding']




def updatestockname(dbname):

    reload(sys)
    sys.setdefaultencoding('utf8')
    result={}
    df=ts.get_stock_basics()
    reload(sys)
    sys.setdefaultencoding('utf8')
    for index, row in df.iterrows():
            #print row['name'].encode('gbk')
            #result[index]=row['name']
            #print get_charset(row['name'])
        result[index]=str(row['name'],'utf8')
        #result[index]=(((row['name']).encode('gbk')).decode('gbk')).encode('utf8')

    updatestocknameInDb(dbname, result)

    print(result)


def updateallstocks(daynums, dbname):
    for i in range(0, 10):
        try:
            codelist=getallstockcodes()
            break
        except Exception as e:
            print("getallstockcodes error")
            print(e)
    now_time = datetime.datetime.now()
    yes_time = now_time + datetime.timedelta(days=-daynums)
    endstr=now_time.strftime('%Y-%m-%d')
    startstr=yes_time.strftime('%Y-%m-%d')
    stocknum=0

    for item in codelist:
        try:
            print("stocknum=%d" %(stocknum))
            df=ts.get_hist_data(item ,start=startstr,end=endstr)
            #print df
            #result=convertdfTodbrows(df,item)
            result=convertdfTodbrowsNOturnover(df,item)
            #print df
            testsqlite(dbname,result,0)
            stocknum+=1
            #raw_input()

        except Exception as e:
            print("exception")
            print(e)

def  get_shangHaiIndexAndReturnNearestTradeDate( ):

    df=ts.get_hist_data("sh")
    return  df.index[0]

def    get_today_and_putto_database(dbname ):
    tl=time.localtime()
    if(tl.tm_hour<15)  and   (tl.tm_hour>8):
        print("\ Now trading time , Do not update by This method")
        return 0

    df=ts.get_today_all()
    neardate=get_shangHaiIndexAndReturnNearestTradeDate( )
    result=convertTodaydfTorows(df,neardate)
    testsqlite(dbname,result,0)
    #print result
    #print df

def builddbwithturnover(dbname):
    manyrows=[]
    testsqlite(dbname ,manyrows, 1)
    updateallstocks(1200, dbname)

if  __name__=="__main__":

        #updateallstocks(60, 'C://good//mypanel//turnover1200db.db')
    get_today_and_putto_database('C://good//mypanel//turnover1200db.db' )
    #get_shangHaiIndexAndReturnNearestTradeDate( )
    #builddbwithturnover('C://good//mypanel//turnover1200db.db')
    updatestockname('C://good//mypanel//turnover1200db.db')
