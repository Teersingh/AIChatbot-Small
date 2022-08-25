import mysql.connector
def databaseconnect(Name,Age,Email,Phone,Query):
    mydb = mysql.connector.connect( 
                host="localhost", 
                user="root",  
               passwd="mum96766",
               database="collectdata"
               )
    mycursor = mydb.cursor() 
    sql='INSERT INTO visitors (name,age,email,phone,query) VALUES ("{0}","{1}", "{2}","{3}", "{4}");'.format(Name,Age,Email,Phone,Query) 
    mycursor.execute(sql) 
    mydb.commit()
    # print(mycursor.rowcount,"datainserted")

# if __name__=="__main__":
#         databaseconnect("teersinfg", 22, 'te33@gmail.com',"887660987","Nothing")
