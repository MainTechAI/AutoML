import sqlite3
from sqlite3 import Error
import pathlib

# DBMS wasn't implemented due to the time constrains,
# although it makes sense

def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


db_path = pathlib.Path().resolve().joinpath("db.sqlite")
con = create_connection(db_path)  # "E:\\sm_app.sqlite"

cur = con.cursor()

# create DATASET table
query = """
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    meta_feature1 INTEGER NOT NULL,
    meta_feature2 INTEGER NOT NULL,
    meta_feature3 INTEGER NOT NULL
);"""
cur.execute(query)
con.commit()


# Insert a row of data
cur.execute("INSERT INTO dataset VALUES (1,'BUY',100,'some text','abcd')")
cur.execute(query)
con.commit()



# END - close connection
con.close()

