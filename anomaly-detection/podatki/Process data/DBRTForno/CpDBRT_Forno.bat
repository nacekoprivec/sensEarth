rem del C:\Users\UTENTE\Documents\DBRTForno\ArchivioValoriProcesso0.rdb
rem del C:\Users\UTENTE\Documents\DBRTForno\ArchivioValoriProcesso0.csv
xcopy V:\ArchivioValoriProcesso0.rdb C:\Users\UTENTE\Documents\DBRTForno\ArchivioValoriProcesso0.rdb /y
C:\Users\UTENTE\Documents\DBRTForno\sqlite3 -csv -separator ; C:\Users\UTENTE\Documents\DBRTForno\ArchivioValoriProcesso0.rdb "SELECT VarName, VarValue, Validity, Time_ms, datetime((Time_ms/1000000)+julianday('1899-12-31','-1 day')) as Time_cnv from logdata" > "C:\Users\UTENTE\Documents\DBRTForno\ArchivioValoriProcesso0.csv"


