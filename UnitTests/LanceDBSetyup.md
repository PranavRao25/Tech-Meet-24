```
import lancedb
```

# to connect to a database
```
db = lancedb.connect('lancedb/test')
table_name = 'demo'  # table creation
```

# open an existing table
```
tbl = db.open_table(table_name)
```

# data structure
# add list of key-value pairs / json file contents
# each dictionary must have a vector key present
```
data = [
    {"vector":v, "metadata":"vec1"}
    for v in ([1,2,3],[2,3,4],[4,5,6])
]
```

# to add data into existing table
```
tbl.add(data)
```

# to create and upsert data into it (the schema is inferred from the data
```
table = db.create_table(table_name, data=data)
```

# to create an empty table, you must specify the schema

# to query a table
```
query = "check check"
response = tbl.search(query).limit(5).to_pandas()  # convert to list/df/any format you want
```

# to delete a table
```
db.drop_table(table_name)
```

# count size of the db
```
size = tbl.count_rows()
```