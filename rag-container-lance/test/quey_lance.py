import lancedb

db = lancedb.connect("/usr/src/.lancedb")
table = db.open_table("mul_cond")
# table.create_fts_index("filename")

results = (
    #table.search("a",fts_columns="filename")
    table.search([5.4, 9.5])
    .where("org_list like '%o1%' AND person_list like '%p1%' AND secret_level='s1'")
    .limit(20).to_list()
)
print(f"row count:{table.count_rows()}")
print(results)
#print([r['filename'] for r in results])