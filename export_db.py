import csv
import os
import subprocess
import sys

CONTAINER = "mysql-curriculum-skill"
DB = "skillcrawl"
USER = "root"
PASS = "root"
OUTDIR = "csv_exports"

csv.field_size_limit(sys.maxsize)
os.makedirs(OUTDIR, exist_ok=True)

def run_mysql(query):
    cmd = [
        "docker", "exec", CONTAINER,
        "mysql",
        f"-u{USER}",
        f"-p{PASS}",
        "--batch",
        "--quick",
        "--default-character-set=utf8mb4",
        DB,
        "-e",
        query,
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

tables_output = run_mysql("SHOW TABLES;")
tables = tables_output.strip().splitlines()[1:]

for table in tables:
    print(f"Exporting {table}...")

    tsv = run_mysql(f"SELECT * FROM `{table}`;")

    out_path = os.path.join(OUTDIR, f"{table}.csv")

    reader = csv.reader(tsv.splitlines(), delimiter="\t")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
        writer.writerows(reader)

    print(f"Saved {out_path}")

print("Done.")
