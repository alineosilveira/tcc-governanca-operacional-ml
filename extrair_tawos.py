"""
extrair_tawos.py
----------------
Lê o dump MySQL da TAWOS direto do TAWOS.sql.zip (ou do TAWOS.sql já descompactado),
SEM precisar instalar MySQL, e gera CSVs enxutos na pasta tawos_csv/.

Uso (na pasta onde está o zip):
    python extrair_tawos.py
    python extrair_tawos.py C:\\caminho\\TAWOS.sql.zip

Só usa a biblioteca padrão do Python 3.8+.
"""
import csv
import io
import os
import re
import sys
import time
import zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "tawos_csv")

# Tabelas e colunas que serão exportadas (o resto é ignorado)
KEEP = {
    "Repository": ["ID", "Name"],
    "Project": ["ID", "Project_Key", "Name", "Start_Date", "Last_Update_Date", "Repository_ID"],
    "Version": ["ID", "Name", "Archived", "Released", "Release_Date", "Project_ID"],
    "Component": ["ID", "Name", "Project_ID"],
    "Sprint": ["ID", "Name", "State", "Start_Date", "End_Date", "Complete_Date", "Project_ID"],
    "Issue": ["ID", "Issue_Key", "Title", "Type", "Priority", "Status", "Resolution",
              "Creation_Date", "Estimation_Date", "Resolution_Date", "Last_Updated",
              "Story_Point", "Timespent", "In_Progress_Minutes", "Total_Effort_Minutes",
              "Resolution_Time_Minutes", "Pull_Request_URL", "Creator_ID", "Reporter_ID",
              "Assignee_ID", "Project_ID", "Sprint_ID"],
    "Fix_Version": ["Issue_ID", "Fix_Version_ID"],
    "Affected_Version": ["Issue_ID", "Affected_Version_ID"],
    "Issue_Component": ["Issue_ID", "Component_ID"],
    "Issue_Link": ["Issue_ID", "Name", "Description", "Direction", "Target_Issue_ID"],
    "Change_Log": ["Issue_ID", "Field", "From_String", "To_String", "Creation_Date"],
}
# Colunas derivadas do texto da descrição (o texto em si não é exportado, para o CSV ficar leve)
ISSUE_EXTRA = ["Desc_Len", "Desc_Code_Len"]
# Do Change_Log só interessam estes campos
CHANGELOG_FIELDS = {"status", "resolution", "priority", "fix version", "version"}

TOKEN = re.compile(r"'((?:[^'\\]|\\.|'')*)'|(NULL)|([^,()'\s]+)|([(),])", re.S)
ESC = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}
ESC_RE = re.compile(r"\\(.)", re.S)


def unescape(s):
    s = s.replace("''", "'")
    return ESC_RE.sub(lambda m: ESC.get(m.group(1), m.group(1)), s)


def abrir_dump(caminho):
    if caminho.lower().endswith(".zip"):
        z = zipfile.ZipFile(caminho)
        nomes = [n for n in z.namelist() if n.lower().endswith(".sql")]
        if not nomes:
            sys.exit("Não achei nenhum .sql dentro do zip.")
        nome = max(nomes, key=lambda n: z.getinfo(n).file_size)
        print(f"Lendo {nome} de dentro do zip ({z.getinfo(nome).file_size/1e9:.2f} GB descompactado)")
        return io.TextIOWrapper(z.open(nome), encoding="utf-8", errors="replace", newline="")
    return open(caminho, encoding="utf-8", errors="replace", newline="")


def statements(f):
    """Gera instruções SQL completas (CREATE TABLE / INSERT INTO), juntando linhas quando preciso."""
    buf = []
    in_str = False
    for line in f:
        if not buf:
            head = line.lstrip()[:12].upper()
            if not (head.startswith("INSERT") or head.startswith("CREATE TABLE")):
                continue
        buf.append(line)
        # atalho: instrução inteira numa linha só (padrão do mysqldump)
        if len(buf) == 1 and not in_str and line.rstrip().endswith(";"):
            yield line
            buf = []
            continue
        # controla se terminamos dentro de uma string (quebra de linha literal dentro de texto)
        i, n = 0, len(line)
        while True:
            j = line.find("'", i)
            if j < 0:
                break
            # conta barras invertidas antes da aspa
            k, bs = j - 1, 0
            while k >= 0 and line[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 0:
                in_str = not in_str
            i = j + 1
        if not in_str and line.rstrip().endswith(";"):
            yield "".join(buf)
            buf = []


CREATE_RE = re.compile(r"CREATE TABLE\s+`?(\w+)`?\s*\((.*)\)", re.S | re.I)
COL_RE = re.compile(r"^\s*`(\w+)`", re.M)
INSERT_RE = re.compile(r"INSERT\s+(?:IGNORE\s+)?INTO\s+`?(\w+)`?\s*(\([^)]*\))?\s*VALUES\s*", re.I)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else None
    if not caminho:
        for cand in ("TAWOS.sql.zip", "TAWOS.sql"):
            p = os.path.join(AQUI, cand)
            if os.path.exists(p):
                caminho = p
                break
    if not caminho or not os.path.exists(caminho):
        sys.exit("Coloque o TAWOS.sql.zip na mesma pasta deste script (ou passe o caminho como argumento).")

    os.makedirs(SAIDA, exist_ok=True)
    colunas = {}      # tabela -> lista de colunas do dump
    writers, files, contagem = {}, {}, {}
    t0 = time.time()
    ultimo = [t0]

    def writer_for(tab):
        if tab not in writers:
            fh = open(os.path.join(SAIDA, f"{tab}.csv"), "w", encoding="utf-8", newline="")
            w = csv.writer(fh)
            header = KEEP[tab] + (ISSUE_EXTRA if tab == "Issue" else [])
            w.writerow(header)
            writers[tab], files[tab], contagem[tab] = w, fh, 0
        return writers[tab]

    with abrir_dump(caminho) as f:
        for st in statements(f):
            if st.lstrip()[:6].upper() == "CREATE":
                m = CREATE_RE.search(st)
                if m:
                    colunas[m.group(1)] = COL_RE.findall(m.group(2))
                continue
            m = INSERT_RE.match(st.lstrip())
            if not m:
                continue
            tab = m.group(1)
            if tab not in KEEP:
                continue
            cols = COL_RE.findall(m.group(2).replace(",", "\n")) if m.group(2) else colunas.get(tab)
            if not cols:
                print(f"Aviso: sem definição de colunas para {tab}, pulando")
                continue
            idx = {c: i for i, c in enumerate(cols)}
            faltando = [c for c in KEEP[tab] if c not in idx]
            if faltando:
                print(f"Aviso: {tab} sem colunas {faltando} (serão vazias)")
            w = writer_for(tab)
            body = st.lstrip()[m.end():]
            row, depth = [], 0
            for tk in TOKEN.finditer(body):
                s, null, lit, punct = tk.groups()
                if punct == "(":
                    depth += 1
                    row = []
                elif punct == ")":
                    depth -= 1
                    rec = {}
                    for c in KEEP[tab]:
                        i = idx.get(c)
                        v = row[i] if i is not None and i < len(row) else None
                        rec[c] = v
                    if tab == "Change_Log":
                        fv = rec["Field"]
                        fld = (fv[0] if isinstance(fv, tuple) else (fv or "")).lower()
                        if fld not in CHANGELOG_FIELDS and "sprint" not in fld and "story point" not in fld:
                            continue
                    out = [unescape(v[0]) if isinstance(v, tuple) else v for v in (rec[c] for c in KEEP[tab])]
                    if tab == "Issue":
                        d = row[idx["Description"]] if "Description" in idx else None
                        dc = row[idx["Description_Code"]] if "Description_Code" in idx else None
                        out += [len(d[0]) if isinstance(d, tuple) else 0,
                                len(dc[0]) if isinstance(dc, tuple) else 0]
                    w.writerow(["" if v is None else v for v in out])
                    contagem[tab] += 1
                elif punct == ",":
                    pass
                elif depth >= 1:
                    if null:
                        row.append(None)
                    elif s is not None:
                        row.append((s,))   # marcado como string (ainda escapada)
                    else:
                        row.append(lit)
            if time.time() - ultimo[0] > 15:
                ultimo[0] = time.time()
                print(f"  ... {tab}: {contagem[tab]:,} linhas  ({time.time()-t0:,.0f}s)", flush=True)

    for fh in files.values():
        fh.close()
    print("\nConcluído em %.0f s. Arquivos em %s" % (time.time() - t0, SAIDA))
    for tab, n in contagem.items():
        print(f"  {tab}.csv: {n:,} linhas")


if __name__ == "__main__":
    main()
