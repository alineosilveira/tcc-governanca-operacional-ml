"""
preparar_bases.py
-----------------
Gera, a partir dos CSVs extraídos da TAWOS (pasta tawos_csv/), as duas bases analíticas do trabalho:

  bases/base_bugs.csv      -> 1 linha por bug corrigido (Modelo 1: tempo de correção)
  bases/base_releases.csv  -> 1 linha por release publicada (Modelo 2: risco da release)
  bases/bases_tcc.xlsx     -> as duas bases + dicionário de dados

Regra de ouro: as variáveis explicativas (features) usam APENAS informação disponível
no momento em que o bug foi aberto / a release foi publicada (sem vazamento de dados).

Uso:  python preparar_bases.py
"""
import os
import re
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(AQUI, "tawos_csv")
OUT = os.path.join(AQUI, "bases")
os.makedirs(OUT, exist_ok=True)

# Projetos com versões numeradas (x.y.z), mistura de features e bugs e bom volume de releases.
# (Os projetos da Atlassian ficaram de fora porque lá features entram como "Suggestion" e não
#  são vinculadas a releases; outros projetos têm poucas versões publicadas.)
PROJETOS = ["MDL", "SERVER", "TIMOB", "MULE", "NEXUS", "TISTUD", "MESOS", "JAVA", "FAB", "DNN"]

JANELA_POS_RELEASE = 180          # dias após a release em que um bug "afetando" a versão é contado
TIPOS_FEATURE = {"New Feature", "Story", "Improvement", "Enhancement Request"}
RESOLUCOES_CORRIGIDO = {"Fixed", "Done", "Complete", "Resolved", "Implemented"}

# Prioridade normalizada (1 = trivial ... 5 = bloqueante) — os projetos usam esquemas diferentes
PRIORIDADE = {
    "Trivial": 1, "Trivial - P5": 1, "Lowest": 1,
    "Minor": 2, "Minor - P4": 2, "Low": 2,
    "Major": 3, "Major - P3": 3, "Medium": 3, "None": 3, "Not a Priority": 1,
    "Critical": 4, "Critical - P2": 4, "High": 4,
    "Blocker": 5, "Blocker - P1": 5, "Highest": 5,
}


def ler(nome, **kw):
    return pd.read_csv(os.path.join(SRC, nome), low_memory=False, **kw)


def classificar_versao(nome):
    """Retorna (tipo_release, major, minor, patch) a partir do nome da versão."""
    s = str(nome)
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", s)
    if not m:
        return None, np.nan, np.nan, np.nan
    ma, mi, pa = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if re.search(r"(rc|alpha|beta|\bM\d|preview|-m\d|milestone|itr|snapshot|ea\b)", s, re.I):
        tipo = "pre-release"
    elif mi == 0 and pa == 0:
        tipo = "major"
    elif pa == 0:
        tipo = "minor"
    else:
        tipo = "patch"
    return tipo, ma, mi, pa


DIAG, FUNIL = {}, []


def montar_sprints(iss, chg, proj):
    """Reconstrói, a partir do histórico do campo Sprint, o que foi planejado, adicionado, removido,
    concluído e o que "escorregou" (spillover) em cada sprint fechado."""
    TOL = pd.Timedelta(days=1)   # tolerância: itens incluídos até 1 dia após o início contam como planejados
    spr = ler("Sprint.csv").merge(proj[["Project_ID", "Project_Key"]], on="Project_ID")
    for c in ["Start_Date", "End_Date", "Complete_Date"]:
        spr[c] = pd.to_datetime(spr[c], errors="coerce")
    FUNIL.append(("Sprints", "Sprints registrados nos projetos selecionados", len(spr)))
    spr = spr[(spr.State == "CLOSED") & spr.Start_Date.notna() & spr.Complete_Date.notna()]
    spr = spr[spr.Complete_Date > spr.Start_Date]
    FUNIL.append(("Sprints", "encerrados e com datas de início e fim", len(spr)))
    dup = spr.duplicated(["Project_Key", "Name"], keep=False)       # nomes ambíguos no mesmo projeto
    spr = spr[~dup].rename(columns={"ID": "Sprint_ID", "Name": "Sprint"})
    FUNIL.append(("Sprints", "com nome não ambíguo no projeto", len(spr)))

    ev = chg[chg.Field == "Sprint"][["Issue_ID", "From_String", "To_String", "Creation_Date"]].copy()
    ev = ev.merge(iss[["ID", "Project_Key"]], left_on="Issue_ID", right_on="ID").sort_values("Creation_Date")
    tok = lambda s: set() if pd.isna(s) else {t.strip() for t in str(s).split(",") if t.strip()}
    linhas = []
    for r in ev.itertuples(index=False):
        de, para = tok(r.From_String), tok(r.To_String)
        for n in para - de:
            linhas.append((r.Issue_ID, r.Project_Key, n, r.Creation_Date, 1))
        for n in de - para:
            linhas.append((r.Issue_ID, r.Project_Key, n, r.Creation_Date, -1))
    # o Jira não registra no histórico o sprint definido já na criação do item: recupera-se esse valor inicial
    # (a) pelo "From" do primeiro evento de sprint e (b) pelo sprint final dos itens sem nenhum evento de sprint
    criado = iss.set_index("ID").Creation_Date
    prim = ev.drop_duplicates("Issue_ID")
    for r in prim.itertuples(index=False):
        for n in tok(r.From_String):
            linhas.append((r.Issue_ID, r.Project_Key, n, criado[r.Issue_ID], 1))
    nomes = spr.set_index("Sprint_ID").Sprint
    sem_ev = iss[iss.Sprint_ID.notna() & ~iss.ID.isin(ev.Issue_ID)]
    for r in sem_ev.itertuples(index=False):
        sid = int(r.Sprint_ID)
        if sid in nomes.index:
            linhas.append((r.ID, r.Project_Key, nomes[sid], r.Creation_Date, 1))
    DIAG["Sprints: itens com sprint definido na criação (recuperados)"] = int(len(sem_ev) + prim.From_String.notna().sum())
    mv = pd.DataFrame(linhas, columns=["Issue_ID", "Project_Key", "Sprint", "Data", "Mov"]).sort_values("Data", kind="stable")
    mv = mv.merge(spr[["Project_Key", "Sprint", "Sprint_ID", "Start_Date", "Complete_Date"]], on=["Project_Key", "Sprint"])

    def estado_em(g, t):
        g = g[g.Data <= t]
        return len(g) > 0 and g.Mov.iloc[-1] == 1

    it = iss.set_index("ID")
    reg = []
    for (sid, iid), g in mv.groupby(["Sprint_ID", "Issue_ID"], sort=False):
        ini, fim = g.Start_Date.iloc[0], g.Complete_Date.iloc[0]
        planejado = estado_em(g, ini + TOL)
        no_fim = estado_em(g, fim)
        adicionado = (not planejado) and ((g.Mov == 1) & (g.Data > ini + TOL) & (g.Data <= fim)).any()
        removido = (planejado or adicionado) and not no_fim
        res = it.at[iid, "Resolution_Date"]
        if pd.notna(res) and res < ini:          # já estava resolvido antes do sprint começar
            continue
        concluido = no_fim and pd.notna(res) and res <= fim + TOL
        reg.append((sid, iid, planejado, adicionado, removido, no_fim, concluido))
    m = pd.DataFrame(reg, columns=["Sprint_ID", "Issue_ID", "Planejado", "Adicionado_Durante",
                                   "Removido_Durante", "No_Fim", "Concluido"])
    m = m.join(it[["Type", "Story_Point", "Prioridade_Nivel", "Desc_Len", "Qtd_Componentes", "Assignee_ID"]], on="Issue_ID")
    m["Escorregou"] = (m.No_Fim & ~m.Concluido)
    m["SP"] = m.Story_Point.fillna(0)

    agg = m.groupby("Sprint_ID").apply(lambda d: pd.Series({
        "Itens_Planejados": d.Planejado.sum(),
        "Itens_Adicionados_Durante": d.Adicionado_Durante.sum(),
        "Itens_Removidos_Durante": d.Removido_Durante.sum(),
        "Itens_No_Fim": d.No_Fim.sum(),
        "Itens_Concluidos": d.Concluido.sum(),
        "Itens_Escorregaram": d.Escorregou.sum(),
        "Planejados_Concluidos": (d.Planejado & d.Concluido).sum(),
        "SP_Planejados": d.loc[d.Planejado, "SP"].sum(),
        "SP_Concluidos": d.loc[d.Concluido, "SP"].sum(),
        "SP_Planejados_Concluidos": d.loc[d.Planejado & d.Concluido, "SP"].sum(),
        "Pct_Bugs_Planejados": (d.loc[d.Planejado, "Type"] == "Bug").mean() if d.Planejado.any() else np.nan,
        "Pessoas_No_Sprint": d.loc[d.Planejado | d.Adicionado_Durante, "Assignee_ID"].nunique(),
    }))
    s = spr.set_index("Sprint_ID").join(agg, how="inner").reset_index()
    FUNIL.append(("Sprints", "com itens reconstruídos pelo histórico", len(s)))
    s["Duracao_Dias"] = (s.Complete_Date - s.Start_Date).dt.total_seconds() / 86400
    s = s[(s.Itens_Planejados >= 3) & (s.Duracao_Dias >= 3) & (s.Duracao_Dias <= 60)]
    FUNIL.append(("Sprints", "com pelo menos três itens planejados e duração de 3 a 60 dias", len(s)))
    # validação da reconstrução (parecer C11): itens cujo sprint final (Issue.Sprint_ID) é um sprint incluído
    # devem aparecer na reconstrução como presentes nesse sprint no fechamento
    fin = iss[iss.Sprint_ID.isin(s.Sprint_ID)][["ID", "Sprint_ID"]].rename(columns={"ID": "Issue_ID"})
    fin["Sprint_ID"] = fin.Sprint_ID.astype(int)
    chk = fin.merge(m[["Sprint_ID", "Issue_ID", "No_Fim"]], on=["Sprint_ID", "Issue_ID"], how="left")
    DIAG["Validação sprints: itens cujo sprint final é um sprint incluído"] = len(chk)
    DIAG["Validação sprints: % reconstruídos como presentes no fechamento desse sprint"] = round(100 * chk.No_Fim.fillna(False).astype(bool).mean(), 1)
    s["Taxa_Cumprimento"] = s.Planejados_Concluidos / s.Itens_Planejados
    s["Taxa_Escorregamento"] = s.Itens_Escorregaram / s.Itens_No_Fim.replace(0, np.nan)
    s["Pct_Escopo_Adicionado"] = s.Itens_Adicionados_Durante / s.Itens_Planejados
    s["Pct_SP_Entregue"] = s.SP_Planejados_Concluidos / s.SP_Planejados.replace(0, np.nan)
    s = s.sort_values(["Project_Key", "Start_Date"])
    s["Velocity_SP"] = s.SP_Concluidos
    s["Velocity_Media_3_Anteriores"] = s.groupby("Project_Key").SP_Concluidos.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    s["Taxa_Cumprimento_Anterior"] = s.groupby("Project_Key").Taxa_Cumprimento.shift(1)
    cols_s = ["Sprint_ID", "Project_Key", "Sprint", "Start_Date", "Complete_Date", "Duracao_Dias",
              "Itens_Planejados", "Itens_Adicionados_Durante", "Itens_Removidos_Durante", "Itens_No_Fim",
              "Itens_Concluidos", "Planejados_Concluidos", "Itens_Escorregaram", "SP_Planejados", "SP_Concluidos", "SP_Planejados_Concluidos",
              "Pct_Bugs_Planejados", "Pessoas_No_Sprint", "Taxa_Cumprimento", "Taxa_Escorregamento",
              "Pct_Escopo_Adicionado", "Pct_SP_Entregue", "Velocity_SP", "Velocity_Media_3_Anteriores",
              "Taxa_Cumprimento_Anterior"]
    s = s[cols_s]

    # nível de item: cada item PLANEJADO no início do sprint → foi concluído dentro do sprint?
    p = m[m.Planejado & m.Sprint_ID.isin(s.Sprint_ID)].merge(
        s[["Sprint_ID", "Project_Key", "Start_Date", "Duracao_Dias", "Itens_Planejados", "SP_Planejados",
           "Pessoas_No_Sprint", "Taxa_Cumprimento_Anterior"]], on="Sprint_ID")
    p["Itens_Do_Responsavel_No_Sprint"] = p.groupby(["Sprint_ID", "Assignee_ID"]).Issue_ID.transform("size")
    p.loc[p.Assignee_ID.isna(), "Itens_Do_Responsavel_No_Sprint"] = np.nan
    p["Tem_Responsavel"] = p.Assignee_ID.notna().astype(int)
    p["Tem_Story_Point"] = p.Story_Point.notna().astype(int)
    p["Vezes_Em_Sprints_Anteriores"] = 0
    # quantos sprints anteriores o item já tinha passado (carregado de sprint em sprint)
    ordem = m.merge(s[["Sprint_ID", "Start_Date"]], on="Sprint_ID").sort_values("Start_Date")
    ordem["n_ant"] = ordem.groupby("Issue_ID").cumcount()
    p = p.merge(ordem[["Sprint_ID", "Issue_ID", "n_ant"]], on=["Sprint_ID", "Issue_ID"], how="left")
    p["Vezes_Em_Sprints_Anteriores"] = p.n_ant.fillna(0).astype(int)
    p["Tipo_Item"] = np.select([p.Type.eq("Bug"), p.Type.isin(TIPOS_FEATURE)], ["bug", "feature"], "outro")
    p["Nao_Concluiu_No_Sprint"] = (~p.Concluido).astype(int)
    cols_p = ["Sprint_ID", "Issue_ID", "Project_Key", "Start_Date", "Tipo_Item", "Prioridade_Nivel", "Story_Point",
              "Tem_Story_Point", "Desc_Len", "Qtd_Componentes", "Tem_Responsavel", "Itens_Do_Responsavel_No_Sprint",
              "Vezes_Em_Sprints_Anteriores", "Duracao_Dias", "Itens_Planejados", "SP_Planejados",
              "Pessoas_No_Sprint", "Taxa_Cumprimento_Anterior", "Removido_Durante", "Nao_Concluiu_No_Sprint"]
    p = p[cols_p].rename(columns={"Desc_Len": "Tamanho_Descricao", "Duracao_Dias": "Duracao_Sprint_Dias"})
    FUNIL.append(("Itens de sprint", "Itens planejados no início dos sprints incluídos", len(p)))
    p["Removido_Durante"] = p.Removido_Durante.astype(int)
    return s, p.sort_values("Start_Date")


def main():
    global DIAG, FUNIL
    DIAG, FUNIL = {}, []
    print("Lendo CSVs da TAWOS...")
    proj = ler("Project.csv")[["ID", "Project_Key", "Last_Update_Date"]].rename(columns={"ID": "Project_ID"})
    proj = proj[proj.Project_Key.isin(PROJETOS)]
    proj["Last_Update_Date"] = pd.to_datetime(proj.Last_Update_Date)

    iss = ler("Issue.csv")
    iss = iss.merge(proj[["Project_ID", "Project_Key"]], on="Project_ID")
    for c in ["Creation_Date", "Resolution_Date"]:
        iss[c] = pd.to_datetime(iss[c], errors="coerce")
    iss["Title_Len"] = iss.Title.fillna("").str.len()

    # Prioridade no momento da abertura: se houve mudança de prioridade, a original é o From_String da 1ª mudança
    chg = ler("Change_Log.csv")
    chg = chg[chg.Issue_ID.isin(iss.ID)]
    chg["Creation_Date"] = pd.to_datetime(chg.Creation_Date, errors="coerce")
    pri_chg = chg[chg.Field == "priority"].sort_values("Creation_Date").drop_duplicates("Issue_ID")
    pri_ini = pri_chg.set_index("Issue_ID").From_String
    iss["Priority_Inicial"] = iss.ID.map(pri_ini).fillna(iss.Priority)
    iss["Prioridade_Nivel"] = iss.Priority_Inicial.map(PRIORIDADE).fillna(3).astype(int)
    reab = chg[(chg.Field == "status") & (chg.To_String == "Reopened")].groupby("Issue_ID").size()
    iss["Qtd_Reaberturas"] = iss.ID.map(reab).fillna(0).astype(int)

    comp = ler("Issue_Component.csv")
    iss["Qtd_Componentes"] = iss.ID.map(comp.groupby("Issue_ID").size()).fillna(0).astype(int)

    # ---------------- Versões / releases ----------------
    ver = ler("Version.csv").rename(columns={"ID": "Version_ID", "Name": "Versao"})
    ver = ver.merge(proj, on="Project_ID")
    ver["Release_Date"] = pd.to_datetime(ver.Release_Date, errors="coerce")
    FUNIL.append(("Releases", "Versões cadastradas nos dez projetos", len(ver)))
    ver = ver[(ver.Released == 1) & ver.Release_Date.notna()].copy()
    FUNIL.append(("Releases", "publicadas e com data de publicação", len(ver)))
    cls = ver.Versao.apply(classificar_versao)
    ver["Tipo_Release"] = [c[0] for c in cls]
    ver["V_Major"] = [c[1] for c in cls]
    ver["V_Minor"] = [c[2] for c in cls]
    ver["V_Patch"] = [c[3] for c in cls]
    ver = ver[ver.Tipo_Release.notna()].sort_values(["Project_Key", "Release_Date"])
    FUNIL.append(("Releases", "com numeração identificável (x.y.z)", len(ver)))
    ver["Dias_Desde_Release_Anterior"] = ver.groupby("Project_Key").Release_Date.diff().dt.days
    # só releases com janela pós-release completa dentro do período coberto pela base
    ver["Janela_Completa"] = ver.Release_Date + pd.Timedelta(days=JANELA_POS_RELEASE) <= ver.Last_Update_Date

    fixv = ler("Fix_Version.csv").rename(columns={"Fix_Version_ID": "Version_ID"})
    affv = ler("Affected_Version.csv").rename(columns={"Affected_Version_ID": "Version_ID"})
    fixv = fixv[fixv.Version_ID.isin(ver.Version_ID) & fixv.Issue_ID.isin(iss.ID)]
    affv = affv[affv.Version_ID.isin(ver.Version_ID) & affv.Issue_ID.isin(iss.ID)]

    # ---- reconstrução histórica pelo Change_Log (parecer C6) ----
    # nomes de versão -> ID (por projeto) para interpretar os eventos "Fix Version" e "Version"
    vfull = ler("Version.csv")[["ID", "Name", "Project_ID"]].merge(proj[["Project_ID", "Project_Key"]], on="Project_ID")
    vfull = vfull.drop_duplicates(["Project_Key", "Name"], keep=False)
    vmap = {(k, n): i for k, n, i in zip(vfull.Project_Key, vfull.Name, vfull.ID)}
    evs = chg[chg.Field.isin(["Fix Version", "Version"])].merge(
        iss[["ID", "Project_Key", "Creation_Date"]].rename(columns={"ID": "Issue_ID", "Creation_Date": "Criado"}), on="Issue_ID")
    evs["id_para"] = [vmap.get((k, n)) for k, n in zip(evs.Project_Key, evs.To_String)]
    evs["id_de"] = [vmap.get((k, n)) for k, n in zip(evs.Project_Key, evs.From_String)]

    # (a) composição da release NA DATA DE PUBLICAÇÃO: retira itens incluídos depois e devolve os removidos depois
    rd = ver.set_index("Version_ID").Release_Date
    fe = evs[evs.Field == "Fix Version"]
    add = fe[fe.id_para.notna()].assign(Version_ID=lambda d: d.id_para.astype(int))
    add = add[add.Version_ID.isin(rd.index)]
    add = add[add.Creation_Date > add.Version_ID.map(rd) + pd.Timedelta(days=1)]
    rem = fe[fe.id_de.notna()].assign(Version_ID=lambda d: d.id_de.astype(int))
    rem = rem[rem.Version_ID.isin(rd.index)]
    rem = rem[rem.Creation_Date > rem.Version_ID.map(rd) + pd.Timedelta(days=1)]
    k_add = set(zip(add.Issue_ID, add.Version_ID))
    fixv_final = fixv.copy()
    fixv = fixv[[t not in k_add for t in zip(fixv.Issue_ID, fixv.Version_ID)]]
    fixv = pd.concat([fixv, rem[["Issue_ID", "Version_ID"]]]).drop_duplicates()
    fixv = fixv[fixv.Issue_ID.isin(iss.ID)]
    DIAG["Vínculos de fix version (estado final)"] = len(fixv_final)
    DIAG["Vínculos incluídos depois da publicação (retirados)"] = len(fixv_final) - len(fixv_final[[t not in k_add for t in zip(fixv_final.Issue_ID, fixv_final.Version_ID)]])
    DIAG["Vínculos removidos depois da publicação (devolvidos)"] = len(set(zip(rem.Issue_ID, rem.Version_ID)) - set(zip(fixv_final.Issue_ID, fixv_final.Version_ID)))

    # (b) versões afetadas NO MOMENTO DA ABERTURA do bug (mudanças até 1 hora após a abertura contam como abertura)
    ae = evs[(evs.Field == "Version") & (evs.Creation_Date > evs.Criado + pd.Timedelta(hours=1))]
    a_add = set(zip(ae.loc[ae.id_para.notna(), "Issue_ID"], ae.loc[ae.id_para.notna(), "id_para"].astype(int)))
    a_rem = ae[ae.id_de.notna()][["Issue_ID", "id_de"]].rename(columns={"id_de": "Version_ID"})
    a_rem["Version_ID"] = a_rem.Version_ID.astype(int)
    affv_final = affv.copy()
    affv_ini = affv[[t not in a_add for t in zip(affv.Issue_ID, affv.Version_ID)]]
    affv_ini = pd.concat([affv_ini, a_rem[a_rem.Version_ID.isin(ver.Version_ID)]]).drop_duplicates()
    DIAG["Bugs com versão afetada alterada após a abertura"] = ae.Issue_ID.nunique()

    # composição de cada release (o que foi entregue nela)
    it = iss.set_index("ID")
    f = fixv.join(it[["Type", "Story_Point", "Prioridade_Nivel", "Desc_Len"]], on="Issue_ID")
    f["is_feature"] = f.Type.isin(TIPOS_FEATURE)
    f["is_bugfix"] = f.Type.eq("Bug")
    f["is_alta"] = f.Prioridade_Nivel >= 4
    comp_rel = f.merge(comp, on="Issue_ID").groupby("Version_ID").Component_ID.nunique()
    agg = f.groupby("Version_ID").agg(
        Qtd_Itens=("Issue_ID", "nunique"),
        Qtd_Features=("is_feature", "sum"),
        Qtd_Correcoes=("is_bugfix", "sum"),
        Qtd_Itens_Alta_Prioridade=("is_alta", "sum"),
        Story_Points_Total=("Story_Point", "sum"),
        Tamanho_Medio_Descricao=("Desc_Len", "mean"),
    )
    agg["Qtd_Outros"] = agg.Qtd_Itens - agg.Qtd_Features - agg.Qtd_Correcoes
    agg["Pct_Features"] = agg.Qtd_Features / agg.Qtd_Itens
    agg["Pct_Correcoes"] = agg.Qtd_Correcoes / agg.Qtd_Itens
    agg["Qtd_Componentes_Tocados"] = comp_rel
    agg["Qtd_Itens_Estado_Final"] = fixv_final.groupby("Version_ID").Issue_ID.nunique()
    rel = ver.set_index("Version_ID").join(agg, how="inner")
    rel["Qtd_Componentes_Tocados"] = rel.Qtd_Componentes_Tocados.fillna(0)

    # bugs pós-release: cada bug é atribuído à versão afetada MAIS ANTIGA (a que primeiro "carregou" o
    # defeito), sem contar o mesmo bug em várias releases; conta se foi aberto até N dias após a publicação
    bugs_all = iss[iss.Type == "Bug"].set_index("ID")
    a = affv.join(ver.set_index("Version_ID")[["Release_Date"]], on="Version_ID", how="inner")
    a = a.sort_values("Release_Date").drop_duplicates("Issue_ID")[["Issue_ID", "Version_ID"]]
    a = a.join(bugs_all[["Creation_Date", "Resolution_Date", "Prioridade_Nivel", "Resolution"]], on="Issue_ID", how="inner")
    a = a.join(rel[["Release_Date"]], on="Version_ID", how="inner")
    a["dias"] = (a.Creation_Date - a.Release_Date).dt.total_seconds() / 86400
    a = a[(a.dias >= 0) & (a.dias <= JANELA_POS_RELEASE)]
    a["lead"] = (a.Resolution_Date - a.Creation_Date).dt.total_seconds() / 86400
    a["alta"] = a.Prioridade_Nivel >= 4
    pos = a.groupby("Version_ID").agg(
        Bugs_Pos_Release=("Issue_ID", "nunique"),
        Bugs_Pos_Release_Alta=("alta", "sum"),
        Dias_Medianos_Correcao_Pos_Release=("lead", "median"),
    )
    rel = rel.join(pos)
    rel[["Bugs_Pos_Release", "Bugs_Pos_Release_Alta"]] = rel[["Bugs_Pos_Release", "Bugs_Pos_Release_Alta"]].fillna(0).astype(int)
    rel["Bugs_Por_Item_Entregue"] = rel.Bugs_Pos_Release / rel.Qtd_Itens

    # sensibilidade (parecer C7): janelas de 90 e 365 dias e regra "conta em todas as versões afetadas"
    a_all = affv.join(bugs_all[["Creation_Date"]], on="Issue_ID", how="inner").join(rel[["Release_Date"]], on="Version_ID", how="inner")
    a_all["dias"] = (a_all.Creation_Date - a_all.Release_Date).dt.total_seconds() / 86400
    a_one = affv.join(ver.set_index("Version_ID")[["Release_Date"]], on="Version_ID", how="inner")
    a_one = a_one.sort_values("Release_Date").drop_duplicates("Issue_ID")[["Issue_ID", "Version_ID"]]
    a_one = a_one.join(bugs_all[["Creation_Date"]], on="Issue_ID", how="inner").join(rel[["Release_Date"]], on="Version_ID", how="inner")
    a_one["dias"] = (a_one.Creation_Date - a_one.Release_Date).dt.total_seconds() / 86400
    for jan in (90, 365):
        rel[f"Bugs_Pos_Release_{jan}d"] = a_one[(a_one.dias >= 0) & (a_one.dias <= jan)].groupby("Version_ID").size()
    rel["Bugs_Pos_Release_Todas_Versoes"] = a_all[(a_all.dias >= 0) & (a_all.dias <= JANELA_POS_RELEASE)].groupby("Version_ID").Issue_ID.nunique()
    for c in ["Bugs_Pos_Release_90d", "Bugs_Pos_Release_365d", "Bugs_Pos_Release_Todas_Versoes"]:
        rel[c] = rel[c].fillna(0).astype(int)
    rel["Janela_365d_Completa"] = rel.Release_Date + pd.Timedelta(days=365) <= rel.Last_Update_Date

    n0 = len(rel)
    rel = rel[rel.Janela_Completa]
    n1 = len(rel)
    rel = rel[rel.Qtd_Itens >= 3].copy()
    n2 = len(rel)
    # alvo binário SEM informação futura (parecer C5): a release é de alto risco quando seus bugs pós-release
    # superam a mediana das releases ANTERIORES do mesmo projeto (mínimo de três releases anteriores)
    rel = rel.sort_values(["Project_Key", "Release_Date"])
    rel["Mediana_Historica_Bugs"] = rel.groupby("Project_Key").Bugs_Pos_Release.transform(
        lambda x: x.shift(1).expanding(min_periods=3).median())
    rel["Alto_Risco"] = (rel.Bugs_Pos_Release > rel.Mediana_Historica_Bugs).astype(int)
    rel.loc[rel.Mediana_Historica_Bugs.isna(), "Alto_Risco"] = -1
    FUNIL += [("Releases", "com ao menos um item entregue", n0),
              ("Releases", "com 180 dias completos de acompanhamento", n1),
              ("Releases", "com pelo menos três itens entregues na data de publicação", n2),
              ("Releases", "com ao menos três releases anteriores no projeto (alvo de alto risco)", int((rel.Alto_Risco >= 0).sum()))]
    rel["Ano"] = rel.Release_Date.dt.year
    rel = rel.reset_index()
    cols_rel = ["Version_ID", "Project_Key", "Versao", "Release_Date", "Ano", "Tipo_Release",
                "V_Major", "V_Minor", "V_Patch", "Dias_Desde_Release_Anterior",
                "Qtd_Itens", "Qtd_Features", "Qtd_Correcoes", "Qtd_Outros", "Pct_Features", "Pct_Correcoes",
                "Qtd_Itens_Alta_Prioridade", "Story_Points_Total", "Qtd_Componentes_Tocados",
                "Tamanho_Medio_Descricao",
                "Qtd_Itens_Estado_Final",
                "Bugs_Pos_Release", "Bugs_Pos_Release_Alta", "Dias_Medianos_Correcao_Pos_Release",
                "Bugs_Por_Item_Entregue", "Bugs_Pos_Release_90d", "Bugs_Pos_Release_365d", "Janela_365d_Completa",
                "Bugs_Pos_Release_Todas_Versoes", "Mediana_Historica_Bugs", "Alto_Risco"]
    rel = rel[cols_rel].sort_values(["Project_Key", "Release_Date"])

    # ---------------- Base de bugs ----------------
    # todos os bugs abertos até 12 meses antes do fim da coleta de cada projeto
    # (os corrigidos formam a base_bugs; todos, com censura, formam a base de sobrevivência)
    b = iss[iss.Type == "Bug"].copy()
    FUNIL.append(("Bugs", "Bugs registrados nos dez projetos", len(b)))
    b = b[~(b.Resolution_Date.notna() & (b.Resolution_Date < b.Creation_Date))]   # datas inconsistentes
    FUNIL.append(("Bugs", "com datas consistentes (resolução não anterior à abertura)", len(b)))
    b = b.merge(proj[["Project_Key", "Last_Update_Date"]], on="Project_Key")
    b = b[b.Creation_Date <= b.Last_Update_Date - pd.Timedelta(days=365)]
    FUNIL.append(("Bugs", "abertos até 12 meses antes do fim da coleta (base de sobrevivência)", len(b)))
    b["Dias_Para_Corrigir"] = (b.Resolution_Date - b.Creation_Date).dt.total_seconds() / 86400
    b["Evento_Corrigido"] = (b.Resolution.isin(RESOLUCOES_CORRIGIDO) & b.Resolution_Date.notna()).astype(int)
    fim = b.Resolution_Date.where(b.Resolution_Date.notna(), b.Last_Update_Date)
    b["Tempo_Dias"] = (fim - b.Creation_Date).dt.total_seconds() / 86400
    b["Situacao_Final"] = np.select([b.Evento_Corrigido == 1, b.Resolution_Date.notna()],
                                    ["corrigido", "encerrado_sem_correcao"], "aberto_na_coleta")

    # release afetada (a mais antiga publicada entre as versões afetadas), reconstruída para o momento da
    # abertura do bug; as versões do estado final ficam em colunas *_Final, só para a análise de sensibilidade
    def release_afetada(av_src, suf):
        av = av_src.join(ver.set_index("Version_ID")[["Release_Date", "Tipo_Release"]], on="Version_ID", how="inner")
        av = av.sort_values("Release_Date").drop_duplicates("Issue_ID")
        av = av.join(agg[["Qtd_Itens", "Pct_Features"]], on="Version_ID").set_index("Issue_ID")
        b["Qtd_Versoes_Afetadas" + suf] = b.ID.map(av_src.groupby("Issue_ID").size()).fillna(0).astype(int)
        b["Tipo_Release_Afetada" + suf] = b.ID.map(av.Tipo_Release).fillna("sem-versao")
        b["Dias_Desde_Release_Afetada" + suf] = (b.Creation_Date - b.ID.map(av.Release_Date)).dt.days
        b["Itens_Release_Afetada" + suf] = b.ID.map(av.Qtd_Itens)
        b["Pct_Features_Release_Afetada" + suf] = b.ID.map(av.Pct_Features)
    release_afetada(affv_ini, "")
    release_afetada(affv_final, "_Final")

    # carga do time: bugs abertos no projeto no momento da abertura (abertos antes e ainda não resolvidos)
    todos = iss[iss.Type == "Bug"][["ID", "Project_Key", "Creation_Date", "Resolution_Date"]]
    carga = []
    for k, g in b.groupby("Project_Key"):
        t = todos[todos.Project_Key == k]
        cri = np.sort(t.Creation_Date.values)
        res = np.sort(t.Resolution_Date.dropna().values)
        x = g.Creation_Date.values
        abertos = np.searchsorted(cri, x, side="left") - np.searchsorted(res, x, side="right")
        carga.append(pd.Series(abertos, index=g.index))
    b["Bugs_Abertos_No_Projeto"] = pd.concat(carga)

    # experiência de quem reportou: quantas issues essa pessoa já tinha aberto no projeto
    iss_sorted = iss.sort_values("Creation_Date")
    iss_sorted["exp"] = iss_sorted.groupby(["Project_Key", "Reporter_ID"]).cumcount()
    b["Experiencia_Reporter"] = b.ID.map(iss_sorted.set_index("ID").exp).fillna(0).astype(int)

    b["Mes_Abertura"] = b.Creation_Date.dt.month
    b["Dia_Semana_Abertura"] = b.Creation_Date.dt.dayofweek
    b["Tem_Codigo_Na_Descricao"] = (b.Desc_Code_Len > 0).astype(int)
    b["Tem_Story_Point"] = b.Story_Point.notna().astype(int)
    b["Horas_Apontadas"] = b.Timespent / 3600  # Timespent da Jira está em segundos (preenchido em poucos casos)
    b["Faixa_Correcao"] = pd.cut(b.Dias_Para_Corrigir, [-0.001, 7, 60, np.inf],
                                 labels=["ate_7_dias", "8_a_60_dias", "mais_de_60_dias"])

    # ---- Pergunta 5: carga do responsável (itens abertos atribuídos à mesma pessoa no dia da abertura)
    b["Tem_Responsavel"] = b.Assignee_ID.notna().astype(int)
    carga_resp = pd.Series(np.nan, index=b.index)
    tem = iss[iss.Assignee_ID.notna()][["Assignee_ID", "Creation_Date", "Resolution_Date"]]
    grupos = {k: (np.sort(g.Creation_Date.values), np.sort(g.Resolution_Date.dropna().values))
              for k, g in tem.groupby("Assignee_ID")}
    for k, g in b[b.Assignee_ID.notna()].groupby("Assignee_ID"):
        cri, res = grupos[k]
        x = g.Creation_Date.values
        carga_resp.loc[g.index] = np.searchsorted(cri, x, side="left") - np.searchsorted(res, x, side="right")
    b["Carga_Responsavel"] = carga_resp.clip(lower=0)
    # concentração: participação do responsável nos bugs resolvidos do projeto (quanto o time depende dele)
    share = iss[(iss.Type == "Bug") & iss.Assignee_ID.notna()].groupby(["Project_Key", "Assignee_ID"]).size()
    share = share / share.groupby(level=0).transform("sum")
    b["Participacao_Responsavel_No_Projeto"] = [share.get((p, a), np.nan) for p, a in zip(b.Project_Key, b.Assignee_ID)]

    # ---- Pergunta 7: reabertura (alvo)
    b["Foi_Reaberto"] = (b.Qtd_Reaberturas > 0).astype(int)

    # ---- Pergunta 8: proxies de impacto em cliente (alvos)
    lk = ler("Issue_Link.csv")
    dup = lk[(lk.Name == "Duplicate") & (lk.Direction == "INBOUND")].groupby("Issue_ID").size()
    b["Qtd_Duplicados"] = b.ID.map(dup).fillna(0).astype(int)
    pc = chg[chg.Field == "priority"].copy()
    pc["de"], pc["para"] = pc.From_String.map(PRIORIDADE), pc.To_String.map(PRIORIDADE)
    esc = pc[pc.de.notna() & pc.para.notna() & (pc.para > pc.de)].groupby("Issue_ID").size()
    b["Escalou_Prioridade"] = b.ID.map(esc).fillna(0).gt(0).astype(int)
    b["Alta_Criticidade"] = ((b.Qtd_Duplicados > 0) | (b.Escalou_Prioridade == 1)).astype(int)

    cols_b = ["ID", "Issue_Key", "Project_Key", "Creation_Date", "Resolution_Date",
              "Priority_Inicial", "Prioridade_Nivel", "Title_Len", "Desc_Len", "Desc_Code_Len",
              "Tem_Codigo_Na_Descricao", "Qtd_Componentes", "Qtd_Versoes_Afetadas",
              "Tipo_Release_Afetada", "Dias_Desde_Release_Afetada", "Itens_Release_Afetada",
              "Pct_Features_Release_Afetada", "Qtd_Versoes_Afetadas_Final", "Tipo_Release_Afetada_Final",
              "Dias_Desde_Release_Afetada_Final", "Itens_Release_Afetada_Final", "Pct_Features_Release_Afetada_Final",
              "Bugs_Abertos_No_Projeto", "Experiencia_Reporter",
              "Mes_Abertura", "Dia_Semana_Abertura", "Tem_Story_Point",
              "Assignee_ID", "Tem_Responsavel", "Carga_Responsavel", "Participacao_Responsavel_No_Projeto",
              "Horas_Apontadas", "Qtd_Reaberturas", "Foi_Reaberto", "Qtd_Duplicados", "Escalou_Prioridade",
              "Alta_Criticidade", "Dias_Para_Corrigir", "Faixa_Correcao"]
    ren = {"ID": "Issue_ID", "Assignee_ID": "Responsavel_ID", "Title_Len": "Tamanho_Titulo",
           "Desc_Len": "Tamanho_Descricao", "Desc_Code_Len": "Tamanho_Codigo_Descricao"}
    cols_s = [c for c in cols_b if c not in ("Resolution_Date", "Horas_Apontadas", "Qtd_Reaberturas", "Foi_Reaberto",
                                             "Qtd_Duplicados", "Escalou_Prioridade", "Alta_Criticidade",
                                             "Dias_Para_Corrigir", "Faixa_Correcao")]
    surv = b[cols_s + ["Situacao_Final", "Evento_Corrigido", "Tempo_Dias"]].rename(columns=ren).sort_values("Creation_Date")
    surv.to_csv(os.path.join(OUT, "base_bugs_sobrevivencia.csv"), index=False)
    print(f"base_bugs_sobrevivencia: {len(surv):,} bugs ({surv.Evento_Corrigido.mean():.1%} corrigidos)")
    b = b[b.Evento_Corrigido == 1]
    FUNIL.append(("Bugs", "corrigidos, com data de resolução (base_bugs)", len(b)))
    b = b[cols_b].rename(columns=ren)
    b = b.sort_values("Creation_Date")

    b.to_csv(os.path.join(OUT, "base_bugs.csv"), index=False)
    rel.to_csv(os.path.join(OUT, "base_releases.csv"), index=False)

    # ---------------- Pergunta 3: capacidade (sustentação × evolução), mês a mês ----------------
    x = iss[iss.Resolution.isin(RESOLUCOES_CORRIGIDO) & iss.Resolution_Date.notna()].copy()
    x["Categoria"] = np.select([x.Type.eq("Bug"), x.Type.isin(TIPOS_FEATURE)], ["Sustentacao_Bugs", "Evolucao_Features"], "Outros")
    x["Mes"] = x.Resolution_Date.dt.to_period("M").dt.to_timestamp()
    cap = x.pivot_table(index=["Project_Key", "Mes"], columns="Categoria", values="ID", aggfunc="count", fill_value=0)
    abertos = iss[iss.Type == "Bug"].assign(Mes=lambda d: d.Creation_Date.dt.to_period("M").dt.to_timestamp())
    cap["Bugs_Abertos_No_Mes"] = abertos.groupby(["Project_Key", "Mes"]).size()
    cap = cap.fillna(0).astype(int).reset_index()
    for c in ["Sustentacao_Bugs", "Evolucao_Features", "Outros"]:
        if c not in cap:
            cap[c] = 0
    cap["Itens_Resolvidos"] = cap[["Sustentacao_Bugs", "Evolucao_Features", "Outros"]].sum(axis=1)
    cap["Pct_Sustentacao"] = cap.Sustentacao_Bugs / cap.Itens_Resolvidos.replace(0, np.nan)
    cap = cap.merge(proj[["Project_Key", "Last_Update_Date"]], on="Project_Key")
    cap = cap[cap.Mes < cap.Last_Update_Date.dt.to_period("M").dt.to_timestamp()].drop(columns="Last_Update_Date")
    cap.to_csv(os.path.join(OUT, "base_capacidade_mensal.csv"), index=False)

    # ---------------- Pergunta 1: sprints (planejado × entregue) ----------------
    spr_sprints, spr_itens = montar_sprints(iss, chg, proj)
    spr_sprints.to_csv(os.path.join(OUT, "base_sprints.csv"), index=False)
    spr_itens.to_csv(os.path.join(OUT, "base_itens_sprint.csv"), index=False)
    print(f"base_sprints:  {len(spr_sprints):,} sprints | base_itens_sprint: {len(spr_itens):,} itens planejados")

    eleg = pd.DataFrame(FUNIL, columns=["Base", "Etapa", "Registros"])
    eleg["Perda_na_Etapa"] = eleg.groupby("Base").Registros.diff().mul(-1)
    eleg.to_csv(os.path.join(OUT, "elegibilidade.csv"), index=False)
    diag = pd.DataFrame(list(DIAG.items()), columns=["Diagnóstico", "Valor"])
    diag.to_csv(os.path.join(OUT, "diagnosticos_reconstrucao.csv"), index=False)
    print(eleg.to_string(index=False)); print(diag.to_string(index=False))

    dicionario = pd.DataFrame([
        # base_bugs
        ("base_bugs", "Issue_ID / Issue_Key", "Identificador do bug na TAWOS / no Jira", "id"),
        ("base_bugs", "Project_Key", "Projeto de origem", "feature"),
        ("base_bugs", "Creation_Date / Resolution_Date", "Abertura e resolução do bug", "data"),
        ("base_bugs", "Priority_Inicial / Prioridade_Nivel", "Prioridade no momento da abertura (nível 1-5 normalizado)", "feature"),
        ("base_bugs", "Tamanho_Titulo / Tamanho_Descricao / Tamanho_Codigo_Descricao", "Tamanho do texto (caracteres)", "feature"),
        ("base_bugs", "Tem_Codigo_Na_Descricao", "1 se a descrição traz trecho de código/log", "feature"),
        ("base_bugs", "Qtd_Componentes", "Componentes do sistema associados ao bug", "feature"),
        ("base_bugs", "Qtd_Versoes_Afetadas", "Quantas versões o bug afeta", "feature"),
        ("base_bugs", "Tipo_Release_Afetada", "major / minor / patch / pre-release / sem-versao; versões afetadas reconstruídas para o momento da abertura pelo histórico de alterações", "feature"),
        ("base_bugs", "Dias_Desde_Release_Afetada", "Dias entre a publicação da release afetada e a abertura do bug", "feature"),
        ("base_bugs", "Itens_Release_Afetada / Pct_Features_Release_Afetada", "Tamanho e % de features da release afetada", "feature"),
        ("base_bugs", "Bugs_Abertos_No_Projeto", "Carga do time: bugs em aberto no projeto no dia da abertura", "feature"),
        ("base_bugs", "Experiencia_Reporter", "Issues abertas antes pela mesma pessoa no projeto", "feature"),
        ("base_bugs", "Mes_Abertura / Dia_Semana_Abertura", "Sazonalidade", "feature"),
        ("base_bugs", "Horas_Apontadas", "Horas apontadas no Jira (Timespent) - poucos preenchidos; NÃO usar como feature", "descritiva"),
        ("base_bugs", "Qtd_Reaberturas", "Quantas vezes foi reaberto - acontece depois; NÃO usar como feature", "descritiva"),
        ("base_bugs", "Dias_Para_Corrigir", "Dias entre abertura e resolução", "alvo (regressão)"),
        ("base_bugs", "Faixa_Correcao", "ate_7_dias / 8_a_60_dias / mais_de_60_dias", "alvo (classificação)"),
        # base_releases
        ("base_releases", "Version_ID / Versao / Release_Date", "Release publicada", "id"),
        ("base_releases", "Tipo_Release", "major / minor / patch / pre-release (pelo número da versão)", "feature"),
        ("base_releases", "Dias_Desde_Release_Anterior", "Intervalo para a release anterior do projeto", "feature"),
        ("base_releases", "Qtd_Itens / Qtd_Features / Qtd_Correcoes / Qtd_Outros", "Composição do que foi entregue (fix version) NA DATA DE PUBLICAÇÃO, reconstruída pelo histórico", "feature"),
        ("base_releases", "Pct_Features / Pct_Correcoes", "Mix da release", "feature"),
        ("base_releases", "Qtd_Itens_Alta_Prioridade", "Itens entregues com prioridade alta/bloqueante", "feature"),
        ("base_releases", "Story_Points_Total", "Soma de story points (só alguns projetos estimam)", "feature"),
        ("base_releases", "Qtd_Componentes_Tocados", "Componentes distintos alterados na release", "feature"),
        ("base_releases", "Bugs_Pos_Release", f"Bugs abertos até {JANELA_POS_RELEASE} dias após a release e marcados como afetando a versão", "alvo (contagem)"),
        ("base_releases", "Bugs_Pos_Release_Alta", "Idem, só prioridade alta/bloqueante (proxy de impacto em cliente)", "alvo / descritiva"),
        ("base_releases", "Dias_Medianos_Correcao_Pos_Release", "Mediana de dias para corrigir os bugs pós-release", "descritiva"),
        ("base_releases", "Bugs_Por_Item_Entregue", "Bugs pós-release / itens entregues", "descritiva"),
        ("base_releases", "Mediana_Historica_Bugs", "Mediana de bugs pós-release das releases ANTERIORES do mesmo projeto", "auxiliar"),
        ("base_releases", "Alto_Risco", "1 se Bugs_Pos_Release > Mediana_Historica_Bugs; -1 quando há menos de três releases anteriores (fora do modelo)", "alvo (classificação)"),
        ("base_releases", "Bugs_Pos_Release_90d / _365d / _Todas_Versoes", "Sensibilidade: janelas de 90 e 365 dias e contagem em todas as versões afetadas", "sensibilidade"),
        ("base_releases", "Qtd_Itens_Estado_Final", "Itens vinculados à release no estado final (a composição usada é a da data de publicação)", "descritiva"),
        # novas colunas (perguntas complementares)
        ("base_bugs", "Responsavel_ID", "Id anonimizado do responsável (a TAWOS remove nomes)", "id"),
        ("base_bugs", "Tem_Responsavel", "1 se o bug tem responsável atribuído", "feature (P5)"),
        ("base_bugs", "Carga_Responsavel", "Itens em aberto atribuídos ao mesmo responsável no dia da abertura", "feature (P5)"),
        ("base_bugs", "Participacao_Responsavel_No_Projeto", "Fatia dos bugs do projeto atribuídos a esse responsável (concentração)", "feature (P5)"),
        ("base_bugs", "Foi_Reaberto", "1 se o bug foi reaberto ao menos uma vez", "alvo (P7)"),
        ("base_bugs", "Qtd_Duplicados", "Quantos outros reports foram marcados como duplicata dele", "alvo/descritiva (P8)"),
        ("base_bugs", "Escalou_Prioridade", "1 se a prioridade subiu depois da abertura", "alvo/descritiva (P8)"),
        ("base_bugs", "Alta_Criticidade", "1 se o bug recebeu duplicata OU teve a prioridade elevada após a abertura (sinal observável de criticidade; não mede impacto real em clientes)", "alvo (P8)"),
        ("base_bugs", "*_Final (versões afetadas)", "Mesmas variáveis da release afetada, no estado final do registro (só para análise de sensibilidade)", "sensibilidade"),
        ("base_sprints", "Sprint_ID / Sprint / Start_Date / Complete_Date", "Sprint fechado e suas datas", "id"),
        ("base_sprints", "Itens_Planejados", "Itens no sprint até 1 dia após o início (compromisso)", "descritiva (P1)"),
        ("base_sprints", "Itens_Adicionados_Durante / Itens_Removidos_Durante", "Mudança de escopo durante o sprint", "descritiva (P1)"),
        ("base_sprints", "Itens_Concluidos / Planejados_Concluidos", "Itens resolvidos até o fechamento do sprint", "descritiva (P1)"),
        ("base_sprints", "Itens_Escorregaram", "Itens que estavam no sprint no fechamento e não foram concluídos (spillover)", "descritiva (P1)"),
        ("base_sprints", "SP_Planejados / SP_Concluidos / SP_Planejados_Concluidos", "Story points planejados e entregues", "descritiva (P1)"),
        ("base_sprints", "Taxa_Cumprimento", "Planejados concluídos / planejados (previsto × realizado)", "alvo/descritiva (P1)"),
        ("base_sprints", "Taxa_Escorregamento", "Escorregaram / itens no fechamento", "descritiva (P1)"),
        ("base_sprints", "Pct_Escopo_Adicionado", "Itens adicionados durante / planejados", "descritiva (P1)"),
        ("base_sprints", "Velocity_SP / Velocity_Media_3_Anteriores", "Story points entregues e média dos 3 sprints anteriores", "descritiva (P1)"),
        ("base_itens_sprint", "(1 linha por item planejado em um sprint)", "Tipo, prioridade, story points, carga do responsável no sprint, vezes que já passou por sprints", "features (P1)"),
        ("base_itens_sprint", "Nao_Concluiu_No_Sprint", "1 se o item planejado não foi concluído dentro do sprint", "alvo (P1)"),
        ("base_bugs_sobrevivencia", "(mesmas variáveis explicativas da base_bugs)", "Todos os bugs abertos até 12 meses antes do fim da coleta, corrigidos ou não", "features (robustez)"),
        ("base_bugs_sobrevivencia", "Situacao_Final", "corrigido / encerrado_sem_correcao / aberto_na_coleta", "descritiva"),
        ("base_bugs_sobrevivencia", "Evento_Corrigido", "1 se o bug foi corrigido; 0 = observação censurada", "alvo (sobrevivência)"),
        ("base_bugs_sobrevivencia", "Tempo_Dias", "Dias até a correção ou até a censura (encerramento sem correção ou fim da coleta)", "alvo (sobrevivência)"),
        ("base_capacidade_mensal", "Project_Key / Mes", "Projeto e mês de resolução", "id"),
        ("base_capacidade_mensal", "Sustentacao_Bugs / Evolucao_Features / Outros", "Itens resolvidos no mês por categoria", "descritiva (P3)"),
        ("base_capacidade_mensal", "Pct_Sustentacao", "% da capacidade do mês gasta em correção de bugs", "descritiva (P3)"),
        ("base_capacidade_mensal", "Bugs_Abertos_No_Mes", "Bugs novos abertos no mês", "descritiva (P3)"),
    ], columns=["Base", "Coluna", "Descrição", "Papel"])

    with pd.ExcelWriter(os.path.join(OUT, "bases_tcc.xlsx")) as xw:
        dicionario.to_excel(xw, sheet_name="Dicionario", index=False)
        rel.to_excel(xw, sheet_name="base_releases", index=False)
        b.to_excel(xw, sheet_name="base_bugs", index=False)
        cap.to_excel(xw, sheet_name="base_capacidade_mensal", index=False)
        spr_sprints.to_excel(xw, sheet_name="base_sprints", index=False)
        eleg.to_excel(xw, sheet_name="elegibilidade", index=False)
        diag.to_excel(xw, sheet_name="diagnosticos", index=False)
        spr_itens.to_excel(xw, sheet_name="base_itens_sprint", index=False)

    print(f"base_bugs:     {len(b):,} bugs")
    print(f"base_releases: {len(rel):,} releases")
    print("Arquivos em", OUT)


if __name__ == "__main__":
    main()
