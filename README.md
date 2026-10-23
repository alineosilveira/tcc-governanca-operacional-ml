# Aprendizado de Máquina para Suporte à Decisão na Governança Operacional de Equipes de Tecnologia

Trabalho de Conclusão de Curso – MBA em Data Science e Analytics (USP/Esalq)
Autora: Aline Oliveira da Silveira · Orientador: Ricardo Theodoro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alineosilveira/tcc-governanca-operacional-ml/blob/main/TCC_Analise_e_Modelos.ipynb)

## Objetivo

Avaliar como técnicas de analytics e aprendizado de máquina, aplicadas aos registros de demandas e atividades de desenvolvimento de software, podem apoiar decisões de governança operacional de equipes de tecnologia: risco de releases, tempo de correção de bugs, previsibilidade dos sprints, distribuição da capacidade, retrabalho e impacto em clientes.

## Como visualizar ou executar

| Opção | Como |
|---|---|
| Ver os resultados no navegador | Abra [`TCC_Analise_e_Modelos.ipynb`](TCC_Analise_e_Modelos.ipynb) aqui no GitHub (já salvo com todas as saídas) |
| Executar online | Botão **Open in Colab** acima → *Ambiente de execução > Executar tudo* (≈ 4 a 5 min) |
| Ver offline | Baixe [`TCC_Analise_e_Modelos.html`](TCC_Analise_e_Modelos.html) e abra no navegador |
| Executar no computador | `pip install -r requirements.txt` e abra o notebook no Jupyter ou VS Code |

## Estrutura

```
TCC_Analise_e_Modelos.ipynb   análises descritivas, testes estatísticos e modelos preditivos
TCC_Analise_e_Modelos.html    versão estática do notebook
bases/                        bases analíticas (CSV) e bases_tcc.xlsx com o dicionário de dados
figuras/                      gráficos gerados pelo notebook
extrair_tawos.py              extrai do dump da TAWOS (TAWOS.sql.zip) os CSVs necessários
preparar_bases.py             gera as bases analíticas a partir dos CSVs extraídos
requirements.txt              bibliotecas utilizadas
```

## Conteúdo do notebook

1. Visão geral dos dados
2. Release × bugs pós-release: Kruskal-Wallis, Mann-Whitney, Spearman e regressão Binomial Negativa
3. Tempo de correção de bugs
4. Modelo 1 – faixa de tempo de correção (regressão logística, Random Forest, XGBoost; validação temporal; SHAP)
5. Modelo 2 – release de alto risco
6. Perguntas complementares: previsibilidade dos sprints, capacidade (sustentação × evolução), concentração de trabalho, reabertura de bugs e impacto em clientes
7. Conclusões e 8. Limitações

## Reproduzir as bases do zero (opcional)

As bases da pasta `bases/` já estão prontas. Para gerá-las novamente:

1. Baixe o `TAWOS.sql.zip` (≈ 640 MB) em https://doi.org/10.5522/04/21308124 e coloque na pasta do repositório.
2. `python extrair_tawos.py` → cria `tawos_csv/` (não versionada)
3. `python preparar_bases.py` → recria `bases/`

## Dados

Base pública **TAWOS**, versão 1.1 (Tawosi, V.; Al-Subaihin, A.; Moussa, R.; Sarro, F. 2022. A versatile dataset of agile open source software projects. In: International Conference on Mining Software Repositories, 2022, Pittsburgh, PA, EUA. Proceedings... p. 707-711), disponibilizada sob licença Apache 2.0. As bases da pasta `bases/` são derivadas dela; os contribuidores aparecem apenas por identificadores anonimizados na origem.

## Licença

Código sob licença MIT (ver [`LICENSE`](LICENSE)). Dados derivados da TAWOS seguem a licença Apache 2.0 da base original.
