# Aprendizado de Máquina para Suporte à Decisão na Governança Operacional de Equipes de Tecnologia

Trabalho de Conclusão de Curso – MBA em Data Science e Analytics (USP/Esalq)
Autora: Aline Oliveira da Silveira · Orientador: Ricardo Theodoro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alineosilveira/tcc-governanca-operacional-ml/blob/main/TCC_Analise_e_Modelos.ipynb)

## Objetivo

Avaliar como técnicas de analytics e aprendizado de máquina, aplicadas aos registros de demandas e atividades de desenvolvimento de software, podem apoiar decisões de governança operacional de equipes de tecnologia: risco de releases, tempo de correção de bugs, previsibilidade dos sprints, distribuição do trabalho registrado, retrabalho e sinais de criticidade dos bugs. As hipóteses H1 (composição e tipo da release × bugs pós-release) e H2 (previsão do tempo de correção na abertura) são confirmatórias; as demais análises são exploratórias. Os resultados são associações observacionais, não relações de causa e efeito.

## Como visualizar ou executar

| Opção | Como |
|---|---|
| Ver os resultados no navegador | Abra [`TCC_Analise_e_Modelos.ipynb`](TCC_Analise_e_Modelos.ipynb) aqui no GitHub (já salvo com todas as saídas) |
| Executar online | Botão **Open in Colab** acima → *Ambiente de execução > Executar tudo* (≈ 5 a 8 min) |
| Ver offline | Baixe [`TCC_Analise_e_Modelos.html`](TCC_Analise_e_Modelos.html) e abra no navegador |
| Executar no computador | `pip install -r requirements.txt` e abra o notebook no Jupyter ou VS Code (≈ 8 min) |

## Estrutura

```
TCC_Analise_e_Modelos.ipynb   análises descritivas, testes estatísticos e modelos preditivos
TCC_Analise_e_Modelos.html    versão estática do notebook
bases/                        bases analíticas (CSV, inclusive a base de sobrevivência), elegibilidade.csv (funil de seleção),
                              diagnosticos_reconstrucao.csv (reconstrução temporal) e bases_tcc.xlsx com o dicionário de dados
figuras/                      gráficos gerados pelo notebook
extrair_tawos.py              extrai do dump da TAWOS (TAWOS.sql.zip) os CSVs necessários
preparar_bases.py             gera as bases analíticas a partir dos CSVs extraídos
requirements.txt              bibliotecas utilizadas
```

## Conteúdo do notebook

1. Visão geral dos dados e tabela de elegibilidade
2. Release × bugs pós-release: Kruskal-Wallis, Mann-Whitney, Spearman e regressão Binomial Negativa
3. Tempo de correção de bugs
4. Modelo 1 – faixa de tempo de correção (regressão logística, Random Forest, XGBoost; validação temporal; SHAP)
5. Modelo 2 – release de alto risco
6. Perguntas complementares (exploratórias): previsibilidade dos sprints, distribuição do trabalho registrado (sustentação × evolução), concentração de trabalho, reabertura de bugs e sinais de criticidade
7. Robustez das escolhas: Poisson × Binomial Negativa, análise de sobrevivência (Kaplan-Meier e Cox), ajuste de hiperparâmetros com validação temporal, intervalos de confiança por bootstrap, calibração, disponibilidade temporal das variáveis (estado final × reconstruído × estrito), sensibilidade da janela de bugs pós-release e comparação com regras simples e por projeto
8. Conclusões e 9. Limitações

## Prevenção de vazamento de informação

Cada modelo usa só o que era conhecido no momento da previsão. O histórico de alterações (Change_Log) é usado para reconstruir a prioridade inicial e as versões afetadas na abertura do bug, a composição da release na data de publicação e os itens de cada sprint no início. O alvo de release de alto risco compara cada release com a mediana das releases **anteriores** do mesmo projeto. Componentes, responsáveis e story points permanecem no estado final; a seção 7.6 mostra que uma versão estrita, sem eles, tem desempenho praticamente igual.

## Reproduzir as bases do zero (opcional)

As bases da pasta `bases/` já estão prontas. Para gerá-las novamente:

1. Baixe o `TAWOS.sql.zip` (≈ 640 MB) em https://doi.org/10.5522/04/21308124 e coloque na pasta do repositório.
2. `python extrair_tawos.py` → cria `tawos_csv/` (não versionada)
3. `python preparar_bases.py` → recria `bases/`

## Dados

Base pública **TAWOS**, versão 1.1 (Tawosi, V.; Al-Subaihin, A.; Moussa, R.; Sarro, F. 2022. A versatile dataset of agile open source software projects. In: International Conference on Mining Software Repositories, 2022, Pittsburgh, PA, EUA. Proceedings... p. 707-711), disponibilizada sob licença Apache 2.0. As bases da pasta `bases/` são derivadas dela; os contribuidores aparecem apenas por identificadores anonimizados na origem.

## Licença

Código sob licença MIT (ver [`LICENSE`](LICENSE)). Dados derivados da TAWOS seguem a licença Apache 2.0 da base original.
