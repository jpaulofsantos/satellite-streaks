# Inventário de imagens do compilado v7

Documento de referência: `compilado-progresso-tcc-v7-abertura-visual-2026-09-07.docx`.

O compilado mantém as 16 figuras numeradas e adiciona uma imagem de contexto sem numeração no início da introdução. Os caminhos abaixo são relativos à raiz do projeto.

## Imagem de abertura

| Elemento | Arquivo-fonte utilizado | Conteúdo |
|---|---|---|
| Imagem de contexto (sem numeração) | `assets/readme/asta_trilha_contexto_abertura.png` | Recorte sem anotações do frame ASTA `ML1_20171023_002210_red`, contendo uma trilha real. Derivado do painel qualitativo aprovado, sem nova inferência, máscara ou predição. |

| Figura | Arquivo-fonte utilizado | Conteúdo |
|---:|---|---|
| 1 | `work/folhas_auditoria_v3_review/contato_sdss_01.png` | Folha de contato de fundos SDSS classificados como domínio A |
| 2 | `work/folhas_auditoria_v3_review/contato_hubble_01.png` | Folha de contato de imagens Hubble classificadas como domínio B |
| 3 | `work/revisao_visual_v3_complete_submission/overlay_validacao_val.png` | Amostra estratificada da validação com OBBs e negativos |
| 4 | `work/revisao_visual_v3_complete_submission/ampliado_raw_overlay_tracejado.png` | Exemplos raw/overlay do subtipo tracejado |
| 5 | `work/revisao_visual_v3_complete_submission/ampliado_raw_overlay_variavel.png` | Exemplos raw/overlay do subtipo de brilho variável |
| 6 | `work/compilado_v3_20260831_assets/fig_multiseed_f1.png` | F1 do YOLO11n-OBB por seed em validação e teste sintético |
| 7 | `work/compilado_v3_20260831_assets/fig_teste_sintetico_yolo_hough.png` | Comparação YOLO × Hough no teste sintético protegido |
| 8 | `work/asta_visual_review_20260830/folha_01.jpg` | Amostra da auditoria visual do ASTA |
| 9 | `work/compilado_v3_20260831_assets/fig_asta_yolo_resultados.png` | Resultado externo do YOLO no ASTA com métricas e IC95% |
| 10 | `work/compilado_v4_20260903_assets/fig_asta_yolo_hough_v4.png` | Comparação externa YOLO × Hough corrigido v4.1 |
| 11 | `work/review_qual_v4_921b517e/paineis_qualitativos/ML1_20171023_002210_red.jpg` | Sucesso externo do YOLO, F1 0,9772 |
| 12 | `work/review_qual_v4_921b517e/paineis_qualitativos/ML1_20210413_033516_red.jpg` | Caso com múltiplas trilhas, F1 YOLO 0,9842 |
| 13 | `work/review_qual_v4_921b517e/paineis_qualitativos/ML1_20220525_200632_red.jpg` | Caso intermediário com detecção parcial, F1 YOLO 0,6003 |
| 14 | `work/review_qual_v4_921b517e/paineis_qualitativos/ML1_20190822_034309_red.jpg` | Falha completa do YOLO com trilha oficial presente |
| 15 | `work/review_qual_v4_921b517e/paineis_qualitativos/ML1_20200328_002539_red.jpg` | Único negativo oficial do ASTA |
| 16 | `work/compilado_v3_20260831_assets/fig_fluxo_experimental.png` | Fluxo experimental e barreiras contra leakage e ajuste pós-teste |

## Cópias destinadas ao README

Para evitar dependência da pasta de trabalho interna, quatro imagens foram copiadas para `outputs/assets/readme/`:

| Arquivo público | Origem |
|---|---|
| `outputs/assets/readme/dataset_sintetico_validacao.png` | Figura 3 |
| `outputs/assets/readme/resultado_teste_sintetico.png` | Figura 7 |
| `outputs/assets/readme/resultado_asta.png` | Figura 10 |
| `outputs/assets/readme/exemplo_qualitativo_asta.jpg` | Figura 11 |

As cópias mantêm exatamente o mesmo conteúdo e SHA-256 dos arquivos-fonte.
