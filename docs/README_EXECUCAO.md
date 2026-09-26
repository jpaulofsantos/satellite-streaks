# Pipeline TCC v3 — ordem de execução

Os notebooks desta pasta substituem a cadeia anterior sem sobrescrevê-la.

1. `01_coleta_e_padronizacao_fundos_v3.ipynb`
   - usa o raw existente e cria IDs estáveis por SHA-256;
   - coleta adicional é opcional e vem desativada.
2. `02_sync_e_auditoria_dominio_ABC_v3.ipynb`
   - aplica a política assistida v3.2 sem reutilizar a migração legada incorreta;
   - mostra somente alertas lineares para confirmação dirigida;
   - publicação da curadoria vem bloqueada até o gate local ser aprovado.
3. `03_split_agrupado_dos_fundos_v3.ipynb`
   - usa 68/16/16 e só prossegue com no mínimo 40 fundos A em validação e teste;
   - publicação do split vem bloqueada para conferência do resumo local.
4. `04_geracao_positivos_negativos_deterministica_v3.ipynb`
   - cria um RUN_ID imutável e 6.000 amostras retomáveis por seed individual.
5. `05_validacao_formal_visual_ultralytics_v3.ipynb`
   - exige revisão visual e smoke test Ultralytics antes de aprovar.
6. `06_baseline_hough_protocolo_congelado_v3.ipynb`
   - calibra somente em validação;
   - o teste vem desativado e possui trava persistente.
7. `07_pacote_dataset_para_gpu_v3.ipynb`
   - executa em CPU, preferencialmente no runtime ainda ativo do Notebook 06;
   - se o runtime anterior tiver sido perdido, pode criar o TAR diretamente do RUN no Drive,
     sem restaurar antes os 12.004 arquivos para `/content`;
   - reúne os 12.004 arquivos necessários em um TAR único com SHA-256.
8. `08_selecao_yolov8n_yolo11n_validacao_v3.ipynb`
   - exige GPU antes de montar o Drive;
   - compara YOLOv8n-OBB e YOLO11n-OBB somente em treino/validação;
   - mantém teste sintético e ASTA bloqueados.
9. `09_treino_final_multiseed_yolo11n_validacao_v3.ipynb`
   - valida e publica o protocolo final antes do treino;
   - reutiliza a seed de seleção somente sob igualdade exata de hashes/configuração;
   - treina as duas seeds adicionais com retomada e checkpoints no Drive;
   - escolhe o checkpoint operacional somente na validação e congela os três modelos.
10. `10_teste_sintetico_yolo_protegido_comparacao_hough_v3.ipynb`
   - verifica o contrato final e o Hough congelado antes de abrir o teste;
   - avalia os três checkpoints em um evento único e retomável, sem seleção posterior;
   - reporta média e variabilidade entre seeds, subtipos, negativos, geometria e bootstrap por fundo;
   - mantém ASTA bloqueado.
11. `11_auditoria_externa_asta_sem_inferencia_v3.ipynb`
   - executa em CPU e trata `data/raw/asta_real` como somente leitura;
   - confirma os 178 pares contra o índice oficial, calcula hashes e audita máscaras;
   - processa apenas um par local por vez, com checkpoint a cada cinco pares;
   - produz folhas visuais e mantém YOLO, Hough e o protocolo externo bloqueados.
12. `12_protocolo_externo_asta_tiles_centerline_sem_inferencia_v3.ipynb`
   - executa em CPU e reconcilia os artefatos aprovados do Notebook 11;
   - congela tiles 512, overlap, normalização, merging e métricas de centerline;
   - valida skeletons oficiais com checkpoint e revisão dirigida;
   - publica o contrato externo sem carregar ou executar detectores.
13. `13_evento_externo_asta_yolo_hough_protegido_v3.ipynb`
   - exige GPU desde o início e valida o contrato ASTA antes de carregar modelos;
   - executa YOLO operacional e Hough congelado uma única vez sobre 178 frames em tiles;
   - salva checkpoints por frame/método, permite retomada e trava qualquer reexecução concluída;
   - consolida centerline macro/pooled, tempos, bootstrap pareado e casos para revisão qualitativa.
14. `14_analise_pos_teste_erros_figuras_finais_v3.ipynb`
   - executa em CPU e não carrega detectores;
   - reutiliza o YOLO ASTA v3 travado e o Hough corrigido v4.1, sem executar inferência;
   - consolida resultados sintéticos e ASTA sem comparar diretamente endpoints diferentes;
   - descreve erros por subtipo/SNR e produz figuras quantitativas;
   - materializa casos por regras determinísticas em painéis que isolam imagem, referência, YOLO e Hough;
   - inclui frame inteiro e recorte determinístico e publica somente após revisão humana.
15. `15_auditoria_correcao_hough_asta_v4.ipynb`
   - executa em CPU e preserva integralmente o evento ASTA v3;
   - diagnostica em um frame a malha Hough alinhada ao grid 28×28;
   - repõe a diagonal congelada de `minLineLength` e o NMS exato do Notebook 06, sem busca de parâmetros;
   - audita a costura por partição proprietária: cada pixel pertence a exatamente um tile e o merging global transitivo não é usado;
   - após gate humano, executa somente Hough em evento v4 separado e reutiliza os checkpoints YOLO existentes.

## Regras operacionais

- Execute cada notebook do início ao fim e salve as saídas.
- Antes de publicar a saída final de uma etapa, rode o gate local correspondente; depois
  do backup, confirme contagem e hashes no Drive.
- Operações longas exibem barra de progresso, horário, tempo decorrido e estimativa restante.
- Checkpoints e caches não equivalem a aprovação; somente manifests/gates aprovados liberam
  o notebook seguinte.
- Não reutilize números dos notebooks v0/v2 como resultados finais.
- Não altere o teste depois de observar o resultado.
- Os hard negatives gerados são sintéticos; descreva-os dessa forma no TCC.
- A normalização por percentis é computacional, não fotométrica.
- A seleção da família YOLO não consulta o teste sintético nem ASTA.
- O modelo perdedor da seleção não será executado no teste final do TCC.
- O teste YOLO reportará os três checkpoints e sua média; não escolherá a melhor seed pelo teste.
- O checkpoint operacional para ASTA é escolhido somente na validação no Notebook 09.
- A auditoria ASTA não é o teste externo: ela não carrega modelos nem define o resultado.
- Não executar inferência ASTA antes de revisar as folhas skeleton, publicar e verificar o protocolo externo.
- O evento ASTA usa exatamente o contrato publicado; nenhum resultado externo pode recalibrar o pipeline.
- Atualize o compilado final somente após o evento Hough v4.1 e a análise pós-teste v4 estarem
  aprovados; preserve o Hough ASTA v3 apenas como registro do defeito de equivalência.
