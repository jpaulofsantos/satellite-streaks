import ast
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "tcc_pipeline_v3"


def source(path, index):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "".join(nb["cells"][index]["source"])


# Teste intensivo da geometria do gerador sem depender do OpenCV.
ns = {
    "np": np,
    "math": math,
    "CONFIG": {
        "seed_mestra": 20260823,
        "snr_pico": [3.0, 25.0],
        "sigma_ruido_augmentation": [0.0, 2.0],
        "n_segmentos": 96,
    },
    "cv2": None,
}
nb04 = ROOT / "04_geracao_positivos_negativos_deterministica_v3.ipynb"
exec(source(nb04, 7), ns)
rng = np.random.default_rng(20260823)
min_coord, max_coord, min_length = 1.0, 0.0, float("inf")
for _ in range(10000):
    esp = int(rng.integers(1, 4))
    blur = float(rng.uniform(0.8, 2.2))
    meia = esp / 2 + 2.5 * blur
    p0, p1, _ = ns["sortear_segmento_interno"](512, 512, rng, meia)
    obb = ns["cantos_obb"](p0, p1, meia, 512, 512)
    coords = [value for point in obb for value in point]
    min_coord = min(min_coord, min(coords))
    max_coord = max(max_coord, max(coords))
    min_length = min(min_length, math.dist(p0, p1))
assert 0 <= min_coord <= max_coord <= 1
assert min_length >= 0.25 * math.hypot(512, 512)
print(f"Gerador: 10.000 OBBs válidas; faixa [{min_coord:.6f}, {max_coord:.6f}]")


# Executa os testes unitários embutidos no notebook Hough.
nb06 = ROOT / "06_baseline_hough_protocolo_congelado_v3.ipynb"
ns_hough = {"np": np, "math": math, "cv2": None, "Path": Path}
exec(source(nb06, 3), ns_hough)
exec(source(nb06, 5), ns_hough)

# Confirma que o matcher amplo não aceita o antigo deslocamento de 34 px.
limiares = {"iou": 0.3, "angulo": 15.0, "distancia": 14.0, "cobertura": 0.5}
metrica = {"iou": 0.0, "angulo": 0.0, "distancia": 34.0, "cobertura": 0.5}
assert ns_hough["escolher_match"]([metrica], limiares, "geometria") is None
print("Hough: testes geométricos e rejeição do caso deslocado aprovados.")


# Regressão: versões do OpenCV podem devolver HoughLinesP como (N,1,4) ou (N,4).
for linhas in (
    np.array([[[1, 2, 31, 42]], [[5, 6, 45, 56]]]),
    np.array([[1, 2, 31, 42], [5, 6, 45, 56]]),
):
    segmentos = np.asarray(linhas).reshape(-1, 4)
    resultado = [
        ((int(x1), int(y1)), (int(x2), int(y2)))
        for x1, y1, x2, y2 in segmentos
    ]
    assert resultado == [((1, 2), (31, 42)), ((5, 6), (45, 56))]

src02_triagem = source(ROOT / "02_sync_e_auditoria_dominio_ABC_v3.ipynb", 5)
src02_confirmacao = source(ROOT / "02_sync_e_auditoria_dominio_ABC_v3.ipynb", 7)
src05_labels = source(ROOT / "05_validacao_formal_visual_ultralytics_v3.ipynb", 5)
src05_visual = source(ROOT / "05_validacao_formal_visual_ultralytics_v3.ipynb", 9)
src05_aprovacao = source(ROOT / "05_validacao_formal_visual_ultralytics_v3.ipynb", 10)
src05_smoke = source(ROOT / "05_validacao_formal_visual_ultralytics_v3.ipynb", 12)
src05_gate = source(ROOT / "05_validacao_formal_visual_ultralytics_v3.ipynb", 14)
src06_detector = source(nb06, 3)
src06_detalhamento = source(nb06, 15)
src08_avaliacao = source(ROOT / "08_selecao_yolov8n_yolo11n_validacao_v3.ipynb", 9)
nb09 = ROOT / "09_treino_final_multiseed_yolo11n_validacao_v3.ipynb"
src09_config = source(nb09, 5)
src09_treino = source(nb09, 7)
src09_gate = source(nb09, 9)
src09_avaliacao = source(nb09, 11)
src09_publicacao = source(nb09, 13)
nb10 = ROOT / "10_teste_sintetico_yolo_protegido_comparacao_hough_v3.ipynb"
src10_pre = source(nb10, 5)
src10_funcoes = source(nb10, 7)
src10_evento = source(nb10, 9)
src10_pos = source(nb10, 11)
nb12 = ROOT / "12_protocolo_externo_asta_tiles_centerline_sem_inferencia_v3.ipynb"
src12_setup = source(nb12, 2)
src12_gate = source(nb12, 4)
src12_config = source(nb12, 6)
src12_skeleton = source(nb12, 8)
src12_publicacao = source(nb12, 10)
src12_pos = source(nb12, 12)
nb13 = ROOT / "13_evento_externo_asta_yolo_hough_protegido_v3.ipynb"
src13_setup = source(nb13, 1)
src13_pre = source(nb13, 3)
src13_funcoes = source(nb13, 5)
src13_smoke = source(nb13, 7)
src13_evento = source(nb13, 9)
src13_pos = source(nb13, 11)
nb14 = ROOT / "14_analise_pos_teste_erros_figuras_finais_v3.ipynb"
src14_setup = source(nb14, 1)
src14_gate = source(nb14, 3)
src14_quant = source(nb14, 5)
src14_qual = source(nb14, 7)
src14_publicacao = source(nb14, 9)
nb15 = ROOT / "15_auditoria_correcao_hough_asta_v4.ipynb"
src15_setup = source(nb15, 1)
src15_gate = source(nb15, 3)
src15_funcoes = source(nb15, 5)
src15_diag = source(nb15, 7)
src15_contrato = source(nb15, 9)
src15_evento = source(nb15, 11)
src15_verify = source(nb15, 13)
assert "np.asarray(linhas).reshape(-1,4)" in src02_triagem
assert "np.asarray(linhas).reshape(-1,4)" in src06_detector
assert "IDS_COM_TRILHA_OU_CONTAMINACAO = set()" in src02_confirmacao
assert "IDS_HARD_NEGATIVE_REAL_ADICIONAL = set()" in src02_confirmacao
assert "cruz.append(np.cross" not in src05_labels
assert "[('raw',False),('overlay',True)]" in src05_visual
assert "raw_validacao_train.png" in src05_aprovacao
assert "raw_validacao_val.png" in src05_aprovacao
assert "raw_validacao_test.png" in src05_aprovacao
assert "fraction=0.01" not in src05_smoke
assert "balanced_train_val_v2" in src05_smoke
assert "('positivo','tracejado')" in src05_smoke
assert "(RUN/'smoke_ultralytics').resolve()" in src05_smoke
assert "aprovacao_valida('smoke_ultralytics_ok.json',PROTOCOLO_SMOKE_ESPERADO)" in src05_gate
assert "resumo_detalhado_test_v3.json" in src06_detalhamento
assert "bootstrap_f1_por_fundo_v3.csv" in src06_detalhamento
assert "Resumo detalhado e bootstrap salvos no Drive." in src06_detalhamento
assert "import gc,torch" in src08_avaliacao
assert src08_avaliacao.index("import gc,torch") < src08_avaliacao.index("gc.collect();torch.cuda.empty_cache()")
assert "source=str(pasta_imagens_val),stream=True,batch=1" in src08_avaliacao
assert "ids_val_observados" in src08_avaliacao
assert "set(ids_val_observados)==ids_val_esperados" in src08_avaliacao
assert "gc.collect();torch.cuda.empty_cache()" in src08_avaliacao
assert "source=caminhos_val" not in src08_avaliacao
assert "import gc,torch" not in src05_smoke
assert "'seeds':[20260823,20260824,20260825]" in src09_config
assert "'avaliar_tres_checkpoints':True" in src09_config
assert "'selecionar_seed_com_teste':False" in src09_config
assert "'recalibrar_threshold_com_teste':False" in src09_config
assert "'checkpoint_operacional_para_asta':'escolhido somente na validação'" in src09_config
assert "conclusao_selecao['sha256_config_selecao']==HASH_CONFIG_SELECTION" in src09_treino
assert "conclusao_selecao['sha256_best_pt']==contrato_selecao['sha256_best_pt']" in src09_treino
assert "modelo.train(resume=True" in src09_treino
assert "TREINOS_MULTISEED_APROVADOS" in src09_gate
assert "source=str(pasta_val),stream=True,batch=1" in src09_avaliacao
assert "CONFIG_FINAL['threshold_confianca_fixo']" in src09_avaliacao
assert "set(ids)==ids_esperados" in src09_avaliacao
assert "sort_values(['f1','taxa_frames_negativos_com_fp','map50_95_obb','seed']" in src09_avaliacao
assert "len(checkpoints)==3" in src09_publicacao
assert "split='test'" not in src09_treino + src09_avaliacao + src09_publicacao
assert "images/test" not in src09_treino + src09_avaliacao + src09_publicacao
assert "EXPECTED_FINAL_CONTRACT_SHA256='0a8d3537969dcb033dc8cdfa1a9d63dc24de4cc98e212c42fab62621126890a7'" in src10_pre
assert "EXPECTED_FINAL_CONFIG_SHA256='457b70c8e96176827bf4a3f755963e6c6c3e607bec7d8133233518b26be27870'" in src10_pre
assert "EXPECTED_HOUGH_CONFIG_SHA256='14ef0ea56a2e86bf1581225c4b9a0733d482834e5f7d3fab18b3eaf76f03a2e5'" in src10_pre
assert "EXPECTED_HOUGH_RESULT_SHA256='5b476de5a6015b6c38758a43e8c46719050bc036d28362d5f7d43f13e8ffa3b8'" in src10_pre
assert "TESTE_YOLO_PRONTO_PARA_EVENTO_UNICO" in src10_pre
assert "verificar_hash_canonico(CONFIG_FINAL_TEST,'sha256_config_final')" in src10_pre
assert "hough_hash_calculado==hough_hash_observado" in src10_pre
assert "len(df_test)==900" in src10_pre
assert "df_test.sha256_fundo.nunique()==40" in src10_pre
assert "YOLO(" not in src10_pre and ".predict(" not in src10_pre
assert "score_match" in src10_funcoes and "max(aceitos,key=lambda x:(x[0],x[1]))" in src10_funcoes
assert "CONFIG_FINAL_TEST['matching']" in src10_funcoes
assert "EXECUTAR_TESTE_PROTEGIDO=False" in src10_evento
assert "RETOMAR_EVENTO_INCOMPLETO=False" in src10_evento
assert "LOCK_PATH.open('x'" in src10_evento
assert "def atualizar_lock(evento_atual):" in src10_evento
assert "os.replace(temporario,LOCK_PATH)" in src10_evento
assert "Teste YOLO final já concluído. Nova inferência é proibida." in src10_evento
assert "source=str(pasta_test),stream=True,batch=1" in src10_evento
assert "len(ids)==len(ids_test)" in src10_evento and "set(ids)==ids_test" in src10_evento
assert "split='test'" in src10_evento
assert "shutil.copy2(registro_local,registro_path)" in src10_evento
assert src10_evento.index("for arquivo in tqdm(arquivos_seed") < src10_evento.index("shutil.copy2(registro_local,registro_path)")
assert "sorted(r['seed'] for r in registros_teste)==[20260823,20260824,20260825]" in src10_evento
assert "Bootstrap YOLO × Hough por fundo" in src10_evento
assert "delta_f1_yolo_menos_hough" in src10_evento
assert "selecionou_seed_com_teste':False" in src10_evento
assert "recalibrou_threshold_com_teste':False" in src10_evento
assert "TESTE YOLO PROTEGIDO CONCLUÍDO E TRAVADO" in src10_evento
assert "BACKUP TESTE YOLO PROTEGIDO APROVADO" in src10_pos
assert ".predict(" not in src10_pos and "YOLO(" not in src10_pos
src12_todo = src12_setup + src12_gate + src12_config + src12_skeleton + src12_publicacao + src12_pos
assert "EXPECTED_AUDIT_MANIFEST_SHA256='91bfa07ebffceaa46271a254d3862348ce2b242a06ed44fc4c07efeeb58722ca'" in src12_setup
assert "EXPECTED_COMPONENTS_SHA256='cb0089cb07731e5fb233c112f315bae28207b8dad58cec0f733f829565846059'" in src12_setup
assert "EXPECTED_OPERATIONAL_WEIGHT_SHA256='a2b64773f32868ac952d81c33b331f1759b2250a3a04f49d5ab105f7bb79dd0c'" in src12_setup
assert "EXPECTED_HOUGH_CONFIG_SHA256='14ef0ea56a2e86bf1581225c4b9a0733d482834e5f7d3fab18b3eaf76f03a2e5'" in src12_setup
assert "len(manifest)==178" in src12_gate and "len(componentes)==int(manifest.componentes_relevantes.sum())==291" in src12_gate
assert "vazios==['ML1_20200328_002539_red']" in src12_gate
assert "checkpoint_operacional_seed']==20260824" in src12_gate
assert "peso_operacional.is_file() and sha256_arquivo(peso_operacional)==EXPECTED_OPERATIONAL_WEIGHT_SHA256" in src12_gate
assert "'tamanho':512,'overlap':128,'stride':384" in src12_config
assert "'tiles_por_frame':784,'tiles_total':139552" in src12_config
assert "'representacao':'skeleton_centerline'" in src12_config
assert "'primaria':'f1_centerline_macro_frames_positivos'" in src12_config
assert "'tolerancia_centerline_px':14.0" in src12_config
assert "'obb_iou_primario':False" in src12_config
assert "EXECUTAR_VALIDACAO_SKELETON=False" in src12_skeleton
assert "skeleton=skeletonize(binaria)" in src12_skeleton
assert "diagnostico[['id_asta','mascara_pixels_positivos','skeleton_pixels']]" in src12_skeleton
assert "combinada.mascara_pixels_positivos_manifest" in src12_skeleton
assert "combinada.mascara_pixels_positivos_diag" in src12_skeleton
assert "CHECKPOINT_EVERY" in src12_skeleton and "checkpoint_skeleton_asta_v3.csv" in src12_skeleton
assert "VALIDACAO_SKELETON_PRONTA_PARA_REVISAO" in src12_skeleton
assert "BACKUP_RASCUNHO_PROTOCOLO_ASTA_APROVADO_SEM_INFERENCIA" in src12_skeleton
assert "APROVAR_E_PUBLICAR_PROTOCOLO=False" in src12_publicacao
assert "with contrato_path.open('x'" in src12_publicacao
assert "PROTOCOLO_ASTA_V3_PUBLICADO_SEM_INFERENCIA" in src12_publicacao
assert "BACKUP_PROTOCOLO_ASTA_V3_APROVADO_SEM_INFERENCIA" in src12_pos
assert "YOLO(" not in src12_todo and ".predict(" not in src12_todo
assert "HoughLines" not in src12_todo and "HoughLinesP" not in src12_todo
assert "EXPECTED_PROTOCOL_SHA256='dc4c58381b1e3e7d2a24f7fdac8fc59276e53b9d900853c1191a497319e862bc'" in src13_setup
assert "ultralytics.__version__!='8.4.127'" in src13_setup
assert "livre/1024**3<10" in src13_setup
assert "torch.backends.cudnn.deterministic=True" in src13_setup
assert "ASTA_PRONTO_PARA_EVENTO_EXTERNO_UNICO" in src13_pre
assert "assert verificar_hash_canonico(contrato,'sha256_contrato_protocolo_asta')==EXPECTED_PROTOCOL_SHA256" in src13_pre
assert "assert cfg['restricoes']['ajuste_por_resultado_asta'] is False" in src13_pre
assert "'batch_tiles_yolo':BATCH_TILES_YOLO" in src13_pre
assert "'precisao_yolo':'fp32'" in src13_pre
assert "'inferencia':'chamada_do_detector_e_conversao_local_das_saidas_com_sincronizacao_cuda'" in src13_pre
assert "'total_frame':'preprocessamento_mais_inferencia_mais_merging'" in src13_pre
assert "def normalizar_frame_uint8" in src13_funcoes
assert "def segmentos_compativeis" in src13_funcoes
assert "def unir_segmentos" in src13_funcoes
assert "def metricas_centerline" in src13_funcoes
assert "len(unir_segmentos([base,duplicado]))==1" in src13_funcoes
assert "identico['f1_centerline']==1.0" in src13_funcoes
assert "SMOKE_IMPLEMENTACAO_EVENTO_ASTA_APROVADO_SEM_DADOS_ASTA" in src13_smoke
assert "source=entradas_smoke" in src13_smoke and "batch=2" in src13_smoke and "half=False" in src13_smoke
assert "asta_lido':False" in src13_smoke and "trava_evento_criada':False" in src13_smoke
assert "EXECUTAR_EVENTO_ASTA=False" in src13_evento
assert "RETOMAR_EVENTO_INCOMPLETO=False" in src13_evento
assert "Execute e aprove primeiro o smoke operacional sem dados ASTA." in src13_evento
assert "with LOCK_PATH.open('x'" in src13_evento
assert "Evento ASTA já concluído. Nova inferência é proibida." in src13_evento
assert "copiar_com_hash(ASTA_RAW/linha.arquivo_imagem,img_local,linha.sha256_imagem)" in src13_evento
assert "copiar_com_hash(ASTA_RAW/linha.arquivo_mascara,mask_local,linha.sha256_mascara)" in src13_evento
assert "modelo.predict(source=lote" in src13_evento
assert "half=False" in src13_evento
assert "torch.cuda.synchronize();t=time.perf_counter()" in src13_evento
assert "cv2.HoughLinesP" in src13_evento
assert "cv2.setRNGSeed(seed_hough)" in src13_evento
assert "detectar_yolo_tiles(modelo,imagem_norm,preenchimento,linha.id_asta)" in src13_evento
assert "detectar_hough_tiles(imagem_norm,preenchimento,linha.id_asta)" in src13_evento
assert "tempo_pre+t_tiles+t_inf+t_merge" in src13_evento
assert "Bootstrap ASTA por frame" in src13_evento
assert "delta_f1_yolo_menos_hough" in src13_evento
assert "selecionou_modelo_seed_ou_threshold_com_asta':False" in src13_evento
assert "EVENTO EXTERNO ASTA CONCLUÍDO E TRAVADO" in src13_evento
assert "BACKUP_EVENTO_EXTERNO_ASTA_V3_APROVADO" in src13_pos
assert "YOLO(" not in src13_pre and ".predict(" not in src13_pre and "HoughLinesP" not in src13_pre
assert "YOLO(" not in src13_pos and ".predict(" not in src13_pos and "HoughLinesP" not in src13_pos

src14_total = "\n".join([src14_setup, src14_gate, src14_quant, src14_qual, src14_publicacao])
assert "eef0f468754b195984a4888cc8c8439f12b5e9db3af94e875082272503542f7c" in src14_setup
assert "303502ee091dfa26649df5bef4f61f0cf8d644925fd9487f96b2085772b444d8" in src14_setup
assert "7d717aea-eed8-4f4b-ba63-d4b0ca51d29c" in src14_setup
assert "878f63dbedea0f72a4b7ed77c238d49cb2b17a89cfce4f51a27ea3b4d7fa0485" in src14_setup
assert "INSUMOS_ANALISE_POS_TESTE_V4_APROVADOS" in src14_gate
assert "metricas_hough_asta_v4.csv" in src14_gate
assert "checkpoints Hough v4.1" in src14_gate
assert "comparavel_diretamente_entre_dominios':False" in src14_quant
assert "erros_sinteticos_por_amostra_multiseed_v3.csv" in src14_quant
assert "erros_validacao_descritivos_pos_congelamento_v3.csv" in src14_quant
assert "GERAR_FOLHAS_QUALITATIVAS=False" in src14_qual
assert "unico_negativo_oficial" in src14_qual
assert "ANALISE_POS_TESTE_V4_PRONTA_PARA_REVISAO" in src14_qual
assert "painel_4x2_full_crop_predicoes_isoladas_v4" in src14_qual
assert "mapa_densidade_preview" in src14_qual
assert "calcular_recorte_deterministico" in src14_qual
assert "mapa_densidade_recorte" in src14_qual
assert "pixels_h_render==int(rh['metricas']['pred_pixels'])" in src14_qual
assert "painel_y=sobrepor_densidade(base,dens_y" in src14_qual
assert "painel_h=sobrepor_densidade(base,dens_h" in src14_qual
assert "painel_y=sobrepor_densidade(gt,dens_y" not in src14_qual
assert "painel_h=sobrepor_densidade(gt,dens_h" not in src14_qual
assert "REFERÊNCIA OFICIAL (verde) | NÃO É DETECÇÃO" in src14_qual
assert "YOLO (ciano) | SOMENTE PREDIÇÃO, SEM MÁSCARA" in src14_qual
assert "HOUGH v4.1 (vermelho) | SOMENTE PREDIÇÃO, SEM MÁSCARA" in src14_qual
assert " — " not in src14_qual
assert "np.array_equal(_painel_y_zero,_base_teste)" in src14_qual
assert "np.array_equal(_painel_h_zero,_base_teste)" in src14_qual
assert "str(registro['id_asta'])==str(id_asta)" in src14_qual
assert "indice_df.sha256_checkpoint_hough.nunique()==len(indice_df)" in src14_qual
assert "indice_df.sha256_segmentos_hough.nunique()==len(indice_df)" in src14_qual
assert "cv2.line(saida,p1,p2,cor,2,cv2.LINE_AA)" not in src14_qual
assert "APROVAR_PUBLICACAO_ANALISE=False" in src14_publicacao
assert "post_test_analysis_v4" in src14_publicacao
assert "resultados_experimentais_alterados':False" in src14_publicacao
assert "hashes_artefatos={nome:esperado for nome,esperado in hashes.items()" in src14_publicacao
assert "Artefatos científicos divergentes no rascunho" in src14_publicacao
assert "relatorio['sha256_zip_folhas']" in src14_publicacao
assert "relatorio['sha256_indice']" in src14_publicacao
assert "YOLO(" not in src14_total and ".predict(" not in src14_total and "HoughLinesP" not in src14_total
assert "import torch" not in src14_total and "import ultralytics" not in src14_total

src15_total = "\n".join([src15_setup, src15_gate, src15_funcoes, src15_diag, src15_contrato, src15_evento, src15_verify])
assert "EXPECTED_OLD_RESULT='eef0f468754b195984a4888cc8c8439f12b5e9db3af94e875082272503542f7c'" in src15_setup
assert "MIN_LENGTH_V3==128 and MIN_LENGTH_SINTETICO==181" in src15_gate
assert "int(math.hypot(TILE,TILE)*float(d['min_length_frac']))" in src15_gate
assert "nms_vetorizado_exato" in src15_funcoes and "nms_escalar_referencia" in src15_funcoes
assert "assert canon(ref)==canon(vet)" in src15_funcoes
assert "EXECUTAR_DIAGNOSTICO_HOUGH=False" in src15_diag
assert "validar_particao_proprietaria" in src15_diag
assert "particao_proprietaria_por_meio_da_sobreposicao" in src15_gate
assert "DIAGNOSTICO_COSTURA_HOUGH_UM_FRAME_CONCLUIDO" in src15_diag
assert "hough_fiel_com_costura_proprietaria" in src15_diag
assert "relatorio_diagnostico_hough_v4_1.json" in src15_diag
assert "relatorio_diagnostico_hough_v4_1.json" in src15_contrato
assert "segmentos_pos_costura_proprietaria" in src15_evento
assert "unir_segmentos(brutos)" not in src15_evento
assert "APROVAR_CORRECAO_HOUGH=False" in src15_contrato
assert "EXECUTAR_EVENTO_HOUGH_V4=False" in src15_evento
assert "RETOMAR_EVENTO_HOUGH_V4=False" in src15_evento
assert "EVENTS_DIR/f'evento_{event_id}'" in src15_evento
assert "OLD_EVENT/'checkpoints/yolo'" in src15_evento
assert "modelo.predict" not in src15_total and "YOLO(" not in src15_total
assert "import torch" not in src15_total and "import ultralytics" not in src15_total
assert "sobrescrever_evento_v3':True" in src15_gate
assert "BACKUP_EVENTO_HOUGH_ASTA_V4_APROVADO" in src15_verify

# A partição de costura Hough cobre o eixo inteiro exatamente uma vez, inclusive
# no último tile parcialmente preenchido do frame ASTA de 10.560 px.
tree15_diag = ast.parse(src15_diag)
nos_particao = [
    no for no in tree15_diag.body
    if isinstance(no, ast.FunctionDef) and no.name in {"limites_propriedade", "validar_particao_proprietaria"}
]
ns_particao = {"TILE": 512}
exec(compile(ast.Module(body=nos_particao, type_ignores=[]), "<nb15_particao>", "exec"), ns_particao)
starts_particao = [i * 384 for i in range(28)]
intervalos_particao = ns_particao["validar_particao_proprietaria"](starts_particao, 10560)
assert intervalos_particao[0] == (0, 448)
assert intervalos_particao[-1] == (10432, 10560)
assert sum(b - a for a, b in intervalos_particao) == 10560

# Executa localmente a normalização e o merging do Notebook 13 sem depender de OpenCV/Colab.
tree13 = ast.parse(src13_funcoes)
nomes_funcoes13 = {
    "percentil_histograma_uint8", "normalizar_frame_uint8", "angulo_segmento",
    "diferenca_angular", "segmentos_compativeis", "agregar_componente", "unir_segmentos",
}
nos13 = []
for no in tree13.body:
    if isinstance(no, ast.Assign):
        alvos = {alvo.id for alvo in no.targets if isinstance(alvo, ast.Name)}
        if alvos & {"TILE", "STRIDE", "STARTS_X", "STARTS_Y"}:
            nos13.append(no)
    elif isinstance(no, ast.FunctionDef) and no.name in nomes_funcoes13:
        nos13.append(no)
modulo13 = ast.Module(body=nos13, type_ignores=[])
ast.fix_missing_locations(modulo13)
cfg13 = {
    "tiles": {"tamanho": 512, "stride": 384, "grid_x": 28, "grid_y": 28},
    "preprocessamento": {"percentil_baixo": 1.0, "percentil_alto": 99.8},
    "merging_global": {
        "angulo_max_graus": 5.0,
        "distancia_perpendicular_max_px": 14.0,
        "gap_axial_max_px": 128.0,
    },
}
ns13 = {"np": np, "math": math, "cfg": cfg13}
exec(compile(modulo13, "nb13_funcoes_selecionadas", "exec"), ns13)
teste13 = np.arange(256, dtype=np.uint8).reshape(16, 16)
normalizado13, meta13 = ns13["normalizar_frame_uint8"](teste13)
assert normalizado13.dtype == np.uint8 and normalizado13.min() == 0 and normalizado13.max() == 255
base13 = {"x1": 10.0, "y1": 20.0, "x2": 300.0, "y2": 20.0, "peso": 1.0,
          "confianca": 0.9, "tile_x": 0, "tile_y": 0}
duplicado13 = {"x1": 280.0, "y1": 21.0, "x2": 560.0, "y2": 21.0, "peso": 1.0,
               "confianca": 0.8, "tile_x": 1, "tile_y": 0}
paralelo13 = {"x1": 10.0, "y1": 50.0, "x2": 300.0, "y2": 50.0, "peso": 1.0,
              "confianca": 0.7, "tile_x": 0, "tile_y": 0}
assert len(ns13["unir_segmentos"]([base13, duplicado13])) == 1
assert len(ns13["unir_segmentos"]([base13, paralelo13])) == 2

def unir_segmentos_escalar13(segmentos):
    n = len(segmentos)
    pai = list(range(n))
    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i
    def unir(i, j):
        ri, rj = raiz(i), raiz(j)
        if ri != rj:
            pai[rj] = ri
    buckets = {}
    for i, s in enumerate(segmentos):
        buckets.setdefault((int(s["tile_x"]), int(s["tile_y"])), []).append(i)
    for (tx, ty), indices in buckets.items():
        for ny in range(ty - 1, ty + 2):
            for nx in range(tx - 1, tx + 2):
                for i in indices:
                    for j in buckets.get((nx, ny), []):
                        if j > i and ns13["segmentos_compativeis"](segmentos[i], segmentos[j]):
                            unir(i, j)
    grupos = {}
    for i in range(n):
        grupos.setdefault(raiz(i), []).append(i)
    return [ns13["agregar_componente"](segmentos, indices) for indices in grupos.values()]

rng13 = np.random.default_rng(20260830)
segmentos13 = []
for _ in range(240):
    tx, ty = int(rng13.integers(0, 4)), int(rng13.integers(0, 4))
    x, y = float(tx * 384 + rng13.uniform(0, 512)), float(ty * 384 + rng13.uniform(0, 512))
    angulo, comprimento = float(rng13.uniform(0, np.pi)), float(rng13.uniform(128, 500))
    segmentos13.append({"x1": x, "y1": y,
        "x2": x + comprimento * np.cos(angulo), "y2": y + comprimento * np.sin(angulo),
        "peso": comprimento, "confianca": 0.0, "tile_x": tx, "tile_y": ty})

def canonico13(lista):
    return sorted((round(s["x1"], 8), round(s["y1"], 8), round(s["x2"], 8),
                   round(s["y2"], 8), round(s["comprimento"], 8), int(s["fragmentos"]))
                  for s in lista)

assert canonico13(ns13["unir_segmentos"](segmentos13)) == canonico13(unir_segmentos_escalar13(segmentos13))
print("ASTA: normalização uint8 e merging global aprovados em execução local isolada.")

# Executa a célula de contrato do Notebook 09 contra um Drive local simulado.
tmp09 = Path(tempfile.mkdtemp(prefix="tcc_nb09_contract_"))
try:
    backup09 = tmp09 / "drive"
    run09 = tmp09 / "run"
    run_id09 = "synthetic_v3_test_contract"
    manifest_hash09 = "a" * 64
    drive_selection09 = backup09 / "experiments/yolo_family_selection_v3" / run_id09
    drive_selection09.mkdir(parents=True)
    run09.mkdir(parents=True)
    config_selection09 = {
        "protocolo": "yolo_family_selection_v1",
        "run_id": run_id09,
        "sha256_manifest": manifest_hash09,
        "ultralytics": "8.4.127",
        "treino": {"patience": 20, "imgsz": 512, "batch": 16, "workers": 2},
        "predicao_val": {"conf_min": 0.001, "iou_nms": 0.7, "max_det": 100},
        "matching": {"iou": 0.3, "angulo": 15.0, "distancia": 14.0, "cobertura": 0.5, "modo": "combinado"},
    }
    config_text09 = json.dumps(config_selection09, sort_keys=True, separators=(",", ":"))
    config_selection09["sha256_config_selecao"] = hashlib.sha256(config_text09.encode()).hexdigest()
    contract09 = {
        "protocolo": "yolo_family_selection_v1",
        "run_id": run_id09,
        "sha256_manifest": manifest_hash09,
        "vencedor_recomendado": "yolo11n_obb",
        "threshold_confianca": 0.45,
        "sha256_best_pt": "b" * 64,
        "teste_sintetico_lido": False,
        "asta_lido": False,
        "aprovado": True,
        "seeds_finais": [20260823, 20260824, 20260825],
    }
    contract_text09 = json.dumps(contract09, sort_keys=True, separators=(",", ":"))
    contract09["sha256_contrato_selecao"] = hashlib.sha256(contract_text09.encode()).hexdigest()
    (drive_selection09 / "protocolo_selecao_yolo_v3.json").write_text(json.dumps(config_selection09), encoding="utf-8")
    (drive_selection09 / "contrato_selecao_yolo_v3.json").write_text(json.dumps(contract09), encoding="utf-8")
    pointer09 = backup09 / "experiments/yolo_family_selection_v3/LATEST_SELECTION_V3.json"
    pointer09.parent.mkdir(parents=True, exist_ok=True)
    pointer09.write_text(json.dumps({
        "run_id": run_id09,
        "sha256_manifest": manifest_hash09,
        "sha256_contrato_selecao": contract09["sha256_contrato_selecao"],
        "vencedor": "yolo11n_obb",
    }), encoding="utf-8")
    ns09 = {
        "BACKUP_DIR": backup09,
        "RUN": run09,
        "RUN_ID": run_id09,
        "HASH_MANIFEST": manifest_hash09,
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
        "shutil": shutil,
    }
    exec(src09_config, ns09)
    assert ns09["CONFIG_FINAL"]["seeds"] == [20260823, 20260824, 20260825]
    assert ns09["CONFIG_FINAL"]["threshold_confianca_fixo"] == 0.45
    assert ns09["CONFIG_FINAL"]["plano_teste_protegido"]["selecionar_seed_com_teste"] is False
    assert (backup09 / "experiments/yolo_final_multiseed_v3" / run_id09 / "protocolo_treino_final_multiseed_v3.json").is_file()
finally:
    shutil.rmtree(tmp09)
print("HoughLinesP: formatos (N,1,4) e (N,4) aprovados nos notebooks 02 e 06.")
print("Auditoria: conjuntos vazios da seção 3 tipados corretamente.")
print("Labels OBB: convexidade 2D sem API depreciada do NumPy.")
print("Revisão visual: grades raw e overlay exigidas para os três splits.")
print("Smoke Ultralytics: subconjuntos balanceados, output absoluto e protocolo v2 exigidos.")
print("Hough: detalhamento e distribuição bootstrap persistidos no Drive.")
print("YOLO: inferência de validação em batch unitário, IDs exatos e limpeza de VRAM exigidos.")
print("YOLO multiseed: três seeds, retomada, threshold fixo, seleção apenas em validação e teste bloqueado.")
print("YOLO multiseed: contrato final executado com sucesso contra Drive local simulado.")
