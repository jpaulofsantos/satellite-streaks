import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "tcc_pipeline_v3"
PATH = ROOT / "11_auditoria_externa_asta_sem_inferencia_v3.ipynb"
notebook = json.loads(PATH.read_text(encoding="utf-8"))

code_sources = []
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    code_sources.append(source)
    cleaned = "\n".join(
        "pass  # magic removida" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )
    ast.parse(cleaned, filename=f"notebook_11_cell_{index}")

source = "\n".join(code_sources)

# Inventário e origem oficial.
assert "records/11642424/files/Processed.zip" in source
assert "EXPECTED_ZIP_MD5='37c6c2096c74ce22778d3baad056d0b9'" in source
assert "len(pares_oficiais)!=178" in source
assert "faltantes_imagem" in source and "faltantes_mascara" in source
assert "extras_imagem" in source and "extras_mascara" in source
assert "fingerprint_nomes_tamanhos" in source

# Proteção operacional e retomada.
assert "EXECUTAR_AUDITORIA_PESADA=False" in source
assert "CHECKPOINT_EVERY=5" in source
assert "checkpoint_manifest_auditoria_asta_v3.csv" in source
assert "staging_um_par" in source
assert "copiar_para_stage_com_hash" in source
assert "raw_somente_leitura':True" in source

# Auditoria das máscaras e material para revisão.
assert "cv2.connectedComponentsWithStats" in source
assert "AREA_MINIMA=15" in source
assert "componentes_relevantes" in source
assert "dtype_mascara" in source and "canais_mascara" in source
assert "folhas_auditoria_asta_v3.zip" in source
assert "AUDITORIA_AUTOMATICA_ASTA_CONCLUIDA_AGUARDA_REVISAO_VISUAL" in source
assert "revisao_visual_aprovada':False" in source
assert "protocolo_tiles_congelado':False" in source

# Este notebook não pode abrir a avaliação externa.
for forbidden in ("YOLO(", ".predict(", ".val(", "HoughLinesP", "EXECUTAR_TESTE", "EXECUTAR_INFERENCIA"):
    assert forbidden not in source, f"Operação proibida encontrada: {forbidden}"
assert "yolo_executado':False" in source
assert "hough_executado':False" in source
assert "Image.MAX_IMAGE_PIXELS = None" not in source

print("Notebook 11: inventário, retomada, auditoria de máscaras e bloqueio de inferência aprovados.")
