import ast
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "tcc_pipeline_v3"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def cell_source(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def parse_code_cells(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(source, filename=f"cell_{index}")


nb07 = load("07_pacote_dataset_para_gpu_v3.ipynb")
nb08 = load("08_selecao_yolov8n_yolo11n_validacao_v3.ipynb")
parse_code_cells(nb07)
parse_code_cells(nb08)

src07 = "\n".join(
    "".join(cell["source"]) for cell in nb07["cells"] if cell["cell_type"] == "code"
)
assert "arquivos)==12004" in src07
assert "EMPACOTAR_DIRETO_DO_DRIVE=False" in src07
assert "RUN_FONTE=DRIVE_RUN" in src07
assert "arquivo.relative_to(RUN_FONTE)" in src07
assert "filter=metadado_tar_deterministico" in src07
assert "PUBLICAR_PACOTE_NO_DRIVE=False" in src07
assert "RUN_FONTE_APROVADO_PARA_EMPACOTAMENTO" in src07
assert "BACKUP_PACOTE_YOLO_APROVADO" in src07
assert "formato':'tar_sem_compressao'" in src07
assert "hashes_de_conteudo_verificados" in src07
assert "hashes_verificados==12000" in src07
assert "linha.sha256_imagem" in src07 and "linha.sha256_label" in src07

tree07 = ast.parse(src07)
normalizer_function = next(
    node for node in ast.walk(tree07)
    if isinstance(node, ast.FunctionDef) and node.name == "metadado_tar_deterministico"
)
normalizer_namespace = {}
exec(
    compile(ast.Module(body=[normalizer_function], type_ignores=[]), "tar_normalizer", "exec"),
    normalizer_namespace,
)
normalize_tar = normalizer_namespace["metadado_tar_deterministico"]
with tempfile.TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    sources = [temporary / "source_a.txt", temporary / "source_b.txt"]
    for source in sources:
        source.write_text("mesmo conteúdo", encoding="utf-8")
    os.utime(sources[0], (1000, 1000))
    os.utime(sources[1], (2000, 2000))
    tar_hashes = []
    for index, source in enumerate(sources):
        archive = temporary / f"archive_{index}.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(source, arcname="run/arquivo.txt", filter=normalize_tar)
        tar_hashes.append(hashlib.sha256(archive.read_bytes()).hexdigest())
    assert len(set(tar_hashes)) == 1, "O TAR depende de metadados da fonte."

gpu_gate = cell_source(nb08, 1)
restore = cell_source(nb08, 3)
protocol = cell_source(nb08, 5)
training = cell_source(nb08, 7)
evaluation = cell_source(nb08, 9)
approval = cell_source(nb08, 11)

assert "torch.cuda.is_available()" in gpu_gate
assert "drive.mount" not in gpu_gate
assert "GPU_APROVADA" in gpu_gate
assert "ultralytics==8.4.127" in restore
assert "Restaurando TAR do dataset" in restore
assert "yolov8n-obb.pt" in protocol
assert "yolo11n-obb.pt" in protocol
assert "'seed_selecao':20260823" in protocol
assert "'seeds_finais':[20260823,20260824,20260825]" in protocol
assert "'teste_sintetico_lido':False" in protocol
assert "'asta_lido':False" in protocol
assert "'mosaic':0.0" in protocol and "'hsv_v':0.0" in protocol
assert "EXECUTAR_SELECAO=False" in training
assert "fraction=" not in training
assert "on_model_save" in training
assert "df_val=df[df.split=='val']" in evaluation
assert "df[df.split=='test']" not in evaluation
assert "escolher_familia" in evaluation
assert "delta_f1_material" in evaluation
assert "comparacao_modelos_val_v3.csv" in evaluation
assert "APROVAR_SELECAO_YOLO=False" in approval
assert "SELEÇÃO YOLO PUBLICADA" in approval

tree = ast.parse(evaluation)
choice_function = next(
    node for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and node.name == "escolher_familia"
)
namespace = {}
exec(compile(ast.Module(body=[choice_function], type_ignores=[]), "choice", "exec"), namespace)
choose = namespace["escolher_familia"]
rule = {
    "delta_f1_material": 0.02,
    "delta_fp_frames_material": 0.01,
    "delta_map_material": 0.01,
    "desempate_final": "yolo11n_obb",
}


def candidates(v8, y11):
    return pd.DataFrame([
        {"modelo": "yolov8n_obb", **v8},
        {"modelo": "yolo11n_obb", **y11},
    ])


winner, _ = choose(candidates(
    {"f1": .60, "taxa_frames_negativos_com_fp": .10, "map50_95_obb": .50},
    {"f1": .57, "taxa_frames_negativos_com_fp": .01, "map50_95_obb": .70},
), rule)
assert winner.modelo == "yolov8n_obb"

winner, _ = choose(candidates(
    {"f1": .60, "taxa_frames_negativos_com_fp": .12, "map50_95_obb": .60},
    {"f1": .59, "taxa_frames_negativos_com_fp": .05, "map50_95_obb": .55},
), rule)
assert winner.modelo == "yolo11n_obb"

winner, _ = choose(candidates(
    {"f1": .60, "taxa_frames_negativos_com_fp": .05, "map50_95_obb": .60},
    {"f1": .59, "taxa_frames_negativos_com_fp": .05, "map50_95_obb": .60},
), rule)
assert winner.modelo == "yolo11n_obb"

print("Notebooks 07–08: sintaxe, gates, isolamento do teste e contrato YOLO aprovados.")
