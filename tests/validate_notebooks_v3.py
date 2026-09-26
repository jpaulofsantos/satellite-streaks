import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tcc_pipeline_v3"

errors = []
reports = []
for path in sorted(OUT.glob("*.ipynb")):
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.name}: JSON inválido: {exc}")
        continue
    if nb.get("nbformat") != 4:
        errors.append(f"{path.name}: nbformat != 4")
    code_cells = 0
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = "".join(cell.get("source", []))
        cleaned = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                indent = line[: len(line) - len(stripped)]
                cleaned.append(indent + "pass  # IPython magic removida na validação estática")
            else:
                cleaned.append(line)
        try:
            ast.parse("\n".join(cleaned), filename=f"{path.name}:cell{index}")
        except SyntaxError as exc:
            errors.append(f"{path.name} célula {index}: {exc}")
    reports.append((path.name, len(nb.get("cells", [])), code_cells, path.stat().st_size))

all_text = "\n".join(p.read_text(encoding="utf-8") for p in OUT.glob("*.ipynb"))
required = [
    "manifest_base_v3.csv",
    "manifest_curado_v3.csv",
    "manifest_split_fundos_v3.csv",
    "manifest_geracao_v3.csv",
    "resumo_validacao_v3.json",
    "config_congelada_v3.json",
    "contrato_modelos_finais_v3.json",
    "TRAVA_EVENTO_TESTE_YOLO_V3.json",
    "resumo_final_yolo_test_v3.json",
    "manifest_auditoria_asta_v3.csv",
    "resumo_auditoria_asta_v3.json",
    "folhas_auditoria_asta_v3.zip",
    "protocolo_asta_candidato_v3.json",
    "folhas_revisao_skeleton_asta_v3.zip",
    "contrato_protocolo_asta_v3.json",
    "TRAVA_EVENTO_EXTERNO_ASTA_V3.json",
    "resumo_final_evento_asta_v3.json",
    "metricas_por_frame_metodo_asta_v3.csv",
    "bootstrap_frames_asta_v3.csv",
]
for item in required:
    if item not in all_text:
        errors.append(f"Referência obrigatória ausente: {item}")

if "-0.5" in all_text or "1.5]" in all_text:
    errors.append("Faixa antiga de coordenadas encontrada nos notebooks v3")

for report in reports:
    print(f"{report[0]}: {report[1]} células, {report[2]} de código, {report[3]} bytes")

if errors:
    print("\nERROS:")
    print("\n".join(errors))
    raise SystemExit(1)
print("\nValidação estrutural e sintática concluída sem erros.")
