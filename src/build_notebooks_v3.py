import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tcc_pipeline_v3"
OUT.mkdir(parents=True, exist_ok=True)


def _src(text):
    return textwrap.dedent(text).strip("\n") + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text).splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text).splitlines(True),
    }


def write_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT / filename
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def build_01():
    cells = [
        md(r'''
        # 01 — Coleta e padronização reprodutível dos fundos (v3)

        Esta versão mantém `data/raw/` imutável e produz fundos processados com identificadores
        estáveis derivados de SHA-256. A padronização usa recorte central quadrado, sem deformar
        a razão de aspecto, e normalização robusta por percentis.

        **Saídas:**
        - `data/processed_v3/backgrounds_base/`
        - `data/processed_v3/manifest_base_v3.csv`
        - metadados de coleta em `data/raw/metadata_v3/`

        A normalização é uma transformação computacional, não uma calibração fotométrica.
        ''') ,
        code(r'''
        %pip install -q kaggle astroquery opencv-python-headless pillow tqdm pandas

        from google.colab import drive, userdata
        from pathlib import Path
        from datetime import date, datetime, timezone, timedelta
        import csv, gc, hashlib, json, math, os, shutil, time
        import cv2
        import numpy as np
        import pandas as pd

        # `force_remount` evita o estado FUSE quebrado do Colab
        # (OSError 107: Transport endpoint is not connected).
        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR = Path('/content/drive/MyDrive/tcc-satellite-streaks')
        RAW = Path('data/raw')
        META = RAW / 'metadata_v3'
        PROC = Path('data/processed_v3')
        BASE = PROC / 'backgrounds_base'
        for pasta in [RAW, META, PROC, BASE]:
            pasta.mkdir(parents=True, exist_ok=True)

        DATA_EXECUCAO = datetime.now(timezone.utc).isoformat()
        SEED_COLETA = 20260823
        TAMANHO = 512
        ''') ,
        md(r'''
        ## 1. Restaurar somente o dado bruto

        O bruto é cumulativo e imutável; por isso a restauração pode ser incremental. Pastas
        processadas não são restauradas aqui, evitando que artefatos antigos reapareçam.
        ''') ,
        code(r'''
        # ASTA é teste externo intocado e NÃO é restaurado nesta etapa de fundos.
        # A lista explícita também impede que outras pastas acidentais entrem no pipeline.
        PASTAS_RAW_FUNDOS = ('hubble', 'jwst', 'sdss', 'apod', 'metadata_v3')

        def copiar_raw_seletivo(origem_raiz, destino_raiz):
            origem_raiz, destino_raiz = Path(origem_raiz), Path(destino_raiz)
            copiadas = []
            for nome in PASTAS_RAW_FUNDOS:
                origem = origem_raiz / nome
                if not origem.is_dir():
                    continue
                try:
                    shutil.copytree(origem, destino_raiz / nome, dirs_exist_ok=True)
                    copiadas.append(nome)
                except OSError as exc:
                    raise RuntimeError(
                        'O Google Drive desconectou durante a cópia. '
                        'Reconecte o Drive, execute novamente a célula de configuração '
                        'e depois repita esta célula; a cópia incremental pode ser retomada.'
                    ) from exc
            return copiadas

        origem_raw = BACKUP_DIR / 'data/raw'
        if origem_raw.is_dir():
            copiadas = copiar_raw_seletivo(origem_raw, RAW)
            print('Pastas de fundos restauradas:', copiadas or 'nenhuma disponível')
            print('ASTA preservado no Drive e excluído desta etapa.')
        else:
            print('Primeira execução: ainda não existe backup bruto.')
        ''') ,
        md(r'''
        ## 2. Coleta opcional Kaggle

        Estas coleções são preservadas como bruto, mas serão categoria B por padrão na auditoria,
        pois contêm majoritariamente imagens icônicas processadas.
        ''') ,
        code(r'''
        BAIXAR_KAGGLE = False  # altere para True somente se os arquivos ainda não estiverem no raw
        if BAIXAR_KAGGLE:
            os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
            (RAW / 'hubble').mkdir(parents=True, exist_ok=True)
            (RAW / 'jwst').mkdir(parents=True, exist_ok=True)
            !kaggle datasets download -d redwankarimsony/top-100-hubble-telescope-images -p data/raw/hubble --unzip
            !kaggle datasets download -d goelyash/james-webb-telescope-images-original-size -p data/raw/jwst --unzip
        ''') ,
        md(r'''
        ## 3. Coleta SDSS determinística e com metadados

        Os pontos candidatos são reproduzíveis. Cada FITS é identificado por coordenadas, e os
        metadados preservam RA/Dec e a data da consulta.
        ''') ,
        code(r'''
        from astropy import coordinates as coords
        from astropy import units as u
        from astroquery.sdss import SDSS

        COLETAR_SDSS = False
        META_SDSS = META / 'sdss_metadata_v3.csv'
        N_CAMPOS_SDSS = 250

        if COLETAR_SDSS:
            pasta = RAW / 'sdss'
            pasta.mkdir(parents=True, exist_ok=True)
            existentes = pd.read_csv(META_SDSS) if META_SDSS.exists() else pd.DataFrame()
            ids_existentes = set(existentes.get('id_consulta', pd.Series(dtype=str)).astype(str))
            linhas = existentes.to_dict('records') if len(existentes) else []
            rng = np.random.default_rng(SEED_COLETA)
            candidatos = [(rng.uniform(100, 260), rng.uniform(-5, 60)) for _ in range(N_CAMPOS_SDSS * 12)]

            for indice, (ra, dec) in enumerate(candidatos):
                if len(linhas) >= N_CAMPOS_SDSS:
                    break
                id_consulta = f'{indice:05d}_{ra:.6f}_{dec:.6f}'
                if id_consulta in ids_existentes:
                    continue
                pos = coords.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                try:
                    encontrados = SDSS.query_region(pos, radius=2*u.arcmin, spectro=False)
                    if encontrados is None or len(encontrados) == 0:
                        continue
                    imagens = SDSS.get_images(matches=encontrados[:1], band='r')
                    if not imagens:
                        continue
                    nome = f'sdss_{id_consulta}.fits'
                    imagens[0].writeto(pasta / nome, overwrite=False)
                    linhas.append({
                        'id_consulta': id_consulta, 'arquivo_original': nome,
                        'ra_deg': ra, 'dec_deg': dec, 'banda': 'r',
                        'url_origem': 'https://www.sdss.org/',
                        'data_coleta_utc': DATA_EXECUCAO,
                    })
                    pd.DataFrame(linhas).to_csv(META_SDSS, index=False)
                except Exception as exc:
                    print(f'Consulta {id_consulta} ignorada: {type(exc).__name__}')
            print(f'SDSS registrados: {len(linhas)}')
        ''') ,
        md(r'''
        ## 4. Coleta APOD com proveniência por imagem
        ''') ,
        code(r'''
        import requests

        COLETAR_APOD = False
        META_APOD = META / 'apod_metadata_v3.csv'
        N_DIAS_APOD = 100

        if COLETAR_APOD:
            api_key = userdata.get('NASA_API_KEY')
            pasta = RAW / 'apod'
            pasta.mkdir(parents=True, exist_ok=True)
            existentes = pd.read_csv(META_APOD) if META_APOD.exists() else pd.DataFrame()
            datas_existentes = set(existentes.get('data_apod', pd.Series(dtype=str)).astype(str))
            linhas = existentes.to_dict('records') if len(existentes) else []
            hoje = date.today()
            for deslocamento in range(1, N_DIAS_APOD + 1):
                dia = (hoje - timedelta(days=deslocamento)).isoformat()
                if dia in datas_existentes:
                    continue
                resposta = requests.get('https://api.nasa.gov/planetary/apod',
                                        params={'api_key': api_key, 'date': dia}, timeout=30)
                resposta.raise_for_status()
                meta = resposta.json()
                if meta.get('media_type') != 'image':
                    continue
                url = meta.get('hdurl') or meta.get('url')
                imagem = requests.get(url, timeout=60)
                imagem.raise_for_status()
                extensao = Path(url.split('?')[0]).suffix.lower()
                if extensao not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
                    extensao = '.jpg'
                nome = f'apod_{dia}{extensao}'
                (pasta / nome).write_bytes(imagem.content)
                linhas.append({
                    'data_apod': dia, 'arquivo_original': nome, 'titulo': meta.get('title'),
                    'copyright': meta.get('copyright'), 'url_origem': url,
                    'url_api': f'https://api.nasa.gov/planetary/apod?date={dia}',
                    'data_coleta_utc': DATA_EXECUCAO,
                })
                pd.DataFrame(linhas).to_csv(META_APOD, index=False)
                time.sleep(0.5)
            print(f'APOD registrados: {len(linhas)}')
        ''') ,
        md(r'''
        ## 5. Padronização determinística e IDs estáveis

        A classificação humana posterior será vinculada a `fonte + arquivo_original + sha256_raw`,
        nunca a um contador sequencial. A leitura é limitada antes da conversão para `float32`,
        e cada imagem concluída recebe um checkpoint versionado no Drive para permitir retomada.
        ''') ,
        code(r'''
        from astropy.io import fits
        from PIL import Image

        EXT_IMAGEM = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        FONTES = ['hubble', 'jwst', 'sdss', 'apod']
        URL_FONTE = {
            'hubble': 'https://www.kaggle.com/datasets/redwankarimsony/top-100-hubble-telescope-images',
            'jwst': 'https://www.kaggle.com/datasets/goelyash/james-webb-telescope-images-original-size',
            'sdss': 'https://www.sdss.org/',
            'apod': 'https://apod.nasa.gov/apod/',
        }
        LICENCA_FONTE = {
            'hubble': 'Kaggle: copyright-authors; verificar crédito original NASA/ESA/Hubble',
            'jwst': 'Kaggle: CC0-1.0; preservar proveniência original',
            'sdss': 'SDSS Collaboration; uso público com atribuição',
            'apod': 'Crédito/licença por imagem; verificar autor individual',
        }

        def sha256_arquivo(caminho, bloco=1024*1024):
            h = hashlib.sha256()
            with open(caminho, 'rb') as arquivo:
                for parte in iter(lambda: arquivo.read(bloco), b''):
                    h.update(parte)
            return h.hexdigest()

        MAX_LADO_INTERMEDIARIO = 2048
        MAX_PIXELS_RASTER_NAO_JPEG = 30_000_000

        def ler_e_reduzir(caminho, fonte, tamanho=512):
            """Retorna uma imagem 2D pequena sem materializar o arquivo astronômico inteiro.

            FITS usa memmap, recorte central e subamostragem antes de virar float32. JPEG usa
            o decoder reduzido do Pillow. Raster enorme sem decoder reduzido é recusado de
            forma controlada, em vez de derrubar a sessão por falta de RAM.
            """
            if fonte == 'sdss' or caminho.suffix.lower() == '.fits':
                with fits.open(caminho, memmap=True, do_not_scale_image_data=True) as hdul:
                    dados = hdul[0].data
                    if dados is None:
                        raise ValueError('FITS sem dados no HDU primário')
                    while dados.ndim > 2:
                        dados = dados[0]
                    if dados.ndim != 2:
                        raise ValueError(f'FITS com dimensionalidade não suportada: {dados.ndim}')
                    altura, largura = dados.shape
                    lado = min(altura, largura)
                    y0, x0 = (altura-lado)//2, (largura-lado)//2
                    passo = max(1, math.ceil(lado / MAX_LADO_INTERMEDIARIO))
                    recorte = dados[y0:y0+lado:passo, x0:x0+lado:passo]
                    reduzida = cv2.resize(
                        np.asarray(recorte, dtype=np.float32),
                        (tamanho, tamanho), interpolation=cv2.INTER_AREA
                    )
                return reduzida

            with Image.open(caminho) as pil:
                formato = (pil.format or '').upper()
                if formato in {'JPEG', 'JPG', 'MPO'}:
                    pil.draft('L', (MAX_LADO_INTERMEDIARIO, MAX_LADO_INTERMEDIARIO))
                largura, altura = pil.size
                if (largura * altura > MAX_PIXELS_RASTER_NAO_JPEG
                        and formato not in {'JPEG', 'JPG', 'MPO'}):
                    raise MemoryError(
                        f'raster {largura}x{altura} ignorado para proteger a RAM; '
                        'converter previamente para JPEG/PNG reduzido se for indispensável'
                    )
                lado = min(largura, altura)
                x0, y0 = (largura-lado)//2, (altura-lado)//2
                recorte = pil.crop((x0, y0, x0+lado, y0+lado)).convert('L')
                reduzida = recorte.resize((tamanho, tamanho), Image.Resampling.LANCZOS)
                return np.asarray(reduzida, dtype=np.float32)

        def normalizar_robusto(imagem, p_baixo=1.0, p_alto=99.8):
            finitos = imagem[np.isfinite(imagem)]
            if finitos.size == 0:
                raise ValueError('imagem sem pixels finitos')
            baixo, alto = np.percentile(finitos, [p_baixo, p_alto])
            if alto <= baixo:
                raise ValueError('imagem sem faixa dinâmica utilizável')
            return np.clip((imagem - baixo) / (alto - baixo), 0, 1) * 255

        tarefas = []
        for fonte in FONTES:
            pasta = RAW / fonte
            if not pasta.is_dir():
                continue
            for caminho in sorted(p for p in pasta.rglob('*') if p.is_file()):
                if fonte == 'sdss' and caminho.suffix.lower() != '.fits':
                    continue
                if fonte != 'sdss' and caminho.suffix.lower() not in EXT_IMAGEM:
                    continue
                tarefas.append((fonte, caminho))

        linhas = []
        falhas = []
        hashes_vistos = {}
        nomes_esperados = set()
        # Cache versionado no Drive: uma queda posterior não obriga a reprocessar arquivos já concluídos.
        CHECKPOINT_BASE = (BACKUP_DIR / 'data/processed_v3/checkpoints'
                           / 'preprocess_safe_v3_1' / 'backgrounds_base')
        CHECKPOINT_BASE.mkdir(parents=True, exist_ok=True)
        for indice, (fonte, caminho) in enumerate(tarefas, start=1):
            pequena = None
            processada = None
            try:
                hash_raw = sha256_arquivo(caminho)
                if hash_raw in hashes_vistos:
                    raise ValueError(
                        f'conteúdo duplicado de {hashes_vistos[hash_raw]}; '
                        'segunda cópia excluída do manifest'
                    )
                id_fundo = f'{fonte}_{hash_raw[:16]}'
                nome_final = f'{id_fundo}.png'
                destino = BASE / nome_final
                checkpoint = CHECKPOINT_BASE / nome_final
                recuperado = False
                if checkpoint.is_file():
                    try:
                        shutil.copy2(checkpoint, destino)
                        teste_cache = cv2.imread(str(destino), cv2.IMREAD_GRAYSCALE)
                        recuperado = teste_cache is not None and teste_cache.shape == (TAMANHO, TAMANHO)
                        teste_cache = None
                    except OSError as exc:
                        print(f'Aviso: checkpoint indisponível para {nome_final}: {exc}')
                        recuperado = False

                if not recuperado:
                    pequena = ler_e_reduzir(caminho, fonte, TAMANHO)
                    processada = normalizar_robusto(pequena).astype(np.uint8)
                    if not cv2.imwrite(str(destino), processada):
                        raise IOError('cv2.imwrite retornou False')
                    try:
                        shutil.copy2(destino, checkpoint)
                    except OSError as exc:
                        # O resultado local continua válido; apenas a retomada fica indisponível.
                        print(f'Aviso: não foi possível salvar checkpoint de {nome_final}: {exc}')
                hash_processado = sha256_arquivo(destino)
                nomes_esperados.add(nome_final)
                linhas.append({
                    'id_fundo': id_fundo, 'fonte': fonte,
                    'arquivo_original': caminho.name,
                    'caminho_raw_relativo': caminho.relative_to(RAW).as_posix(),
                    'arquivo_final': nome_final, 'sha256_raw': hash_raw,
                    'sha256_processado': hash_processado,
                    'url_fonte': URL_FONTE[fonte], 'licenca_fonte': LICENCA_FONTE[fonte],
                    'normalizacao': 'percentis_1_99.8_apos_reducao_segura',
                    'transformacao_geometrica': 'center_crop_square_resize_512',
                })
                hashes_vistos[hash_raw] = caminho.relative_to(RAW).as_posix()
            except Exception as exc:
                falhas.append({
                    'fonte': fonte,
                    'arquivo_original': caminho.name,
                    'caminho_raw_relativo': caminho.relative_to(RAW).as_posix(),
                    'tipo_erro': type(exc).__name__,
                    'motivo': str(exc)[:500],
                })
                print(f'IGNORADO {fonte}/{caminho.name}: {type(exc).__name__}: {exc}')
            finally:
                # Impede acúmulo de buffers de decodificação entre arquivos grandes.
                pequena = None
                processada = None
                if indice % 10 == 0:
                    gc.collect()
                    print(f'Processados {indice}/{len(tarefas)}; válidos={len(linhas)}; ignorados={len(falhas)}')

        # Remove somente sobras da pasta processada v3; o bruto nunca é tocado.
        for caminho in BASE.glob('*'):
            if caminho.is_file() and caminho.name not in nomes_esperados:
                caminho.unlink()

        if not linhas:
            raise RuntimeError('Nenhum fundo pôde ser processado; consulte manifest_falhas_v3.csv.')
        df_base = pd.DataFrame(linhas).sort_values(['fonte', 'id_fundo']).reset_index(drop=True)
        assert df_base['id_fundo'].is_unique
        assert df_base['sha256_raw'].is_unique, 'Duplicata de conteúdo bruto encontrada; revisar antes de seguir.'
        assert set(p.name for p in BASE.glob('*.png')) == set(df_base['arquivo_final'])
        df_base.to_csv(PROC / 'manifest_base_v3.csv', index=False)
        pd.DataFrame(falhas, columns=[
            'fonte', 'arquivo_original', 'caminho_raw_relativo', 'tipo_erro', 'motivo'
        ]).to_csv(PROC / 'manifest_falhas_v3.csv', index=False)
        print(pd.crosstab(index=df_base['fonte'], columns='quantidade'))
        print(f'Total processado: {len(df_base)}')
        print(f'Total ignorado com justificativa: {len(falhas)}')
        ''') ,
        md(r'''
        ## 6. Backup

        `backgrounds_base` é reconstruível. O destino é espelhado por nome para não preservar
        arquivos órfãos. O raw continua incremental e imutável.
        ''') ,
        code(r'''
        def espelhar_pasta_plana(origem, destino):
            origem, destino = Path(origem), Path(destino)
            destino.mkdir(parents=True, exist_ok=True)
            locais = {p.name: p for p in origem.iterdir() if p.is_file()}
            remotos = {p.name: p for p in destino.iterdir() if p.is_file()}
            if not locais:
                raise RuntimeError('Espelhamento bloqueado: origem vazia.')
            for nome, caminho in locais.items():
                if nome not in remotos or sha256_arquivo(caminho) != sha256_arquivo(remotos[nome]):
                    shutil.copy2(caminho, destino / nome)
            for nome, caminho in remotos.items():
                if nome not in locais:
                    caminho.unlink()
            assert set(locais) == {p.name for p in destino.iterdir() if p.is_file()}

        # Backup incremental somente das fontes de fundos. Nunca toca em data/raw/asta_real.
        copiadas = copiar_raw_seletivo(RAW, BACKUP_DIR / 'data/raw')
        print('Pastas brutas atualizadas no Drive:', copiadas)
        destino_proc = BACKUP_DIR / 'data/processed_v3'
        espelhar_pasta_plana(BASE, destino_proc / 'backgrounds_base')
        destino_proc.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROC / 'manifest_base_v3.csv', destino_proc / 'manifest_base_v3.csv')
        shutil.copy2(PROC / 'manifest_falhas_v3.csv', destino_proc / 'manifest_falhas_v3.csv')
        print('Backup v3 concluído e verificado.')
        ''') ,
    ]
    return write_notebook('01_coleta_e_padronizacao_fundos_v3.ipynb', cells)


def build_02():
    cells = [
        md(r'''
        # 02 — Sincronização segura e auditoria de domínio ABC (v3)

        Nenhuma imagem recebe categoria A por omissão. A auditoria é vinculada ao hash estável
        e exige confirmação explícita de que o fundo não contém trilha de satélite preexistente.

        **Categorias:**
        - A: representativo do domínio de levantamento/ASTA;
        - B: astronômico fora do domínio, permitido apenas como augmentation de treino;
        - C: inadequado ou contaminado;
        - REVISAR: bloqueia a finalização.
        ''') ,
        code(r'''
        %pip install -q opencv-python-headless pandas matplotlib tqdm

        from google.colab import drive
        from pathlib import Path
        import hashlib, math, os, shutil, time
        import cv2
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        def copiar_arvore_com_progresso(origem, destino, descricao):
            origem, destino = Path(origem), Path(destino)
            arquivos = sorted(p for p in origem.rglob('*') if p.is_file())
            log_etapa(f'{descricao}: início ({len(arquivos)} arquivos)')
            inicio = time.perf_counter()
            for arquivo in tqdm(arquivos, desc=descricao, unit='arq'):
                alvo = destino / arquivo.relative_to(origem)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arquivo, alvo)
            log_etapa(f'{descricao}: concluído em {time.perf_counter()-inicio:.1f}s')

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR = Path('/content/drive/MyDrive/tcc-satellite-streaks')
        PROC = Path('data/processed_v3')
        BASE = PROC / 'backgrounds_base'
        AUDIT = PROC / 'auditoria_v3'
        CURATED = PROC / 'backgrounds_curated'
        for pasta in [BASE, AUDIT, CURATED]:
            pasta.mkdir(parents=True, exist_ok=True)

        origem = BACKUP_DIR / 'data/processed_v3'
        log_etapa('Restaurando manifest do Notebook 01')
        shutil.copy2(origem / 'manifest_base_v3.csv', PROC / 'manifest_base_v3.csv')
        copiar_arvore_com_progresso(origem / 'backgrounds_base', BASE, 'Restaurando fundos base')
        # Em uma nova sessão Colab, preserva decisões humanas já salvas no Drive.
        if (origem / 'auditoria_v3').is_dir():
            copiar_arvore_com_progresso(origem / 'auditoria_v3', AUDIT, 'Restaurando auditoria anterior')
        df_base = pd.read_csv(PROC / 'manifest_base_v3.csv')
        print(f'Fundos base: {len(df_base)}')
        ''') ,
        md(r'''
        ## 1. Verificação de integridade e migração conservadora

        Decisões B/C legadas podem ser migradas por `fonte + arquivo_original`. Decisões A antigas
        voltam para `REVISAR`, pois a auditoria anterior foi permissiva demais.
        ''') ,
        code(r'''
        def sha256_arquivo(caminho, bloco=1024*1024):
            h = hashlib.sha256()
            with open(caminho, 'rb') as arquivo:
                for parte in iter(lambda: arquivo.read(bloco), b''):
                    h.update(parte)
            return h.hexdigest()

        faltantes, corrompidos = [], []
        for linha in tqdm(df_base.itertuples(), total=len(df_base), desc='Validando hashes', unit='arq'):
            caminho = BASE / linha.arquivo_final
            if not caminho.exists():
                faltantes.append(linha.arquivo_final)
            elif sha256_arquivo(caminho) != linha.sha256_processado:
                corrompidos.append(linha.arquivo_final)
        assert not faltantes and not corrompidos, f'Integridade falhou: faltantes={faltantes[:3]}, corrompidos={corrompidos[:3]}'

        decisoes = df_base[['id_fundo','fonte','arquivo_original','caminho_raw_relativo','arquivo_final','sha256_raw','sha256_processado',
                            'url_fonte','licenca_fonte']].copy()
        decisoes['categoria'] = 'REVISAR'
        decisoes['motivo'] = ''
        decisoes['sem_trilha_confirmado'] = False
        decisoes.loc[decisoes['fonte'].isin(['hubble','jwst']), ['categoria','motivo']] = [
            'B', 'coleção de imagens icônicas/processadas; apenas augmentation de treino'
        ]

        legado = BACKUP_DIR / 'data/processed/manifest_curado.csv'
        if legado.exists():
            df_legado = pd.read_csv(legado)
            chaves = ['fonte','arquivo_original']
            cols = chaves + ['categoria','motivo','arquivo_final']
            migrado = decisoes.merge(df_legado[cols], on=chaves, how='left', suffixes=('','_legado'))
            preservar = migrado['categoria_legado'].isin(['B','C'])
            decisoes.loc[preservar, 'categoria'] = migrado.loc[preservar, 'categoria_legado'].values
            decisoes.loc[preservar, 'motivo'] = migrado.loc[preservar, 'motivo_legado'].fillna('decisão legada B/C').values

            # Correções descobertas após a primeira aprovação, mapeadas pelo ID legado para o original estável.
            forcar_c = {'apod_0215.png','apod_0222.png','apod_0227.png'}
            forcar_b = {'apod_0296.png','apod_0261.png','apod_0229.png','apod_0276.png',
                        'apod_0266.png','apod_0216.png'}
            mapa = df_legado.set_index('arquivo_final')
            for ids, categoria, motivo in [
                (forcar_c, 'C', 'reclassificação externa: paisagem/texto/trilha preexistente ou composição'),
                (forcar_b, 'B', 'reclassificação externa: astrofotografia processada, fora do domínio ASTA'),
            ]:
                chaves_originais = {(mapa.loc[i,'fonte'], mapa.loc[i,'arquivo_original']) for i in ids if i in mapa.index}
                mascara = decisoes.apply(lambda r: (r['fonte'],r['arquivo_original']) in chaves_originais, axis=1)
                decisoes.loc[mascara, ['categoria','motivo']] = [categoria, motivo]

        caminho_decisoes = AUDIT / 'auditoria_decisoes_v3.csv'
        if caminho_decisoes.exists():
            anterior = pd.read_csv(caminho_decisoes)
            manter = anterior[['id_fundo','categoria','motivo','sem_trilha_confirmado']]
            decisoes = decisoes.drop(columns=['categoria','motivo','sem_trilha_confirmado']).merge(manter, on='id_fundo', how='left')
            decisoes['categoria'] = decisoes['categoria'].fillna('REVISAR')
            decisoes['motivo'] = decisoes['motivo'].fillna('')
            decisoes['sem_trilha_confirmado'] = decisoes['sem_trilha_confirmado'].fillna(False).astype(str).str.lower().isin(['true','1','sim','yes'])
        decisoes.to_csv(caminho_decisoes, index=False)
        print(decisoes['categoria'].value_counts(dropna=False))
        ''') ,
        md(r'''
        ## 2. Diagnóstico e folhas de contato legíveis

        As métricas apenas ordenam a revisão. Cada folha contém no máximo 40 imagens para evitar
        aprovações baseadas em miniaturas pequenas demais.
        ''') ,
        code(r'''
        def diagnostico(caminho):
            img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
            mediana = float(np.median(img))
            mad = float(np.median(np.abs(img.astype(float) - mediana)))
            return pd.Series({'media': float(img.mean()), 'desvio': float(img.std()),
                              'mediana': mediana, 'mad_robusto': 1.4826*mad,
                              'fracao_saturada': float((img >= 250).mean())})

        tqdm.pandas(desc='Calculando diagnósticos')
        diag = decisoes['arquivo_final'].progress_apply(lambda n: diagnostico(BASE/n))
        decisoes = pd.concat([decisoes.reset_index(drop=True), diag.reset_index(drop=True)], axis=1)
        decisoes.to_csv(AUDIT / 'auditoria_decisoes_v3.csv', index=False)

        por_pagina = 40
        grupos_fonte = list(decisoes.sort_values(['fonte','desvio','id_fundo']).groupby('fonte'))
        for fonte, grupo in tqdm(grupos_fonte, desc='Gerando folhas por fonte', unit='fonte'):
            registros = list(grupo.to_dict('records'))
            for pagina, inicio in enumerate(range(0, len(registros), por_pagina), start=1):
                lote = registros[inicio:inicio+por_pagina]
                fig, axes = plt.subplots(5, 8, figsize=(20, 14))
                axes = axes.flatten()
                for ax, linha in zip(axes, lote):
                    img = cv2.imread(str(BASE/linha['arquivo_final']), cv2.IMREAD_GRAYSCALE)
                    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                    ax.set_title(f"{linha['id_fundo']}\n{linha['categoria']} σ={linha['desvio']:.1f}", fontsize=6)
                    ax.axis('off')
                for ax in axes[len(lote):]: ax.axis('off')
                plt.tight_layout()
                destino = AUDIT / f'contato_{fonte}_{pagina:02d}.png'
                plt.savefig(destino, dpi=160, bbox_inches='tight')
                plt.show()
                plt.close(fig)
        print('Folhas geradas em', AUDIT)
        ''') ,
        md(r'''
        ## 3. Pausa humana obrigatória

        Abra `auditoria_decisoes_v3.csv` no Drive ou Colab, preencha todas as linhas `REVISAR` e
        marque `sem_trilha_confirmado=True` para todo fundo A ou B aceito. Critérios A incluem:

        - campo amplo/levantamento, sem processamento estético dominante;
        - ausência de texto, paisagem, planeta isolado ou composição;
        - ausência de trilha de satélite/meteoro preexistente;
        - conteúdo e faixa dinâmica suficientes para representar o domínio.

        Salve o CSV e execute as células seguintes. Não transforme automaticamente `REVISAR` em A.
        ''') ,
        code(r'''
        # Sincroniza o material de auditoria para edição/revisão humana.
        destino_audit = BACKUP_DIR / 'data/processed_v3/auditoria_v3'
        copiar_arvore_com_progresso(AUDIT, destino_audit, 'Enviando material de auditoria')
        print('Edite no Drive:', destino_audit / 'auditoria_decisoes_v3.csv')
        ''') ,
        md(r'''
        ## 4. Finalizar manifesto curado e espelhar a pasta aceita

        Execute somente depois de salvar o CSV revisado no Drive.
        ''') ,
        code(r'''
        caminho_drive_decisoes = BACKUP_DIR / 'data/processed_v3/auditoria_v3/auditoria_decisoes_v3.csv'
        df_audit = pd.read_csv(caminho_drive_decisoes)
        df_audit['categoria'] = df_audit['categoria'].astype(str).str.upper().str.strip()
        df_audit['sem_trilha_confirmado'] = df_audit['sem_trilha_confirmado'].astype(str).str.lower().isin(['true','1','sim','yes'])

        assert set(df_audit['categoria']).issubset({'A','B','C','REVISAR'})
        pendentes = df_audit[df_audit['categoria'] == 'REVISAR']
        assert len(pendentes) == 0, f'Auditoria incompleta: {len(pendentes)} fundos ainda estão REVISAR.'
        aceitos = df_audit[df_audit['categoria'].isin(['A','B'])]
        nao_confirmados = aceitos[~aceitos['sem_trilha_confirmado']]
        assert len(nao_confirmados) == 0, f'{len(nao_confirmados)} fundos aceitos sem confirmação de ausência de trilha.'
        assert df_audit['sha256_raw'].is_unique, 'Há duplicatas brutas por conteúdo.'
        assert df_audit['sha256_processado'].is_unique, 'Há duplicatas processadas por conteúdo.'

        nomes_aceitos = set(aceitos['arquivo_final'])
        CURATED.mkdir(parents=True, exist_ok=True)
        for caminho in CURATED.glob('*'):
            if caminho.is_file() and caminho.name not in nomes_aceitos:
                caminho.unlink()
        for nome in tqdm(sorted(nomes_aceitos), desc='Montando pasta curada local', unit='img'):
            origem_arq, destino_arq = BASE/nome, CURATED/nome
            if not destino_arq.exists() or sha256_arquivo(destino_arq) != sha256_arquivo(origem_arq):
                shutil.copy2(origem_arq, destino_arq)

        df_curado = df_audit.copy()
        df_curado.to_csv(PROC/'manifest_curado_v3.csv', index=False)
        assert set(p.name for p in CURATED.glob('*.png')) == nomes_aceitos
        print(pd.crosstab(df_curado['fonte'], df_curado['categoria']))
        print('A:', (df_curado.categoria=='A').sum(), 'B:', (df_curado.categoria=='B').sum(), 'C:', (df_curado.categoria=='C').sum())

        destino_proc = BACKUP_DIR/'data/processed_v3'
        destino_curado = destino_proc/'backgrounds_curated'
        destino_curado.mkdir(parents=True, exist_ok=True)
        for p in destino_curado.glob('*'):
            if p.is_file() and p.name not in nomes_aceitos: p.unlink()
        for nome in tqdm(sorted(nomes_aceitos), desc='Publicando curadoria no Drive', unit='img'):
            shutil.copy2(CURATED/nome, destino_curado/nome)
        shutil.copy2(PROC/'manifest_curado_v3.csv', destino_proc/'manifest_curado_v3.csv')
        print('Manifesto curado v3 salvo e espelhado.')
        ''') ,
    ]
    return write_notebook('02_sync_e_auditoria_dominio_ABC_v3.ipynb', cells)


def build_03():
    cells = [
        md(r'''
        # 03 — Split agrupado dos fundos (v3)

        O split é definido por hash antes da geração. Categoria A é estratificada por fonte;
        categoria B fica somente no treino; categoria C nunca entra.
        ''') ,
        code(r'''
        from google.colab import drive
        from pathlib import Path
        import hashlib, json, shutil
        import numpy as np
        import pandas as pd

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR = Path('/content/drive/MyDrive/tcc-satellite-streaks')
        PROC = Path('data/processed_v3'); PROC.mkdir(parents=True, exist_ok=True)
        origem = BACKUP_DIR/'data/processed_v3/manifest_curado_v3.csv'
        shutil.copy2(origem, PROC/'manifest_curado_v3.csv')
        df = pd.read_csv(PROC/'manifest_curado_v3.csv')

        SEED_SPLIT = 42
        PROPORCOES = {'train': 0.70, 'val': 0.15, 'test': 0.15}
        MIN_A_VAL_TEST = 40
        ''') ,
        md(r'''
        ## 1. Divisão determinística por fonte e hash
        ''') ,
        code(r'''
        assert not df['categoria'].isin(['REVISAR']).any()
        assert df['sha256_processado'].is_unique
        df_a = df[df.categoria == 'A'].copy()
        df_b = df[df.categoria == 'B'].copy()
        assert len(df_a) > 0

        def alocar_maior_resto(total, proporcoes):
            bruto = {k: total*v for k,v in proporcoes.items()}
            base = {k: int(np.floor(v)) for k,v in bruto.items()}
            faltam = total - sum(base.values())
            ordem = sorted(proporcoes, key=lambda k: (bruto[k]-base[k], k), reverse=True)
            for k in ordem[:faltam]: base[k] += 1
            return base

        partes = []
        for fonte, grupo in df_a.groupby('fonte', sort=True):
            grupo = grupo.sort_values('sha256_processado').copy()
            rng = np.random.default_rng(SEED_SPLIT + int(hashlib.sha256(fonte.encode()).hexdigest()[:8],16))
            ordem = rng.permutation(len(grupo))
            grupo = grupo.iloc[ordem].reset_index(drop=True)
            contagens = alocar_maior_resto(len(grupo), PROPORCOES)
            splits = (['train']*contagens['train'] + ['val']*contagens['val'] + ['test']*contagens['test'])
            grupo['split'] = splits
            partes.append(grupo)

        df_a_split = pd.concat(partes, ignore_index=True)
        df_b['split'] = 'train'
        df_split = pd.concat([df_a_split, df_b], ignore_index=True)
        ''') ,
        md(r'''
        ## 2. Gate de validade e salvamento
        ''') ,
        code(r'''
        assert df_split['id_fundo'].is_unique
        assert df_split.groupby('sha256_processado')['split'].nunique().max() == 1
        assert not ((df_split.categoria=='B') & (df_split.split!='train')).any()
        assert not df_split.categoria.eq('C').any()

        contagem_a = pd.crosstab(df_a_split['split'], df_a_split['fonte'])
        print('Categoria A por split/fonte:')
        print(contagem_a)
        print('\nTotal A por split:')
        print(df_a_split['split'].value_counts())
        n_val = int((df_a_split.split=='val').sum())
        n_test = int((df_a_split.split=='test').sum())
        assert n_val >= MIN_A_VAL_TEST and n_test >= MIN_A_VAL_TEST, (
            f'Diversidade insuficiente: A-val={n_val}, A-test={n_test}. Colete/audite mais fundos antes de gerar.'
        )

        saida = PROC/'manifest_split_fundos_v3.csv'
        df_split.sort_values(['split','categoria','fonte','id_fundo']).to_csv(saida, index=False)
        hash_manifest = hashlib.sha256(saida.read_bytes()).hexdigest()
        contrato = {
            'versao': 'v3', 'seed_split': SEED_SPLIT, 'proporcoes': PROPORCOES,
            'sha256_manifest_split': hash_manifest,
            'contagens': df_split['split'].value_counts().to_dict(),
            'contagens_A': df_a_split['split'].value_counts().to_dict(),
        }
        (PROC/'contrato_split_v3.json').write_text(json.dumps(contrato, indent=2), encoding='utf-8')
        destino = BACKUP_DIR/'data/processed_v3'; destino.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saida, destino/saida.name)
        shutil.copy2(PROC/'contrato_split_v3.json', destino/'contrato_split_v3.json')
        print('\nSplit v3 congelado:', hash_manifest)
        ''') ,
    ]
    return write_notebook('03_split_agrupado_dos_fundos_v3.ipynb', cells)


def build_04():
    cells = [
        md(r'''
        # 04 — Geração determinística de positivos e negativos (v3)

        Gera 3.000 positivos e 3.000 negativos em um diretório versionado pelo hash do
        manifesto e da configuração. Cada amostra tem seed independente, permitindo retomada
        idêntica. As OBBs são integralmente válidas no intervalo `[0,1]`.

        Fundos B são usados somente no treino e em proporção explícita. Os hard negatives desta
        versão são **sintéticos**; não representam uma avaliação de artefatos reais.
        ''') ,
        code(r'''
        %pip install -q opencv-python-headless pandas tqdm

        from google.colab import drive
        from pathlib import Path
        import hashlib, json, math, os, shutil, time
        import cv2
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        def copiar_arvore_com_progresso(origem, destino, descricao):
            origem, destino = Path(origem), Path(destino)
            arquivos = sorted(p for p in origem.rglob('*') if p.is_file())
            log_etapa(f'{descricao}: início ({len(arquivos)} arquivos)')
            inicio = time.perf_counter()
            for arquivo in tqdm(arquivos, desc=descricao, unit='arq'):
                alvo = destino / arquivo.relative_to(origem)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arquivo, alvo)
            log_etapa(f'{descricao}: concluído em {time.perf_counter()-inicio:.1f}s')

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR = Path('/content/drive/MyDrive/tcc-satellite-streaks')
        PROC = Path('data/processed_v3'); PROC.mkdir(parents=True, exist_ok=True)
        BG = PROC/'backgrounds_curated'; BG.mkdir(parents=True, exist_ok=True)

        for nome in ['manifest_split_fundos_v3.csv','contrato_split_v3.json']:
            shutil.copy2(BACKUP_DIR/'data/processed_v3'/nome, PROC/nome)
        copiar_arvore_com_progresso(
            BACKUP_DIR/'data/processed_v3/backgrounds_curated', BG,
            'Restaurando fundos curados'
        )
        df_fundos = pd.read_csv(PROC/'manifest_split_fundos_v3.csv')

        CONFIG = {
            'revisao_gerador': 3,
            'seed_mestra': 20260823,
            'total_positivos': 3000,
            'total_negativos': 3000,
            'split': {'train':0.70,'val':0.15,'test':0.15},
            'brilho': {'constante':0.60,'variavel':0.25,'tracejado':0.15},
            'negativos': {'limpo':0.80,'hard_sintetico':0.20},
            'fracao_B_no_treino': 0.20,
            'tamanho': 512,
            'n_segmentos': 96,
            'snr_pico': [3.0, 25.0],
            'sigma_ruido_augmentation': [0.0, 2.0],
        }

        bytes_contrato = (PROC/'manifest_split_fundos_v3.csv').read_bytes() + json.dumps(CONFIG,sort_keys=True).encode()
        RUN_ID = 'synthetic_v3_' + hashlib.sha256(bytes_contrato).hexdigest()[:12]
        RUN = Path('data/synthetic_runs')/RUN_ID
        DRIVE_RUN = BACKUP_DIR/'data/synthetic_runs'/RUN_ID
        if DRIVE_RUN.exists() and not RUN.exists():
            RUN.parent.mkdir(parents=True,exist_ok=True)
            copiar_arvore_com_progresso(DRIVE_RUN, RUN, 'Retomando RUN existente')
        RUN.mkdir(parents=True, exist_ok=True)
        print('RUN_ID:', RUN_ID)
        ''') ,
        md(r'''
        ## 1. Verificar os fundos contra o manifesto congelado
        ''') ,
        code(r'''
        def sha256_arquivo(caminho, bloco=1024*1024):
            h = hashlib.sha256()
            with open(caminho, 'rb') as arquivo:
                for parte in iter(lambda: arquivo.read(bloco), b''):
                    h.update(parte)
            return h.hexdigest()

        faltantes, divergentes = [], []
        for linha in df_fundos.itertuples():
            caminho = BG/linha.arquivo_final
            if not caminho.exists(): faltantes.append(linha.arquivo_final)
            elif sha256_arquivo(caminho) != linha.sha256_processado: divergentes.append(linha.arquivo_final)
        assert not faltantes and not divergentes, f'Fundos inválidos: faltantes={faltantes[:3]}, divergentes={divergentes[:3]}'
        assert df_fundos.groupby('sha256_processado')['split'].nunique().max() == 1
        assert not ((df_fundos.categoria=='B') & (df_fundos.split!='train')).any()
        print('Integridade dos fundos confirmada.')
        ''') ,
        md(r'''
        ## 2. Plano exato e balanceado
        ''') ,
        code(r'''
        def seed_estavel(texto):
            material = f"{CONFIG['seed_mestra']}::{texto}".encode()
            return int(hashlib.sha256(material).hexdigest()[:16], 16) % (2**32)

        def maior_resto(total, proporcoes):
            bruto = {k: total*v for k,v in proporcoes.items()}
            base = {k: int(np.floor(v)) for k,v in bruto.items()}
            faltam = total - sum(base.values())
            ordem = sorted(proporcoes, key=lambda k:(bruto[k]-base[k],k), reverse=True)
            for k in ordem[:faltam]: base[k] += 1
            return base

        def montar_tipo(tipo, total, subtipos):
            linhas = []
            cont_split = maior_resto(total, CONFIG['split'])
            for split in ['train','val','test']:
                n = cont_split[split]
                cont_sub = maior_resto(n, subtipos)
                lista_sub = [sub for sub in subtipos for _ in range(cont_sub[sub])]
                rng = np.random.default_rng(seed_estavel(f'plano::{tipo}::{split}'))
                rng.shuffle(lista_sub)
                if split == 'train':
                    n_b = int(round(n*CONFIG['fracao_B_no_treino']))
                    categorias = ['B']*n_b + ['A']*(n-n_b)
                    rng.shuffle(categorias)
                else:
                    categorias = ['A']*n
                for local, (subtipo, categoria) in enumerate(zip(lista_sub, categorias)):
                    id_amostra = f"{tipo[:3]}_{split}_{local:05d}"
                    linhas.append({'id':id_amostra,'tipo':tipo,'subtipo':subtipo,
                                   'split':split,'categoria_fundo_planejada':categoria,
                                   'seed_amostra':seed_estavel(id_amostra)})
            return linhas

        plano = pd.DataFrame(
            montar_tipo('positivo', CONFIG['total_positivos'], CONFIG['brilho']) +
            montar_tipo('negativo', CONFIG['total_negativos'], CONFIG['negativos'])
        )

        # Distribuição balanceada dos fundos dentro de cada split/categoria.
        plano['fundo_origem'] = ''
        for (split,categoria), indices in plano.groupby(['split','categoria_fundo_planejada']).groups.items():
            fundos = df_fundos[(df_fundos.split==split)&(df_fundos.categoria==categoria)].sort_values('sha256_processado')
            assert len(fundos) > 0, f'Sem fundos para {split}/{categoria}'
            nomes = fundos['arquivo_final'].tolist()
            rng = np.random.default_rng(seed_estavel(f'fundos::{split}::{categoria}'))
            atribuicao = []
            while len(atribuicao) < len(indices):
                ciclo = nomes.copy(); rng.shuffle(ciclo); atribuicao.extend(ciclo)
            plano.loc[list(indices),'fundo_origem'] = atribuicao[:len(indices)]

        mapa_meta = df_fundos.set_index('arquivo_final').to_dict('index')
        print(pd.crosstab([plano.split,plano.tipo],plano.subtipo))
        print('\nCategoria de fundo por split:')
        print(pd.crosstab(plano.split,plano.categoria_fundo_planejada))
        ''') ,
        md(r'''
        ## 3. Renderização e OBB integralmente interna

        O brilho é definido por SNR de pico aproximada em relação ao MAD robusto do fundo. Ruído
        adicional é tratado explicitamente como augmentation gaussiana, sem alegação de modelo
        físico do detector.
        ''') ,
        code(r'''
        def perfil_brilho(t, modo, amplitude, n_ciclos):
            if modo == 'constante': return 1.0
            onda = 0.5*(1+math.sin(2*math.pi*n_ciclos*t))
            if modo == 'variavel': return 1-amplitude+2*amplitude*onda
            if modo == 'tracejado': return onda if onda > 0.15 else 0.0
            raise ValueError(modo)

        def sigma_robusto(img):
            med = np.median(img)
            return max(1.0, 1.4826*np.median(np.abs(img.astype(float)-med)))

        def aumentar_fundo(img, rng):
            flip_h, flip_v = bool(rng.integers(0,2)), bool(rng.integers(0,2))
            if flip_h: img = cv2.flip(img,1)
            if flip_v: img = cv2.flip(img,0)
            ang = float(rng.uniform(-15,15))
            h,w = img.shape[:2]
            matriz = cv2.getRotationMatrix2D((w/2,h/2),ang,1.0)
            img = cv2.warpAffine(img,matriz,(w,h),borderMode=cv2.BORDER_REFLECT)
            return img, {'flip_h':flip_h,'flip_v':flip_v,'rotacao_fundo_deg':ang}

        def limite_ate_borda(cx,cy,ux,uy,w,h,margem,sinal):
            dx,dy = sinal*ux,sinal*uy
            limites=[]
            if dx>0: limites.append((w-1-margem-cx)/dx)
            elif dx<0: limites.append((margem-cx)/dx)
            if dy>0: limites.append((h-1-margem-cy)/dy)
            elif dy<0: limites.append((margem-cy)/dy)
            return min(v for v in limites if v>=0)

        def sortear_segmento_interno(w,h,rng,meia_largura):
            margem = float(math.ceil(meia_largura+2))
            diagonal = math.hypot(w,h)
            for _ in range(500):
                cx,cy = rng.uniform(margem,w-1-margem),rng.uniform(margem,h-1-margem)
                ang = rng.uniform(0,math.pi)
                ux,uy = math.cos(ang),math.sin(ang)
                t_pos = limite_ate_borda(cx,cy,ux,uy,w,h,margem,1)
                t_neg = limite_ate_borda(cx,cy,ux,uy,w,h,margem,-1)
                f_pos,f_neg = rng.uniform(0.55,0.98,2)
                p0=(cx-ux*t_neg*f_neg,cy-uy*t_neg*f_neg)
                p1=(cx+ux*t_pos*f_pos,cy+uy*t_pos*f_pos)
                if math.dist(p0,p1)>=0.25*diagonal:
                    return p0,p1,math.degrees(ang)
            raise RuntimeError('Não foi possível sortear segmento interno.')

        def cantos_obb(p0,p1,meia_largura,w,h):
            x0,y0=p0; x1,y1=p1; dx,dy=x1-x0,y1-y0
            comp=math.hypot(dx,dy); px,py=-dy/comp,dx/comp
            cantos=[(x0+px*meia_largura,y0+py*meia_largura),(x1+px*meia_largura,y1+py*meia_largura),
                    (x1-px*meia_largura,y1-py*meia_largura),(x0-px*meia_largura,y0-py*meia_largura)]
            norm=[(x/w,y/h) for x,y in cantos]
            assert all(0<=v<=1 for ponto in norm for v in ponto)
            return norm

        def render_trilha(img,rng,modo):
            h,w=img.shape[:2]
            esp=int(rng.integers(1,4)); blur=float(rng.uniform(0.8,2.2))
            meia=esp/2+2.5*blur
            p0,p1,ang=sortear_segmento_interno(w,h,rng,meia)
            amplitude=float(rng.uniform(0.3,0.5)); ciclos=float(rng.uniform(1.5,3.0))
            snr=float(rng.uniform(*CONFIG['snr_pico']))
            camada=np.zeros((h,w),np.float32)
            for i in range(CONFIG['n_segmentos']):
                ta,tb=i/CONFIG['n_segmentos'],(i+1)/CONFIG['n_segmentos']; tm=(ta+tb)/2
                intensidade=perfil_brilho(tm,modo,amplitude,ciclos)
                if intensidade<=0: continue
                a=(int(p0[0]+(p1[0]-p0[0])*ta),int(p0[1]+(p1[1]-p0[1])*ta))
                b=(int(p0[0]+(p1[0]-p0[0])*tb),int(p0[1]+(p1[1]-p0[1])*tb))
                cv2.line(camada,a,b,float(intensidade),esp,cv2.LINE_AA)
            camada=cv2.GaussianBlur(camada,(0,0),sigmaX=blur)
            pico_alvo=min(220.0,snr*sigma_robusto(img))
            camada*=pico_alvo/max(float(camada.max()),1e-6)
            resultado=np.clip(img.astype(np.float32)+camada,0,255)
            obb=cantos_obb(p0,p1,meia,w,h)
            return resultado,obb,{'angulo_trilha_deg':ang,'comprimento_px':math.dist(p0,p1),
                'espessura_px':esp,'desfoque_sigma':blur,'amplitude':amplitude,
                'n_ciclos':ciclos,'snr_pico_alvo':snr,'p0':list(p0),'p1':list(p1)}

        def render_pico_sintetico(img,rng):
            h,w=img.shape[:2]; camada=np.zeros((h,w),np.float32)
            cx,cy=int(rng.uniform(.2,.8)*w),int(rng.uniform(.2,.8)*h)
            comprimento=float(rng.uniform(.12,.30)*math.hypot(w,h)); base=float(rng.uniform(0,90))
            esp=int(rng.integers(1,4)); blur=float(rng.uniform(1.2,2.2)); brilho=float(rng.uniform(40,180))
            for offset in [0,90,180,270]:
                a=math.radians(base+offset); ponta=(cx+math.cos(a)*comprimento,cy+math.sin(a)*comprimento)
                for i in range(24):
                    t0,t1=i/24,(i+1)/24; intensidade=brilho*(1-(t0+t1)/2)**1.5
                    p0=(int(cx+(ponta[0]-cx)*t0),int(cy+(ponta[1]-cy)*t0))
                    p1=(int(cx+(ponta[0]-cx)*t1),int(cy+(ponta[1]-cy)*t1))
                    cv2.line(camada,p0,p1,float(intensidade),esp,cv2.LINE_AA)
            cv2.circle(camada,(cx,cy),2*esp,255,-1)
            camada=cv2.GaussianBlur(camada,(0,0),sigmaX=blur)
            return np.clip(img.astype(np.float32)+camada,0,255),{
                'pico_centro':[cx,cy],'pico_comprimento':comprimento,'pico_angulo_base':base,
                'pico_espessura':esp,'pico_blur':blur,'pico_brilho':brilho}

        def ruido_augmentation(img,rng):
            sigma=float(rng.uniform(*CONFIG['sigma_ruido_augmentation']))
            if sigma>0: img=img+rng.normal(0,sigma,img.shape)
            return np.clip(img,0,255).astype(np.uint8),sigma
        ''') ,
        md(r'''
        ## 4. Geração retomável por amostra

        Se uma execução for interrompida, cada ID é regenerado de forma idêntica. O manifesto
        parcial é salvo junto das imagens; não há dependência da posição atual do RNG global.
        ''') ,
        code(r'''
        for split in ['train','val','test']:
            (RUN/'images'/split).mkdir(parents=True,exist_ok=True)
            (RUN/'labels'/split).mkdir(parents=True,exist_ok=True)

        caminho_manifest=RUN/'manifest_geracao_v3.csv'
        registros_existentes={}
        if caminho_manifest.exists():
            antigo=pd.read_csv(caminho_manifest)
            registros_existentes={r['id']:r for r in antigo.to_dict('records')}
        registros=[]
        arquivos_pendentes_backup=[]

        def salvar_label(caminho,obb):
            if obb is None: caminho.write_text('',encoding='utf-8'); return
            valores=' '.join(f'{v:.8f}' for ponto in obb for v in ponto)
            caminho.write_text(f'0 {valores}\n',encoding='utf-8')

        def salvar_manifesto_atomico(registros):
            tmp=caminho_manifest.with_suffix('.tmp.csv')
            pd.DataFrame(registros).to_csv(tmp,index=False)
            tmp.replace(caminho_manifest)

        def sincronizar_pendentes(descricao):
            unicos = sorted(set(arquivos_pendentes_backup), key=lambda p: p.as_posix())
            if unicos:
                log_etapa(f'{descricao}: {len(unicos)} arquivos novos')
                for arquivo in tqdm(unicos, desc=descricao, unit='arq'):
                    destino = DRIVE_RUN / arquivo.relative_to(RUN)
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(arquivo, destino)
                arquivos_pendentes_backup.clear()
            if caminho_manifest.exists():
                DRIVE_RUN.mkdir(parents=True, exist_ok=True)
                shutil.copy2(caminho_manifest, DRIVE_RUN/caminho_manifest.name)

        for numero,linha in enumerate(tqdm(plano.to_dict('records'),desc='Gerando v3')):
            img_path=RUN/'images'/linha['split']/(linha['id']+'.png')
            lbl_path=RUN/'labels'/linha['split']/(linha['id']+'.txt')
            anterior=registros_existentes.get(linha['id'])
            if anterior and img_path.exists() and lbl_path.exists() and anterior.get('fundo_origem')==linha['fundo_origem']:
                if sha256_arquivo(img_path)==anterior.get('sha256_imagem') and sha256_arquivo(lbl_path)==anterior.get('sha256_label'):
                    registros.append(anterior); continue

            rng=np.random.default_rng(int(linha['seed_amostra']))
            fundo=cv2.imread(str(BG/linha['fundo_origem']),cv2.IMREAD_GRAYSCALE)
            assert fundo is not None
            fundo,meta_aug=aumentar_fundo(fundo,rng)
            params={}
            if linha['tipo']=='positivo':
                resultado,obb,params=render_trilha(fundo,rng,linha['subtipo'])
            elif linha['subtipo']=='hard_sintetico':
                resultado,params=render_pico_sintetico(fundo,rng); obb=None
            else:
                resultado=fundo.astype(np.float32); obb=None
            resultado,sigma_aug=ruido_augmentation(resultado,rng)
            cv2.imwrite(str(img_path),resultado); salvar_label(lbl_path,obb)
            meta_fundo=mapa_meta[linha['fundo_origem']]
            registro={**linha,'id_fundo':meta_fundo['id_fundo'],'fonte_fundo':meta_fundo['fonte'],
                'categoria_fundo':meta_fundo['categoria'],'sha256_fundo':meta_fundo['sha256_processado'],
                **meta_aug,**params,'sigma_ruido_augmentation':sigma_aug,
                'sha256_imagem':sha256_arquivo(img_path),'sha256_label':sha256_arquivo(lbl_path)}
            registros.append(registro)
            arquivos_pendentes_backup.extend([img_path, lbl_path])
            if (numero+1)%200==0:
                salvar_manifesto_atomico(registros)
                sincronizar_pendentes(f'Checkpoint {numero+1}/{len(plano)}')

        salvar_manifesto_atomico(registros)
        (RUN/'config_geracao_v3.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
        (RUN/'run_id.txt').write_text(RUN_ID+'\n',encoding='utf-8')
        arquivos_pendentes_backup.extend([RUN/'config_geracao_v3.json', RUN/'run_id.txt'])

        # Remove sobras apenas dentro deste RUN_ID versionado.
        esperados={(r['split'],r['id']) for r in registros}
        for split in ['train','val','test']:
            for p in (RUN/'images'/split).glob('*.png'):
                if (split,p.stem) not in esperados: p.unlink()
            for p in (RUN/'labels'/split).glob('*.txt'):
                if (split,p.stem) not in esperados: p.unlink()

        sincronizar_pendentes('Backup final dos arquivos novos')
        # Remove sobras remotas somente dentro deste RUN_ID versionado.
        for split in ['train','val','test']:
            for pasta,sufixo in [('images','.png'),('labels','.txt')]:
                remoto=DRIVE_RUN/pasta/split
                if remoto.exists():
                    for p in remoto.glob('*'+sufixo):
                        if (split,p.stem) not in esperados: p.unlink()
        print('Geração concluída localmente:',RUN_ID,len(registros))
        print('O ponteiro LATEST_RUN_V3 ainda NÃO foi publicado; execute o gate rápido.')
        ''') ,
        md(r'''
        ## 5. Gate rápido
        ''') ,
        code(r'''
        df_final=pd.read_csv(RUN/'manifest_geracao_v3.csv')
        assert len(df_final)==CONFIG['total_positivos']+CONFIG['total_negativos']
        assert df_final['id'].is_unique
        assert (df_final.tipo=='positivo').sum()==CONFIG['total_positivos']
        assert (df_final.tipo=='negativo').sum()==CONFIG['total_negativos']
        assert not ((df_final.categoria_fundo=='B')&(df_final.split!='train')).any()
        assert df_final.groupby('sha256_fundo')['split'].nunique().max()==1
        print(pd.crosstab([df_final.split,df_final.tipo],df_final.subtipo))
        print('\nUso de fundos únicos:',df_final.groupby('split')['sha256_fundo'].nunique().to_dict())

        # Somente um RUN aprovado pelo gate se torna a versão corrente para os notebooks seguintes.
        ponteiro_local=Path('data/synthetic_runs/LATEST_RUN_V3.txt')
        ponteiro_local.parent.mkdir(parents=True,exist_ok=True)
        ponteiro_local.write_text(RUN_ID+'\n',encoding='utf-8')
        ponteiro_drive=BACKUP_DIR/'data/synthetic_runs/LATEST_RUN_V3.txt'
        ponteiro_drive.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ponteiro_local,ponteiro_drive)
        log_etapa(f'Gate rápido aprovado; ponteiro publicado para {RUN_ID}')
        ''') ,
    ]
    return write_notebook('04_geracao_positivos_negativos_deterministica_v3.ipynb', cells)


def build_05():
    cells = [
        md(r'''
        # 05 — Validação formal, visual e smoke test Ultralytics (v3)

        Este notebook valida conjuntos exatos de arquivos, labels OBB em `[0,1]`, integridade por
        hash, ausência de leakage e compatibilidade real com o carregador Ultralytics. A aprovação
        exige validação automática, revisão visual registrada e smoke test.
        ''') ,
        code(r'''
        %pip install -q opencv-python-headless pandas matplotlib pyyaml ultralytics

        from google.colab import drive
        from pathlib import Path
        import hashlib, json, math, os, shutil, time
        import cv2
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import yaml
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        def copiar_arvore_com_progresso(origem, destino, descricao):
            origem, destino = Path(origem), Path(destino)
            arquivos = sorted(p for p in origem.rglob('*') if p.is_file())
            log_etapa(f'{descricao}: início ({len(arquivos)} arquivos)')
            inicio = time.perf_counter()
            for arquivo in tqdm(arquivos, desc=descricao, unit='arq'):
                alvo = destino / arquivo.relative_to(origem)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arquivo, alvo)
            log_etapa(f'{descricao}: concluído em {time.perf_counter()-inicio:.1f}s')

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        ponteiro=BACKUP_DIR/'data/synthetic_runs/LATEST_RUN_V3.txt'
        RUN_ID=ponteiro.read_text(encoding='utf-8').strip()
        DRIVE_RUN=BACKUP_DIR/'data/synthetic_runs'/RUN_ID
        RUN=Path('data/synthetic_runs')/RUN_ID
        copiar_arvore_com_progresso(DRIVE_RUN, RUN, 'Restaurando dataset sintético')
        df=pd.read_csv(RUN/'manifest_geracao_v3.csv')
        CONFIG=json.loads((RUN/'config_geracao_v3.json').read_text(encoding='utf-8'))
        HASH_MANIFEST=hashlib.sha256((RUN/'manifest_geracao_v3.csv').read_bytes()).hexdigest()
        print(RUN_ID,len(df),HASH_MANIFEST)
        ''') ,
        md(r'''
        ## 1. Arquivos exatos, hashes e imagens válidas
        ''') ,
        code(r'''
        def sha256_arquivo(caminho,bloco=1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as arq:
                for parte in iter(lambda:arq.read(bloco),b''): h.update(parte)
            return h.hexdigest()

        problemas=[]
        for split in ['train','val','test']:
            esperado=set(df.loc[df.split==split,'id'])
            imgs={p.stem for p in (RUN/'images'/split).glob('*.png')}
            lbls={p.stem for p in (RUN/'labels'/split).glob('*.txt')}
            if imgs!=esperado: problemas.append(f'{split}: imagens faltantes/sobras: -{len(esperado-imgs)} +{len(imgs-esperado)}')
            if lbls!=esperado: problemas.append(f'{split}: labels faltantes/sobras: -{len(esperado-lbls)} +{len(lbls-esperado)}')

        for linha in tqdm(df.itertuples(), total=len(df), desc='Validando arquivos e hashes', unit='amostra'):
            imgp=RUN/'images'/linha.split/(linha.id+'.png'); lblp=RUN/'labels'/linha.split/(linha.id+'.txt')
            if imgp.exists() and sha256_arquivo(imgp)!=linha.sha256_imagem: problemas.append(f'{linha.id}: hash da imagem divergente')
            if lblp.exists() and sha256_arquivo(lblp)!=linha.sha256_label: problemas.append(f'{linha.id}: hash do label divergente')
            img=cv2.imread(str(imgp),cv2.IMREAD_GRAYSCALE) if imgp.exists() else None
            if img is None or img.shape!=(CONFIG['tamanho'],CONFIG['tamanho']): problemas.append(f'{linha.id}: imagem inválida')

        print('Problemas de arquivos:',len(problemas))
        if problemas: print('\n'.join(problemas[:20]))
        assert not problemas
        ''') ,
        md(r'''
        ## 2. Labels YOLO-OBB estritos

        Diferentemente da versão anterior, qualquer coordenada fora de `[0,1]` reprova o dataset.
        ''') ,
        code(r'''
        def area_poligono(p):
            return abs(sum(p[i][0]*p[(i+1)%4][1]-p[(i+1)%4][0]*p[i][1] for i in range(4)))/2

        def convexo(p,tol=1e-10):
            cruz=[]
            for i in range(4):
                a,b,c=np.array(p[i]),np.array(p[(i+1)%4]),np.array(p[(i+2)%4])
                ab,bc=b-a,c-b
                # Produto vetorial escalar em 2D; evita a depreciação de np.cross
                # para vetores bidimensionais introduzida no NumPy 2.0.
                cruz.append(ab[0]*bc[1]-ab[1]*bc[0])
            return all(v>=-tol for v in cruz) or all(v<=tol for v in cruz)

        erros_label=[]; minimo=1.0; maximo=0.0; n_fora=0
        for linha in tqdm(df.itertuples(), total=len(df), desc='Validando labels OBB', unit='label'):
            caminho=RUN/'labels'/linha.split/(linha.id+'.txt')
            conteudo=caminho.read_text(encoding='utf-8').strip()
            if linha.tipo=='negativo':
                if conteudo: erros_label.append((linha.id,'negativo com label não vazio'))
                continue
            linhas=[l for l in conteudo.splitlines() if l.strip()]
            if len(linhas)!=1: erros_label.append((linha.id,f'{len(linhas)} objetos, esperado 1')); continue
            tokens=linhas[0].split()
            if len(tokens)!=9: erros_label.append((linha.id,f'{len(tokens)} tokens')); continue
            if tokens[0]!='0': erros_label.append((linha.id,'classe diferente de 0'))
            try: coords=[float(x) for x in tokens[1:]]
            except ValueError: erros_label.append((linha.id,'coordenada não numérica')); continue
            minimo=min(minimo,min(coords)); maximo=max(maximo,max(coords))
            fora=sum(c<0 or c>1 for c in coords); n_fora+=fora
            if fora: erros_label.append((linha.id,'coordenada fora de [0,1]'))
            pontos=[(coords[i],coords[i+1]) for i in range(0,8,2)]
            if area_poligono(pontos)<1e-7: erros_label.append((linha.id,'área degenerada'))
            if not convexo(pontos): erros_label.append((linha.id,'ordem não convexa'))

        print(f'Faixa observada: [{minimo:.8f}, {maximo:.8f}] | coordenadas fora: {n_fora}')
        print('Erros de label:',len(erros_label))
        if erros_label: print(erros_label[:20])
        assert not erros_label
        ''') ,
        md(r'''
        ## 3. Leakage, duplicatas, composição e pseudo-replicação
        ''') ,
        code(r'''
        assert df['id'].is_unique
        assert df.groupby('sha256_fundo')['split'].nunique().max()==1
        assert not ((df.categoria_fundo=='B')&(df.split!='train')).any()
        assert (df.tipo=='positivo').sum()==3000 and (df.tipo=='negativo').sum()==3000

        hashes_multi_split=df.groupby('sha256_imagem')['split'].nunique()
        duplicatas_cruzadas=hashes_multi_split[hashes_multi_split>1]
        assert len(duplicatas_cruzadas)==0,'Imagem sintética idêntica em splits diferentes.'
        duplicatas_mesmo_split=df.duplicated(['split','sha256_imagem']).sum()
        print('Duplicatas dentro do mesmo split:',duplicatas_mesmo_split)
        print('Fundos únicos por split:',df.groupby('split')['sha256_fundo'].nunique().to_dict())
        print('\nDistribuição:')
        print(pd.crosstab([df.split,df.tipo],df.subtipo))
        print('\nCategoria de fundo:')
        print(pd.crosstab(df.split,df.categoria_fundo))
        print('\nUso mínimo/máximo de um fundo por split:')
        print(df.groupby(['split','sha256_fundo']).size().groupby('split').agg(['min','max','median']))
        ''') ,
        md(r'''
        ## 4. Revisão visual estratificada registrada

        A célula produz exatamente 50 itens por split em duas visões com ordem idêntica:
        imagem sem marcação, para julgar realismo/visibilidade, e overlay OBB, para julgar a
        geometria. Depois de inspecionar as seis grades, altere `APROVAR_REVISAO_VISUAL` para
        `True` e execute a célula de registro.
        ''') ,
        code(r'''
        def maior_resto(total,proporcoes):
            bruto={k:total*v for k,v in proporcoes.items()}; base={k:int(np.floor(v)) for k,v in bruto.items()}
            for k in sorted(proporcoes,key=lambda k:(bruto[k]-base[k],k),reverse=True)[:total-sum(base.values())]: base[k]+=1
            return base

        def amostra_exata(df_split,n=50):
            tamanhos=df_split.groupby(['tipo','subtipo']).size()
            props={k:v/len(df_split) for k,v in tamanhos.items()}
            cont=maior_resto(n,props); partes=[]
            for chave,quantidade in cont.items():
                g=df_split[(df_split.tipo==chave[0])&(df_split.subtipo==chave[1])]
                partes.append(g.sample(quantidade,random_state=20260823))
            return pd.concat(partes).sample(frac=1,random_state=20260823).reset_index(drop=True)

        revisados=[]
        for split in ['train','val','test']:
            amostra=amostra_exata(df[df.split==split],50)
            revisados.append(amostra.assign(split_revisado=split))
            for prefixo,desenhar_obb in [('raw',False),('overlay',True)]:
                fig,axes=plt.subplots(5,10,figsize=(20,11)); axes=axes.flatten()
                for ax,linha in zip(axes,amostra.itertuples()):
                    img=cv2.imread(str(RUN/'images'/split/(linha.id+'.png')),cv2.IMREAD_GRAYSCALE)
                    ax.imshow(img,cmap='gray',vmin=0,vmax=255)
                    if desenhar_obb:
                        texto=(RUN/'labels'/split/(linha.id+'.txt')).read_text().split()
                        if len(texto)==9:
                            c=[float(x) for x in texto[1:]]
                            pts=[(c[i]*img.shape[1],c[i+1]*img.shape[0]) for i in range(0,8,2)]
                            ax.add_patch(plt.Polygon(pts,fill=False,edgecolor='red',linewidth=.8))
                    ax.set_title(f'{linha.tipo[:3]}/{linha.subtipo}\n{linha.fonte_fundo}/{linha.categoria_fundo}',fontsize=5)
                    ax.axis('off')
                visao='sem overlay' if not desenhar_obb else 'com OBB'
                plt.suptitle(f'{RUN_ID} — {split} — 50 amostras — {visao}')
                plt.tight_layout()
                plt.savefig(RUN/f'{prefixo}_validacao_{split}.png',dpi=160,bbox_inches='tight')
                plt.show(); plt.close(fig)
        df_revisados=pd.concat(revisados,ignore_index=True)
        df_revisados[['id','split','tipo','subtipo','fundo_origem','sha256_imagem']].to_csv(RUN/'amostra_revisao_visual.csv',index=False)
        ''') ,
        code(r'''
        APROVAR_REVISAO_VISUAL=False  # mude para True apenas depois de examinar as seis grades
        if APROVAR_REVISAO_VISUAL:
            registro={'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,'aprovado':True,
                      'sha256_amostra':sha256_arquivo(RUN/'amostra_revisao_visual.csv')}
            (RUN/'revisao_visual_aprovada.json').write_text(json.dumps(registro,indent=2),encoding='utf-8')
            artefatos_visual = [
                RUN/'revisao_visual_aprovada.json', RUN/'amostra_revisao_visual.csv',
                RUN/'raw_validacao_train.png', RUN/'raw_validacao_val.png',
                RUN/'raw_validacao_test.png',
                RUN/'overlay_validacao_train.png', RUN/'overlay_validacao_val.png',
                RUN/'overlay_validacao_test.png'
            ]
            for arquivo in tqdm(artefatos_visual, desc='Salvando revisão visual', unit='arq'):
                shutil.copy2(arquivo, DRIVE_RUN/arquivo.name)
            print('Revisão visual registrada.')
        else:
            print('Revisão visual ainda não aprovada.')
        ''') ,
        md(r'''
        ## 5. Smoke test real do Ultralytics

        O teste usa apenas uma fração do treino e validação. Ele existe para confirmar que o
        carregador aceita todos os formatos; não produz resultado científico.
        ''') ,
        code(r'''
        dataset_yaml={
            'path':str(RUN.resolve()),'train':'images/train','val':'images/val','test':'images/test',
            'names':{0:'satellite_streak'}
        }
        yaml_path=RUN/'dataset_v3.yaml'
        yaml_path.write_text(yaml.safe_dump(dataset_yaml,sort_keys=False),encoding='utf-8')

        PROTOCOLO_SMOKE='balanced_train_val_v2'
        EXECUTAR_SMOKE_ULTRALYTICS=False  # mude para True quando a validação anterior estiver limpa
        if EXECUTAR_SMOKE_ULTRALYTICS:
            import torch
            from ultralytics import YOLO

            # Não use ``fraction`` sobre os nomes neg_*/pos_*: o fatiamento ordenado pode
            # selecionar somente negativos e produzir um falso positivo no gate.
            grupos=[('positivo','constante'),('positivo','variavel'),('positivo','tracejado'),
                    ('negativo','limpo'),('negativo','hard_sintetico')]

            def selecionar_smoke(split,n_por_grupo=2):
                partes=[]
                for indice,(tipo,subtipo) in enumerate(grupos):
                    candidatos=df[(df.split==split)&(df.tipo==tipo)&(df.subtipo==subtipo)]
                    assert len(candidatos)>=n_por_grupo,f'Sem amostras suficientes para {split}/{tipo}/{subtipo}'
                    partes.append(candidatos.sample(n_por_grupo,random_state=20260824+indice))
                amostra=pd.concat(partes,ignore_index=True).sort_values('id').reset_index(drop=True)
                for linha in amostra.itertuples():
                    tokens=(RUN/'labels'/split/(linha.id+'.txt')).read_text(encoding='utf-8').split()
                    esperado=9 if linha.tipo=='positivo' else 0
                    assert len(tokens)==esperado,f'Label inesperado no smoke: {linha.id}'
                return amostra

            smoke_dir=RUN/'smoke_protocol_v2'
            smoke_dir.mkdir(parents=True,exist_ok=True)
            smoke_amostras={split:selecionar_smoke(split) for split in ['train','val']}
            for split,amostra in smoke_amostras.items():
                caminhos=[str((RUN/'images'/split/(id_amostra+'.png')).resolve()) for id_amostra in amostra.id]
                (smoke_dir/f'{split}.txt').write_text('\n'.join(caminhos)+'\n',encoding='utf-8')

            smoke_yaml={
                'path':str(RUN.resolve()),
                'train':str((smoke_dir/'train.txt').resolve()),
                'val':str((smoke_dir/'val.txt').resolve()),
                'names':{0:'satellite_streak'}
            }
            smoke_yaml_path=smoke_dir/'dataset_smoke_balanced_v2.yaml'
            smoke_yaml_path.write_text(yaml.safe_dump(smoke_yaml,sort_keys=False),encoding='utf-8')
            print('Amostra balanceada do smoke:')
            print(pd.concat(smoke_amostras.values(),ignore_index=True).groupby(['split','tipo','subtipo']).size())

            projeto_smoke=(RUN/'smoke_ultralytics').resolve()
            modelo=YOLO('yolo11n-obb.pt')
            modelo.train(data=str(smoke_yaml_path.resolve()),epochs=1,imgsz=256,batch=4,
                         workers=2,device=0 if torch.cuda.is_available() else 'cpu',
                         project=str(projeto_smoke),name='loader_check_balanced_v2',exist_ok=True,
                         plots=False,verbose=True)
            save_dir=Path(modelo.trainer.save_dir).resolve()
            assert save_dir==projeto_smoke/'loader_check_balanced_v2','Ultralytics salvou fora do diretório esperado.'
            best=save_dir/'weights'/'best.pt'; last=save_dir/'weights'/'last.pt'
            assert best.is_file() and last.is_file(),'Pesos do smoke test não foram produzidos.'

            registro={
                'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,'aprovado':True,
                'modelo':'yolo11n-obb.pt','protocolo':PROTOCOLO_SMOKE,
                'treino':{'imagens':len(smoke_amostras['train']),
                          'positivas':int((smoke_amostras['train'].tipo=='positivo').sum()),
                          'negativas':int((smoke_amostras['train'].tipo=='negativo').sum())},
                'validacao':{'imagens':len(smoke_amostras['val']),
                             'positivas':int((smoke_amostras['val'].tipo=='positivo').sum()),
                             'negativas':int((smoke_amostras['val'].tipo=='negativo').sum())},
                'sha256_best_pt':sha256_arquivo(best)
            }
            tmp=RUN/'smoke_ultralytics_ok.json.tmp'
            tmp.write_text(json.dumps(registro,indent=2),encoding='utf-8')
            tmp.replace(RUN/'smoke_ultralytics_ok.json')
            shutil.copy2(yaml_path, DRIVE_RUN/yaml_path.name)
            copiar_arvore_com_progresso(smoke_dir, DRIVE_RUN/'smoke_protocol_v2',
                                        'Salvando contrato do smoke test')
            copiar_arvore_com_progresso(projeto_smoke, DRIVE_RUN/'smoke_ultralytics',
                                        'Salvando resultados do smoke test')
            shutil.copy2(RUN/'smoke_ultralytics_ok.json', DRIVE_RUN/'smoke_ultralytics_ok.json')
            print(json.dumps(registro,indent=2,ensure_ascii=False))
            print('Smoke test balanceado aprovado.')
        else:
            print('Smoke test não executado nesta célula.')
        ''') ,
        md(r'''
        ## 6. Gate final da versão v3
        ''') ,
        code(r'''
        PROTOCOLO_SMOKE_ESPERADO='balanced_train_val_v2'

        def aprovacao_valida(nome,protocolo=None):
            caminho=RUN/nome
            if not caminho.exists(): return False
            registro=json.loads(caminho.read_text(encoding='utf-8'))
            base=registro.get('aprovado') is True and registro.get('sha256_manifest')==HASH_MANIFEST
            return base and (protocolo is None or registro.get('protocolo')==protocolo)

        visual_ok=aprovacao_valida('revisao_visual_aprovada.json')
        smoke_ok=aprovacao_valida('smoke_ultralytics_ok.json',PROTOCOLO_SMOKE_ESPERADO)
        resumo={'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,'arquivos_ok':True,'labels_ok':True,
                'leakage_ok':True,'revisao_visual_ok':visual_ok,'smoke_ultralytics_ok':smoke_ok,
                'aprovado_para_treino':bool(visual_ok and smoke_ok)}
        (RUN/'resumo_validacao_v3.json').write_text(json.dumps(resumo,indent=2),encoding='utf-8')
        shutil.copy2(RUN/'resumo_validacao_v3.json', DRIVE_RUN/'resumo_validacao_v3.json')
        print(json.dumps(resumo,indent=2,ensure_ascii=False))
        assert resumo['aprovado_para_treino'],'Dataset ainda não passou por revisão visual e smoke test registrados.'
        ''') ,
    ]
    return write_notebook('05_validacao_formal_visual_ultralytics_v3.ipynb', cells)


def build_06():
    cells = [
        md(r'''
        # 06 — Baseline Hough com protocolo congelado (v3)

        O notebook só inicia após a aprovação formal do dataset v3. Detector, NMS e thresholds
        geométricos são escolhidos exclusivamente na validação. O teste possui trava persistente
        no Drive e não roda por padrão.

        São reportados separadamente: IoU, geometria de centerline e critério combinado.
        ''') ,
        code(r'''
        %pip install -q opencv-python-headless pandas tqdm

        from google.colab import drive
        from pathlib import Path
        import hashlib, json, math, shutil, time
        import cv2
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        def copiar_arvore_com_progresso(origem, destino, descricao):
            origem, destino = Path(origem), Path(destino)
            arquivos = sorted(p for p in origem.rglob('*') if p.is_file())
            log_etapa(f'{descricao}: início ({len(arquivos)} arquivos)')
            inicio = time.perf_counter()
            for arquivo in tqdm(arquivos, desc=descricao, unit='arq'):
                alvo = destino / arquivo.relative_to(origem)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arquivo, alvo)
            log_etapa(f'{descricao}: concluído em {time.perf_counter()-inicio:.1f}s')

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        RUN_ID=(BACKUP_DIR/'data/synthetic_runs/LATEST_RUN_V3.txt').read_text().strip()
        DRIVE_RUN=BACKUP_DIR/'data/synthetic_runs'/RUN_ID
        RUN=Path('data/synthetic_runs')/RUN_ID
        copiar_arvore_com_progresso(DRIVE_RUN, RUN, 'Restaurando RUN para o Hough')
        validacao=json.loads((RUN/'resumo_validacao_v3.json').read_text())
        assert validacao.get('aprovado_para_treino') is True,'Dataset v3 ainda não foi aprovado no notebook 05.'
        df=pd.read_csv(RUN/'manifest_geracao_v3.csv')
        HASH_MANIFEST=hashlib.sha256((RUN/'manifest_geracao_v3.csv').read_bytes()).hexdigest()
        assert HASH_MANIFEST==validacao['sha256_manifest']
        HOUGH=RUN/'hough_v3'; HOUGH.mkdir(parents=True,exist_ok=True)
        print(RUN_ID,HASH_MANIFEST)
        ''') ,
        md(r'''
        ## 1. Detector e NMS no espaço geométrico de linhas
        ''') ,
        code(r'''
        def parametros_linha(p0,p1):
            p0,p1=np.array(p0,dtype=float),np.array(p1,dtype=float)
            vetor=p1-p0; comp=float(np.linalg.norm(vetor)); direcao=vetor/comp if comp>0 else np.array([1.,0.])
            return (p0+p1)/2,direcao,comp

        def erro_angular(d1,d2): return float(np.degrees(np.arccos(np.clip(abs(np.dot(d1,d2)),-1,1))))

        def distancia_ponto_linha(ponto,centro,direcao):
            v=np.asarray(ponto)-np.asarray(centro)
            return float(np.linalg.norm(v-np.dot(v,direcao)*direcao))

        def cobertura_projetada(p0a,p1a,p0b,p1b):
            cb,db,lb=parametros_linha(p0b,p1b)
            if lb<=0:return 0.0
            t=sorted([np.dot(np.asarray(p0a)-cb,db),np.dot(np.asarray(p1a)-cb,db)])
            inter=max(0,min(t[1],lb/2)-max(t[0],-lb/2))
            return float(inter/lb)

        def detectar_hough(img,pre_blur_sigma,canny_lo,canny_hi,hough_threshold,min_length_frac,max_gap):
            proc=cv2.GaussianBlur(img,(0,0),pre_blur_sigma) if pre_blur_sigma>0 else img
            bordas=cv2.Canny(proc,canny_lo,canny_hi)
            min_len=int(math.hypot(*img.shape[:2])*min_length_frac)
            linhas=cv2.HoughLinesP(bordas,1,np.pi/180,hough_threshold,minLineLength=min_len,maxLineGap=max_gap)
            if linhas is None:return []
            # Compatível com retornos (N,1,4) e (N,4) do OpenCV.
            segmentos=np.asarray(linhas).reshape(-1,4)
            return [((int(x1),int(y1)),(int(x2),int(y2))) for x1,y1,x2,y2 in segmentos]

        def nms_linhas(linhas,angulo_max=3.0,dist_max=12.0,cobertura_min=0.30):
            ordenadas=sorted(linhas,key=lambda l:-math.dist(l[0],l[1])); mantidas=[]
            for atual in ordenadas:
                ca,da,_=parametros_linha(*atual); suprimir=False
                for mantida in mantidas:
                    cm,dm,_=parametros_linha(*mantida)
                    if erro_angular(da,dm)>angulo_max:continue
                    if max(distancia_ponto_linha(ca,cm,dm),distancia_ponto_linha(cm,ca,da))>dist_max:continue
                    cob=max(cobertura_projetada(*atual,*mantida),cobertura_projetada(*mantida,*atual))
                    if cob>=cobertura_min:suprimir=True;break
                if not suprimir:mantidas.append(atual)
            return mantidas

        def linha_para_obb(p0,p1,espessura,w,h):
            x0,y0=p0;x1,y1=p1;dx,dy=x1-x0,y1-y0;comp=math.hypot(dx,dy) or 1e-6
            px,py=-dy/comp,dx/comp;m=espessura/2
            return [((x0+px*m)/w,(y0+py*m)/h),((x1+px*m)/w,(y1+py*m)/h),
                    ((x1-px*m)/w,(y1-py*m)/h),((x0-px*m)/w,(y0-py*m)/h)]

        def iou_obb(a,b,w,h):
            pa=np.array([(x*w,y*h) for x,y in a],np.int32);pb=np.array([(x*w,y*h) for x,y in b],np.int32)
            ma=np.zeros((h,w),np.uint8);mb=np.zeros((h,w),np.uint8)
            cv2.fillPoly(ma,[pa],1);cv2.fillPoly(mb,[pb],1)
            inter=np.logical_and(ma,mb).sum();uniao=np.logical_or(ma,mb).sum()
            return float(inter/uniao) if uniao else 0.0

        def ler_obb(caminho):
            t=Path(caminho).read_text().split()
            if not t:return None
            v=[float(x) for x in t[1:9]]
            return [(v[i],v[i+1]) for i in range(0,8,2)]

        def obb_para_centerline(obb,w,h):
            p=np.array([(x*w,y*h) for x,y in obb]);d01=np.linalg.norm(p[0]-p[1]);d12=np.linalg.norm(p[1]-p[2])
            if d01>=d12:return tuple((p[0]+p[3])/2),tuple((p[1]+p[2])/2)
            return tuple((p[0]+p[1])/2),tuple((p[2]+p[3])/2)
        ''') ,
        md(r'''
        ## 2. Métricas sem atalhos artificiais

        Ângulo, distância e cobertura são calculados para todo candidato, inclusive quando o IoU
        já ultrapassa o limiar. Assim, as médias finais representam medições reais.
        ''') ,
        code(r'''
        def metricas_candidato(linha,gt,w,h,espessura_match=3):
            p0,p1=linha;obb=linha_para_obb(p0,p1,espessura_match,w,h)
            g0,g1=obb_para_centerline(gt,w,h);cc,dc,_=parametros_linha(p0,p1);cg,dg,_=parametros_linha(g0,g1)
            return {'p0':p0,'p1':p1,'iou':iou_obb(obb,gt,w,h),'angulo':erro_angular(dc,dg),
                    'distancia':distancia_ponto_linha(cc,cg,dg),'cobertura':cobertura_projetada(p0,p1,g0,g1)}

        def escolher_match(metricas,limiares,modo):
            candidatos=[]
            for m in metricas:
                miou=m['iou']>=limiares['iou']
                mgeo=(m['angulo']<=limiares['angulo'] and m['distancia']<=limiares['distancia'] and
                      m['cobertura']>=limiares['cobertura'])
                bate={'iou':miou,'geometria':mgeo,'combinado':miou or mgeo}[modo]
                if not bate:continue
                score_iou=m['iou']/limiares['iou'] if miou else -np.inf
                score_geo=((1-m['angulo']/limiares['angulo'])+(1-m['distancia']/limiares['distancia'])+m['cobertura']) if mgeo else -np.inf
                score={'iou':score_iou,'geometria':score_geo,'combinado':max(score_iou,score_geo)}[modo]
                candidatos.append((score,m))
            return max(candidatos,key=lambda x:x[0])[1] if candidatos else None

        # Testes unitários mínimos
        assert abs(erro_angular(np.array([1.,0.]),np.array([-1.,0.])))<1e-9
        assert abs(erro_angular(np.array([1.,0.]),np.array([0.,1.]))-90)<1e-9
        assert abs(cobertura_projetada((0,0),(50,0),(0,0),(100,0))-.5)<1e-9
        assert abs(distancia_ponto_linha((0,10),(0,0),np.array([1.,0.]))-10)<1e-9
        print('Testes geométricos aprovados.')
        ''') ,
        md(r'''
        ## 3. Avaliação por imagem e conjunto
        ''') ,
        code(r'''
        def avaliar_imagem(linha,detector,limiares):
            imgp=RUN/'images'/linha.split/(linha.id+'.png');lblp=RUN/'labels'/linha.split/(linha.id+'.txt')
            img=cv2.imread(str(imgp),cv2.IMREAD_GRAYSCALE);h,w=img.shape
            brutas=detectar_hough(img,**detector);candidatos=nms_linhas(brutas)
            gt=ler_obb(lblp)
            base={'id':linha.id,'split':linha.split,'tipo':linha.tipo,'subtipo':linha.subtipo,
                  'fundo_origem':linha.fundo_origem,'sha256_fundo':linha.sha256_fundo,
                  'fonte_fundo':linha.fonte_fundo,'n_candidatos':len(candidatos)}
            if gt is None:
                for modo in ['iou','geometria','combinado']:
                    base[f'acertou_{modo}']=False;base[f'tp_{modo}']=0;base[f'fn_{modo}']=0;base[f'fp_{modo}']=len(candidatos)
                base.update({'angulo_match':np.nan,'distancia_match':np.nan,'cobertura_match':np.nan,'iou_match':np.nan})
                return base
            mets=[metricas_candidato(c,gt,w,h) for c in candidatos]
            matches={modo:escolher_match(mets,limiares,modo) for modo in ['iou','geometria','combinado']}
            for modo,m in matches.items():
                base[f'acertou_{modo}']=m is not None;base[f'tp_{modo}']=int(m is not None)
                base[f'fn_{modo}']=int(m is None);base[f'fp_{modo}']=max(0,len(candidatos)-int(m is not None))
            melhor=matches['combinado']
            base.update({'angulo_match':melhor['angulo'] if melhor else np.nan,
                         'distancia_match':melhor['distancia'] if melhor else np.nan,
                         'cobertura_match':melhor['cobertura'] if melhor else np.nan,
                         'iou_match':melhor['iou'] if melhor else np.nan})
            return base

        def resumir(resultados,modo):
            tp=int(resultados[f'tp_{modo}'].sum());fp=int(resultados[f'fp_{modo}'].sum());fn=int(resultados[f'fn_{modo}'].sum())
            p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f1=2*p*r/(p+r) if p+r else 0
            neg=resultados[resultados.tipo=='negativo']; taxa=float((neg[f'fp_{modo}']>0).mean()) if len(neg) else 0
            return {'modo':modo,'TP':tp,'FP':fp,'FN':fn,'precisao':p,'recall':r,'f1':f1,
                    'fp_por_imagem_negativa':float(neg[f'fp_{modo}'].mean()) if len(neg) else 0,
                    'taxa_imagens_negativas_com_fp':taxa}

        def avaliar_conjunto(dados,detector,limiares,descricao):
            linhas=[avaliar_imagem(l,detector,limiares) for l in tqdm(dados.itertuples(),total=len(dados),desc=descricao)]
            resultados=pd.DataFrame(linhas)
            resumos=pd.DataFrame([resumir(resultados,m) for m in ['iou','geometria','combinado']])
            return resultados,resumos
        ''') ,
        md(r'''
        ## 4. Calibração exclusiva na validação

        Distância geométrica é limitada a 6–14 px, em vez de 5% da diagonal (~36 px).
        ''') ,
        code(r'''
        DETECTORES=[
          {'nome':'permissivo','pre_blur_sigma':0,'canny_lo':50,'canny_hi':150,'hough_threshold':40,'min_length_frac':.15,'max_gap':15},
          {'nome':'rigoroso','pre_blur_sigma':0,'canny_lo':50,'canny_hi':150,'hough_threshold':90,'min_length_frac':.25,'max_gap':8},
          {'nome':'pre_desfoque','pre_blur_sigma':2,'canny_lo':50,'canny_hi':150,'hough_threshold':70,'min_length_frac':.20,'max_gap':20},
        ]
        MATCHERS=[
          {'nome_match':'estrito','iou':.30,'angulo':5.0,'distancia':6.0,'cobertura':.70},
          {'nome_match':'equilibrado','iou':.30,'angulo':10.0,'distancia':10.0,'cobertura':.60},
          {'nome_match':'amplo','iou':.30,'angulo':15.0,'distancia':14.0,'cobertura':.50},
        ]

        df_val=df[df.split=='val'].copy()
        grupos=df_val.groupby(['tipo','subtipo']);partes=[]
        for _,g in grupos:
            n=max(1,round(300*len(g)/len(df_val)));partes.append(g.sample(min(n,len(g)),random_state=7))
        amostra=pd.concat(partes).drop_duplicates('id').head(300)
        calibracoes=[]
        for det in DETECTORES:
            nome_det=det['nome'];params_det={k:v for k,v in det.items() if k!='nome'}
            for matcher in MATCHERS:
                resultados,resumos=avaliar_conjunto(amostra,params_det,matcher,f'{nome_det}/{matcher["nome_match"]}')
                combinado=resumos[resumos.modo=='combinado'].iloc[0].to_dict()
                calibracoes.append({'detector':nome_det,**params_det,**matcher,**{f'metrica_{k}':v for k,v in combinado.items() if k!='modo'}})
        df_cal=pd.DataFrame(calibracoes).sort_values(['metrica_f1','metrica_precisao'],ascending=False)
        vencedor=df_cal.iloc[0]
        print(df_cal[['detector','nome_match','metrica_precisao','metrica_recall','metrica_f1']])
        ''') ,
        md(r'''
        ## 5. Congelamento e validação completa
        ''') ,
        code(r'''
        chaves_det=['pre_blur_sigma','canny_lo','canny_hi','hough_threshold','min_length_frac','max_gap']
        chaves_match=['iou','angulo','distancia','cobertura']
        DETECTOR_FINAL={k:(float(vencedor[k]) if k in ['pre_blur_sigma','min_length_frac'] else int(vencedor[k])) for k in chaves_det}
        MATCHER_FINAL={k:float(vencedor[k]) for k in chaves_match}
        congelado={'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,'detector':DETECTOR_FINAL,
                   'matcher':MATCHER_FINAL,'selecionado_somente_em':'val','amostra_val_n':len(amostra)}
        texto=json.dumps(congelado,sort_keys=True)
        congelado['sha256_config']=hashlib.sha256(texto.encode()).hexdigest()
        (HOUGH/'config_congelada_v3.json').write_text(json.dumps(congelado,indent=2),encoding='utf-8')
        df_cal.to_csv(HOUGH/'calibracao_v3.csv',index=False)

        resultados_val,resumos_val=avaliar_conjunto(df_val,DETECTOR_FINAL,MATCHER_FINAL,'Validação completa')
        resultados_val.to_csv(HOUGH/'resultado_val_v3.csv',index=False);resumos_val.to_csv(HOUGH/'resumo_val_v3.csv',index=False)
        print(json.dumps(congelado,indent=2));print(resumos_val)
        copiar_arvore_com_progresso(HOUGH, DRIVE_RUN/'hough_v3', 'Salvando calibração Hough')
        ''') ,
        md(r'''
        ## 6. Teste final protegido

        Revise a configuração congelada e a validação. Depois altere `RODAR_TESTE_FINAL` para
        `True` uma única vez. Se já existir resultado no Drive, a célula recusa nova execução.
        ''') ,
        code(r'''
        RODAR_TESTE_FINAL=False
        resultado_drive=DRIVE_RUN/'hough_v3/resultado_test_v3.csv'
        if resultado_drive.exists():
            print('Teste final já existe e permanece congelado:',resultado_drive)
        elif RODAR_TESTE_FINAL:
            df_test=df[df.split=='test'].copy()
            resultados_test,resumos_test=avaliar_conjunto(df_test,DETECTOR_FINAL,MATCHER_FINAL,'TESTE FINAL v3')
            resultados_test.to_csv(HOUGH/'resultado_test_v3.csv',index=False)
            resumos_test.to_csv(HOUGH/'resumo_test_v3.csv',index=False)
            trava={'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,'sha256_config':congelado['sha256_config'],
                   'sha256_resultado':hashlib.sha256((HOUGH/'resultado_test_v3.csv').read_bytes()).hexdigest()}
            (HOUGH/'TRAVA_TESTE_FINAL.json').write_text(json.dumps(trava,indent=2),encoding='utf-8')
            copiar_arvore_com_progresso(HOUGH, DRIVE_RUN/'hough_v3', 'Salvando teste final Hough')
            print(resumos_test)
        else:
            print('Teste não executado. Altere RODAR_TESTE_FINAL somente após congelar o protocolo.')
        ''') ,
        md(r'''
        ## 7. Detalhamento e intervalo de confiança agrupado por fundo

        Execute após o teste final existir. O bootstrap reamostra fundos, não imagens individuais.
        ''') ,
        code(r'''
        if not (HOUGH/'resultado_test_v3.csv').exists() and resultado_drive.exists():
            shutil.copy2(resultado_drive,HOUGH/'resultado_test_v3.csv')
        if (HOUGH/'resultado_test_v3.csv').exists():
            rt=pd.read_csv(HOUGH/'resultado_test_v3.csv')
            recall_subtipo=rt[rt.tipo=='positivo'].groupby('subtipo')['acertou_combinado'].mean()
            print('Recall combinado por modo de brilho:')
            print(recall_subtipo)
            print('\nFP/imagem e taxa de frames com FP:')
            neg=rt[rt.tipo=='negativo']
            fp_subtipo=neg.groupby('subtipo').agg(
                fp_por_imagem=('fp_combinado','mean'),
                taxa_frames_com_fp=('fp_combinado',lambda s:float((s>0).mean()))
            )
            print(fp_subtipo)
            print('\nMétricas geométricas reais entre matches combinados:')
            ac=rt[(rt.tipo=='positivo')&(rt.acertou_combinado)]
            geometria=ac[['angulo_match','distancia_match','cobertura_match','iou_match']].agg(['mean','median'])
            print(geometria)

            por_fundo=rt.groupby('sha256_fundo')[['tp_combinado','fp_combinado','fn_combinado']].sum()
            rng=np.random.default_rng(20260823);f1s=[];ids=por_fundo.index.to_numpy()
            for _ in tqdm(range(2000), desc='Bootstrap por fundo', unit='rep'):
                amostra_ids=rng.choice(ids,size=len(ids),replace=True);s=por_fundo.loc[amostra_ids].sum()
                p=s.tp_combinado/(s.tp_combinado+s.fp_combinado) if s.tp_combinado+s.fp_combinado else 0
                r=s.tp_combinado/(s.tp_combinado+s.fn_combinado) if s.tp_combinado+s.fn_combinado else 0
                f1s.append(2*p*r/(p+r) if p+r else 0)
            ic95=np.percentile(f1s,[2.5,97.5])
            print('IC95% F1 combinado, bootstrap por fundo:',ic95)

            trava=json.loads((HOUGH/'TRAVA_TESTE_FINAL.json').read_text(encoding='utf-8'))
            resumo_detalhado={
                'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                'sha256_config':trava['sha256_config'],'sha256_resultado':trava['sha256_resultado'],
                'recall_combinado_por_subtipo':{k:float(v) for k,v in recall_subtipo.items()},
                'negativos_por_subtipo':{
                    k:{'fp_por_imagem':float(v.fp_por_imagem),
                       'taxa_frames_com_fp':float(v.taxa_frames_com_fp)}
                    for k,v in fp_subtipo.iterrows()
                },
                'geometria_matches_combinados':{
                    coluna:{'media':float(geometria.loc['mean',coluna]),
                            'mediana':float(geometria.loc['median',coluna])}
                    for coluna in geometria.columns
                },
                'bootstrap_f1_por_fundo':{
                    'seed':20260823,'replicacoes':2000,'n_fundos':int(len(ids)),
                    'ic95_percentil':[float(ic95[0]),float(ic95[1])]
                }
            }
            resumo_path=HOUGH/'resumo_detalhado_test_v3.json'
            bootstrap_path=HOUGH/'bootstrap_f1_por_fundo_v3.csv'
            resumo_path.write_text(json.dumps(resumo_detalhado,indent=2),encoding='utf-8')
            pd.DataFrame({'f1_combinado':f1s}).to_csv(bootstrap_path,index=False)
            for arquivo in [resumo_path,bootstrap_path]:
                shutil.copy2(arquivo,DRIVE_RUN/'hough_v3'/arquivo.name)
            print('Resumo detalhado e bootstrap salvos no Drive.')
            print(json.dumps(resumo_detalhado,indent=2,ensure_ascii=False))
        ''') ,
    ]
    return write_notebook('06_baseline_hough_protocolo_congelado_v3.ipynb', cells)


def build_02():
    cells = [
        md(r'''
        # 02 — Auditoria de domínio assistida e triagem linear (v3.2)

        Esta revisão elimina a migração legada, que atribuiu motivos de Hubble/JWST a arquivos
        SDSS. A política é conservadora e reproduzível:

        - SDSS: categoria A;
        - Hubble/JWST: categoria B, exceto planetas, mosaicos e imagens inadequadas identificadas;
        - APOD: categoria C nesta versão principal;
        - alertas lineares são examinados separadamente antes da publicação.

        A auditoria visual assistida das 13 folhas de contato foi realizada em 2026-08-23.
        ''') ,
        code(r'''
        %pip install -q opencv-python-headless pandas matplotlib tqdm

        from google.colab import drive
        from pathlib import Path
        import hashlib, json, math, shutil, time
        import cv2
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        def copiar_arvore_com_progresso(origem, destino, descricao):
            origem, destino = Path(origem), Path(destino)
            arquivos = sorted(p for p in origem.rglob('*') if p.is_file())
            log_etapa(f'{descricao}: início ({len(arquivos)} arquivos)')
            inicio = time.perf_counter()
            for arquivo in tqdm(arquivos, desc=descricao, unit='arq'):
                alvo = destino / arquivo.relative_to(origem)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arquivo, alvo)
            log_etapa(f'{descricao}: concluído em {time.perf_counter()-inicio:.1f}s')

        def sha256_arquivo(caminho, bloco=1024*1024):
            h = hashlib.sha256()
            with open(caminho, 'rb') as arquivo:
                for parte in iter(lambda: arquivo.read(bloco), b''):
                    h.update(parte)
            return h.hexdigest()

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR = Path('/content/drive/MyDrive/tcc-satellite-streaks')
        PROC = Path('data/processed_v3')
        BASE = PROC/'backgrounds_base'
        AUDIT = PROC/'auditoria_v3_2'
        CURATED = PROC/'backgrounds_curated'
        for pasta in [PROC, BASE, AUDIT, CURATED]:
            pasta.mkdir(parents=True, exist_ok=True)

        origem = BACKUP_DIR/'data/processed_v3'
        shutil.copy2(origem/'manifest_base_v3.csv', PROC/'manifest_base_v3.csv')
        copiar_arvore_com_progresso(origem/'backgrounds_base', BASE, 'Restaurando 424 fundos')
        df_base = pd.read_csv(PROC/'manifest_base_v3.csv')
        print('Fundos carregados:', len(df_base))
        ''') ,
        md(r'''
        ## 1. Integridade e política de domínio

        Nenhuma decisão do manifest legado é reutilizada. As exceções Hubble abaixo foram
        identificadas diretamente nas folhas de contato por conteúdo visual.
        ''') ,
        code(r'''
        faltantes, divergentes = [], []
        for linha in tqdm(df_base.itertuples(), total=len(df_base), desc='Validando hashes', unit='arq'):
            caminho = BASE/linha.arquivo_final
            if not caminho.is_file():
                faltantes.append(linha.arquivo_final)
            elif sha256_arquivo(caminho) != linha.sha256_processado:
                divergentes.append(linha.arquivo_final)
        assert not faltantes and not divergentes, (
            f'Integridade falhou: faltantes={faltantes[:5]}, divergentes={divergentes[:5]}'
        )

        HUBBLE_C = {
            'hubble_efe23662aa7c017a',  # superfície/objeto do Sistema Solar
            'hubble_22ca90ebc60e8348',  # mosaico com área preta
            'hubble_d137a7b103e1bc4d',  # mosaico descontínuo
            'hubble_960e30d3c0a83bef',  # Júpiter
            'hubble_4daacf8341e488a9',  # Júpiter
            'hubble_710fca57f1416f44',  # Saturno
            'hubble_421da51be8540ac8',  # Saturno
            'hubble_05e8da56fca3eae9',  # Júpiter
            'hubble_d4fb185552a12482',  # Júpiter em close
            'hubble_5ac146c1ec0bd8a8',  # Saturno
            'hubble_164d343a6e51da07',  # Urano
            'hubble_6377be5d8f8fd7ed',  # Saturno
            'hubble_9b105078468898fa',  # Saturno
        }

        # Casos SDSS anteriormente marcados B por uma migração incorreta. Visualmente são frames
        # de levantamento com fonte brilhante/objeto extenso: permanecem A e viram candidatos
        # a hard negative real.
        HARD_NEGATIVE_SDSS = {
            'sdss_00d0d736439602c9', 'sdss_00ebbf43498b0b42',
            'sdss_022d8b2c356710d8', 'sdss_02441bd517ff45d2',
            'sdss_0263bb002ee8dd3e', 'sdss_03557782950f11fa',
            'sdss_03a223103031b091', 'sdss_03b60d55b03976e1',
            'sdss_06959242501478de', 'sdss_0993f922f4818e39',
            'sdss_09d154faaa90354a',
        }

        decisoes = df_base.copy()
        decisoes['categoria'] = decisoes['fonte'].map({
            'sdss':'A', 'hubble':'B', 'jwst':'B', 'apod':'C'
        })
        assert decisoes['categoria'].notna().all()
        decisoes.loc[decisoes.id_fundo.isin(HUBBLE_C), 'categoria'] = 'C'

        decisoes['motivo'] = decisoes['fonte'].map({
            'sdss':'frame de levantamento SDSS compatível com o domínio A',
            'hubble':'imagem astronômica processada; somente augmentation de treino B',
            'jwst':'imagem astronômica processada; somente augmentation de treino B',
            'apod':'APOD heterogêneo excluído conservadoramente da versão principal',
        })
        decisoes.loc[decisoes.id_fundo.isin(HUBBLE_C), 'motivo'] = (
            'Sistema Solar, planeta ou mosaico inadequado ao fundo astronômico desta versão'
        )
        decisoes['hard_negative_real'] = decisoes.id_fundo.isin(HARD_NEGATIVE_SDSS)
        decisoes['tipo_artefato'] = np.where(
            decisoes.hard_negative_real, 'fonte_brilhante_ou_objeto_extenso', ''
        )
        decisoes['metodo_decisao'] = 'regra_de_fonte+auditoria_visual_assistida_2026-08-23'

        print(pd.crosstab(decisoes.fonte, decisoes.categoria))
        print('Hard negatives reais pré-identificados:', int(decisoes.hard_negative_real.sum()))
        ''') ,
        md(r'''
        ## 2. Triagem automática de artefatos lineares

        O detector não decide sozinho se uma linha é satélite. Ele apenas reduz a revisão final
        aos fundos A/B que contêm segmentos longos ou anomalias fortes.
        ''') ,
        code(r'''
        def diagnosticar_linhas(caminho):
            img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (512,512):
                raise ValueError(f'Imagem inválida: {caminho}')
            suavizada = cv2.GaussianBlur(img, (0,0), 1.2)
            bordas = cv2.Canny(suavizada, 60, 160)
            diagonal = math.hypot(*img.shape)
            linhas = cv2.HoughLinesP(
                bordas, 1, np.pi/180, threshold=65,
                minLineLength=int(0.30*diagonal), maxLineGap=12
            )
            # OpenCV pode devolver (N,1,4) ou (N,4), dependendo da versão.
            segmentos = np.empty((0,4), dtype=np.float32) if linhas is None else np.asarray(linhas).reshape(-1,4)
            comprimentos = [
                math.hypot(int(x2)-int(x1), int(y2)-int(y1))
                for x1, y1, x2, y2 in segmentos
            ]
            max_frac = max(comprimentos, default=0.0)/diagonal
            mediana = float(np.median(img))
            mad = float(1.4826*np.median(np.abs(img.astype(np.float32)-mediana)))
            return {
                'n_linhas_longas': len(comprimentos),
                'comprimento_max_frac': max_frac,
                'desvio': float(img.std()),
                'mad_robusto': mad,
                'fracao_saturada': float((img>=250).mean()),
            }

        diagnosticos = []
        for linha in tqdm(decisoes.itertuples(), total=len(decisoes), desc='Triagem linear', unit='img'):
            diagnosticos.append(diagnosticar_linhas(BASE/linha.arquivo_final))
        diag = pd.DataFrame(diagnosticos)
        for coluna in diag.columns:
            decisoes[coluna] = diag[coluna].values

        aceito = decisoes.categoria.isin(['A','B'])
        dominio_principal = decisoes.categoria.eq('A')
        # B já foi integralmente inspecionado nas folhas; a triagem automática adicional é
        # concentrada no domínio A, que alimentará validação e teste.
        decisoes['alerta_linear_automatico'] = dominio_principal & (
            (decisoes.n_linhas_longas > 0) |
            (decisoes.desvio < 3.0) |
            (decisoes.fracao_saturada > 0.20)
        )
        decisoes['revisao_final_requerida'] = decisoes.alerta_linear_automatico
        decisoes['sem_trilha_confirmado'] = aceito & ~decisoes.revisao_final_requerida

        revisar = decisoes[decisoes.revisao_final_requerida].copy()
        print('Imagens A/B com alerta para revisão final:', len(revisar))
        print(revisar.groupby(['fonte','hard_negative_real']).size())

        # Folhas somente com alertas, mostradas no próprio Colab.
        por_pagina = 32
        for pagina, inicio in enumerate(range(0, len(revisar), por_pagina), start=1):
            lote = revisar.iloc[inicio:inicio+por_pagina]
            fig, axes = plt.subplots(4, 8, figsize=(20, 11))
            axes = axes.flatten()
            for ax, linha in zip(axes, lote.itertuples()):
                img = cv2.imread(str(BASE/linha.arquivo_final), cv2.IMREAD_GRAYSCALE)
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                ax.set_title(
                    f'{linha.id_fundo}\n{linha.categoria} linhas={linha.n_linhas_longas}',
                    fontsize=6
                )
                ax.axis('off')
            for ax in axes[len(lote):]:
                ax.axis('off')
            plt.tight_layout()
            destino = AUDIT/f'alertas_lineares_{pagina:02d}.png'
            plt.savefig(destino, dpi=160, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        caminho_audit = AUDIT/'auditoria_decisoes_v3_2.csv'
        decisoes.to_csv(caminho_audit, index=False)
        destino_audit = BACKUP_DIR/'data/processed_v3/auditoria_v3_2'
        copiar_arvore_com_progresso(AUDIT, destino_audit, 'Salvando evidências da auditoria')
        print('Envie somente as folhas alertas_lineares_*.png se houver alertas.')
        ''') ,
        md(r'''
        ## 3. Confirmação dirigida e gate local

        Depois da revisão das folhas de alerta, informe somente os IDs realmente contaminados.
        Os demais alertas podem ser confirmados em lote como picos de difração, galáxias ou
        falsos alertas do Hough.
        ''') ,
        code(r'''
        IDS_COM_TRILHA_OU_CONTAMINACAO = set()
        # Exemplo, se necessário: IDS_COM_TRILHA_OU_CONTAMINACAO.add('sdss_...')

        IDS_HARD_NEGATIVE_REAL_ADICIONAL = set()
        # Exemplo, se necessário: IDS_HARD_NEGATIVE_REAL_ADICIONAL.add('sdss_...')
        CONFIRMAR_ALERTAS_RESTANTES_SEM_TRILHA = False

        ids_validos = set(decisoes.id_fundo)
        assert IDS_COM_TRILHA_OU_CONTAMINACAO <= ids_validos
        assert IDS_HARD_NEGATIVE_REAL_ADICIONAL <= ids_validos

        decisoes_final = decisoes.copy()
        mascara_c = decisoes_final.id_fundo.isin(IDS_COM_TRILHA_OU_CONTAMINACAO)
        decisoes_final.loc[mascara_c, ['categoria','motivo']] = [
            'C', 'trilha preexistente ou contaminação confirmada na revisão dirigida'
        ]
        mascara_hard = decisoes_final.id_fundo.isin(IDS_HARD_NEGATIVE_REAL_ADICIONAL)
        decisoes_final.loc[mascara_hard, 'hard_negative_real'] = True
        decisoes_final.loc[mascara_hard, 'tipo_artefato'] = 'artefato_linear_real_nao_satelite'

        if CONFIRMAR_ALERTAS_RESTANTES_SEM_TRILHA:
            restantes = decisoes_final.revisao_final_requerida & ~mascara_c
            decisoes_final.loc[restantes, 'sem_trilha_confirmado'] = True
            decisoes_final.loc[restantes, 'metodo_decisao'] += '+revisao_dirigida_confirmada'

        aceitos = decisoes_final[decisoes_final.categoria.isin(['A','B'])].copy()
        pendentes = aceitos[~aceitos.sem_trilha_confirmado]
        assert pendentes.empty, (
            f'{len(pendentes)} alertas ainda não confirmados. Revise as folhas e altere '
            'CONFIRMAR_ALERTAS_RESTANTES_SEM_TRILHA=True.'
        )
        assert decisoes_final.sha256_raw.is_unique
        assert decisoes_final.sha256_processado.is_unique
        assert set(decisoes_final.categoria) <= {'A','B','C'}

        nomes_aceitos = set(aceitos.arquivo_final)
        for caminho in CURATED.glob('*'):
            if caminho.is_file() and caminho.name not in nomes_aceitos:
                caminho.unlink()
        for nome in tqdm(sorted(nomes_aceitos), desc='Montando curadoria local', unit='img'):
            origem_arq, destino_arq = BASE/nome, CURATED/nome
            if not destino_arq.exists() or sha256_arquivo(destino_arq) != sha256_arquivo(origem_arq):
                shutil.copy2(origem_arq, destino_arq)

        decisoes_final.to_csv(PROC/'manifest_curado_v3.csv', index=False)
        assert set(p.name for p in CURATED.glob('*.png')) == nomes_aceitos

        resumo_local = {
            'status':'APROVADO_LOCAL_PARA_PUBLICACAO',
            'total':len(decisoes_final),
            'A':int((decisoes_final.categoria=='A').sum()),
            'B':int((decisoes_final.categoria=='B').sum()),
            'C':int((decisoes_final.categoria=='C').sum()),
            'hard_negative_real':int(decisoes_final.hard_negative_real.sum()),
            'alertas_revisados':int(decisoes_final.revisao_final_requerida.sum()),
            'aceitos':len(aceitos),
        }
        (AUDIT/'resumo_auditoria_v3_2.json').write_text(
            json.dumps(resumo_local, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(json.dumps(resumo_local, indent=2, ensure_ascii=False))
        print('Gate local aprovado. Ainda não houve publicação da pasta curada.')
        ''') ,
        md(r'''
        ## 4. Publicação e verificação pós-backup

        Execute somente depois de conferir o resumo local. Altere a chave abaixo para `True`.
        ''') ,
        code(r'''
        PUBLICAR_CURADORIA_NO_DRIVE = False

        if not PUBLICAR_CURADORIA_NO_DRIVE:
            print('Publicação bloqueada. Valide o resumo local antes de alterar para True.')
        else:
            destino_proc = BACKUP_DIR/'data/processed_v3'
            destino_curado = destino_proc/'backgrounds_curated'
            destino_curado.mkdir(parents=True, exist_ok=True)
            nomes_aceitos = set(aceitos.arquivo_final)

            remotos = [p for p in destino_curado.glob('*.png') if p.name not in nomes_aceitos]
            for p in tqdm(remotos, desc='Removendo órfãos remotos', unit='img'):
                p.unlink()
            for nome in tqdm(sorted(nomes_aceitos), desc='Publicando curadoria', unit='img'):
                shutil.copy2(CURATED/nome, destino_curado/nome)
            shutil.copy2(PROC/'manifest_curado_v3.csv', destino_proc/'manifest_curado_v3.csv')
            copiar_arvore_com_progresso(AUDIT, destino_proc/'auditoria_v3_2', 'Publicando auditoria')

            encontrados = set(p.name for p in destino_curado.glob('*.png'))
            assert encontrados == nomes_aceitos
            divergentes = []
            mapa_hash = aceitos.set_index('arquivo_final').sha256_processado.to_dict()
            for nome in tqdm(sorted(nomes_aceitos), desc='Verificando hashes no Drive', unit='img'):
                if sha256_arquivo(destino_curado/nome) != mapa_hash[nome]:
                    divergentes.append(nome)
            assert not divergentes, f'Hashes divergentes no Drive: {divergentes[:10]}'
            print(json.dumps({
                'status':'BACKUP_CURADORIA_APROVADO',
                'arquivos_esperados':len(nomes_aceitos),
                'arquivos_encontrados':len(encontrados),
                'hashes_divergentes':len(divergentes),
            }, indent=2, ensure_ascii=False))
        ''') ,
    ]
    return write_notebook('02_sync_e_auditoria_dominio_ABC_v3.ipynb', cells)


def build_03():
    cells = [
        md(r'''
        # 03 — Split agrupado dos fundos (v3.2)

        O split é feito por fundo e hash. Com 250 fundos A do SDSS, a proporção 68/16/16
        preserva 40 fundos independentes tanto na validação quanto no teste. Categoria B é
        exclusiva do treino e categoria C é excluída.
        ''') ,
        code(r'''
        from google.colab import drive
        from pathlib import Path
        import hashlib, json, shutil, time
        import numpy as np
        import pandas as pd

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        drive.mount('/content/drive', force_remount=True)
        BACKUP_DIR = Path('/content/drive/MyDrive/tcc-satellite-streaks')
        PROC = Path('data/processed_v3'); PROC.mkdir(parents=True, exist_ok=True)
        origem = BACKUP_DIR/'data/processed_v3/manifest_curado_v3.csv'
        shutil.copy2(origem, PROC/'manifest_curado_v3.csv')
        df = pd.read_csv(PROC/'manifest_curado_v3.csv')

        SEED_SPLIT = 42
        PROPORCOES = {'train':0.68, 'val':0.16, 'test':0.16}
        MIN_A_VAL_TEST = 40
        ''') ,
        md(r'''
        ## 1. Divisão determinística por fonte e hash
        ''') ,
        code(r'''
        assert set(df.categoria) <= {'A','B','C'}
        assert df.sha256_processado.is_unique
        df_a = df[df.categoria=='A'].copy()
        df_b = df[df.categoria=='B'].copy()
        assert len(df_a) > 0

        def alocar_maior_resto(total, proporcoes):
            bruto = {k:total*v for k,v in proporcoes.items()}
            base = {k:int(np.floor(v)) for k,v in bruto.items()}
            faltam = total-sum(base.values())
            ordem = sorted(proporcoes, key=lambda k:(bruto[k]-base[k],k), reverse=True)
            for k in ordem[:faltam]: base[k] += 1
            return base

        partes=[]
        for fonte, grupo in df_a.groupby('fonte', sort=True):
            grupo=grupo.sort_values('sha256_processado').copy()
            seed=SEED_SPLIT+int(hashlib.sha256(fonte.encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed)
            grupo=grupo.iloc[rng.permutation(len(grupo))].reset_index(drop=True)
            cont=alocar_maior_resto(len(grupo),PROPORCOES)
            grupo['split']=(['train']*cont['train']+['val']*cont['val']+['test']*cont['test'])
            partes.append(grupo)
        df_a_split=pd.concat(partes,ignore_index=True)
        df_b['split']='train'
        df_split=pd.concat([df_a_split,df_b],ignore_index=True)
        ''') ,
        md(r'''
        ## 2. Gate local, publicação e verificação
        ''') ,
        code(r'''
        assert df_split.id_fundo.is_unique
        assert df_split.groupby('sha256_processado').split.nunique().max()==1
        assert not ((df_split.categoria=='B')&(df_split.split!='train')).any()
        assert not df_split.categoria.eq('C').any()

        n_val=int((df_a_split.split=='val').sum())
        n_test=int((df_a_split.split=='test').sum())
        assert n_val>=MIN_A_VAL_TEST and n_test>=MIN_A_VAL_TEST, (
            f'Diversidade insuficiente: A-val={n_val}, A-test={n_test}'
        )
        print(pd.crosstab([df_split.split,df_split.categoria],df_split.fonte))

        saida=PROC/'manifest_split_fundos_v3.csv'
        df_split.sort_values(['split','categoria','fonte','id_fundo']).to_csv(saida,index=False)
        hash_manifest=hashlib.sha256(saida.read_bytes()).hexdigest()
        contrato={
            'versao':'v3.2','seed_split':SEED_SPLIT,'proporcoes':PROPORCOES,
            'sha256_manifest_split':hash_manifest,
            'contagens':df_split.split.value_counts().to_dict(),
            'contagens_A':df_a_split.split.value_counts().to_dict(),
        }
        contrato_path=PROC/'contrato_split_v3.json'
        contrato_path.write_text(json.dumps(contrato,indent=2),encoding='utf-8')
        resumo={
            'status':'SPLIT_LOCAL_APROVADO','hash_manifest':hash_manifest,
            'A_val':n_val,'A_test':n_test,
            'fundos_por_split':df_split.split.value_counts().to_dict(),
        }
        print(json.dumps(resumo,indent=2,ensure_ascii=False))

        PUBLICAR_SPLIT_NO_DRIVE=False
        if not PUBLICAR_SPLIT_NO_DRIVE:
            print('Publicação bloqueada. Confira o resumo e altere a chave para True.')
        else:
            destino=BACKUP_DIR/'data/processed_v3';destino.mkdir(parents=True,exist_ok=True)
            log_etapa('Publicando manifest e contrato do split')
            shutil.copy2(saida,destino/saida.name)
            shutil.copy2(contrato_path,destino/contrato_path.name)
            assert hashlib.sha256((destino/saida.name).read_bytes()).hexdigest()==hash_manifest
            assert hashlib.sha256((destino/contrato_path.name).read_bytes()).hexdigest()==hashlib.sha256(contrato_path.read_bytes()).hexdigest()
            print(json.dumps({'status':'BACKUP_SPLIT_APROVADO','hash_manifest':hash_manifest},indent=2))
        ''') ,
    ]
    return write_notebook('03_split_agrupado_dos_fundos_v3.ipynb', cells)


def build_07():
    cells = [
        md(r'''
        # 07 — Pacote íntegro do dataset para runtimes GPU (v3)

        Este notebook evita uma restauração intermediária de mais de 12 mil arquivos individuais
        do Google Drive. Ele deve ser executado preferencialmente no mesmo runtime em que o
        Notebook 06 terminou, enquanto o RUN local ainda existe. Se esse runtime já tiver sido
        perdido, há um fallback explícito que lê o RUN aprovado diretamente do Drive e cria o TAR
        local em uma única passagem. O dataset é reunido em um TAR imutável, acompanhado de
        contrato e SHA-256.

        Este notebook não treina modelos e pode ser executado em CPU.
        ''') ,
        code(r'''
        from google.colab import drive
        from pathlib import Path
        import hashlib, json, os, shutil, tarfile, time
        import pandas as pd
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}", flush=True)

        def sha256_com_progresso(caminho, descricao, bloco=8*1024*1024):
            caminho=Path(caminho); total=caminho.stat().st_size; h=hashlib.sha256()
            with caminho.open('rb') as arq, tqdm(total=total,desc=descricao,unit='B',unit_scale=True) as barra:
                for parte in iter(lambda:arq.read(bloco),b''):
                    h.update(parte);barra.update(len(parte))
            return h.hexdigest()

        def copiar_arquivo_com_progresso(origem,destino,descricao,bloco=8*1024*1024):
            origem,destino=Path(origem),Path(destino);destino.parent.mkdir(parents=True,exist_ok=True)
            total=origem.stat().st_size
            with origem.open('rb') as entrada,destino.open('wb') as saida, \
                 tqdm(total=total,desc=descricao,unit='B',unit_scale=True) as barra:
                for parte in iter(lambda:entrada.read(bloco),b''):
                    saida.write(parte);barra.update(len(parte))
            shutil.copystat(origem,destino)

        def metadado_tar_deterministico(info):
            info.uid=0;info.gid=0;info.uname='';info.gname='';info.mtime=0
            info.mode=0o644
            return info

        drive.mount('/content/drive',force_remount=True)
        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        RUN_ID=(BACKUP_DIR/'data/synthetic_runs/LATEST_RUN_V3.txt').read_text(encoding='utf-8').strip()
        RUN=Path('/content/data/synthetic_runs')/RUN_ID
        DRIVE_RUN=BACKUP_DIR/'data/synthetic_runs'/RUN_ID
        PACOTES_DRIVE=BACKUP_DIR/'data/yolo_packages'/RUN_ID
        PACOTE_LOCAL=Path('/content')/f'{RUN_ID}_yolo_dataset_v3.tar'
        CONTRATO_LOCAL=Path('/content')/f'{RUN_ID}_contrato_pacote_yolo_v3.json'
        print('RUN esperado:',RUN_ID)
        print('RUN local:',RUN)
        ''') ,
        md(r'''
        ## 1. Gate local antes do empacotamento

        O caminho preferencial usa o RUN que já foi restaurado pelo Notebook 06. Se aquela sessão
        ainda estiver ativa, mantenha `EMPACOTAR_DIRETO_DO_DRIVE=False`.

        Se a sessão anterior tiver sido perdida, altere **somente**
        `EMPACOTAR_DIRETO_DO_DRIVE=True`. Nesse modo o TAR é criado diretamente do RUN aprovado
        no Drive: os arquivos não são restaurados individualmente para `/content` antes do
        empacotamento. A leitura remota continuará demorada, mas ocorrerá uma única vez.
        ''') ,
        code(r'''
        EMPACOTAR_DIRETO_DO_DRIVE=False

        if RUN.is_dir():
            RUN_FONTE=RUN
            MODO_FONTE='run_local_do_runtime'
        elif EMPACOTAR_DIRETO_DO_DRIVE:
            if not DRIVE_RUN.is_dir():
                raise RuntimeError(f'RUN aprovado também está ausente no Drive: {DRIVE_RUN}')
            RUN_FONTE=DRIVE_RUN
            MODO_FONTE='run_aprovado_direto_do_drive'
            print(
                'ATENÇÃO: o runtime local foi perdido. O TAR será criado diretamente do Drive, '
                'sem restauração intermediária. Esta etapa pode demorar bastante.'
            )
        else:
            raise RuntimeError(
                'RUN local ausente. Se a sessão do Notebook 06 ainda estiver ativa, execute o '
                'Notebook 07 nela. Se aquela sessão foi perdida, altere apenas '
                'EMPACOTAR_DIRETO_DO_DRIVE=True e execute novamente esta seção.'
            )

        obrigatorios=['manifest_geracao_v3.csv','config_geracao_v3.json','run_id.txt','resumo_validacao_v3.json']
        for nome in obrigatorios:
            assert (RUN_FONTE/nome).is_file(),f'Ausente: {nome}'
        df=pd.read_csv(RUN_FONTE/'manifest_geracao_v3.csv')
        HASH_MANIFEST=hashlib.sha256((RUN_FONTE/'manifest_geracao_v3.csv').read_bytes()).hexdigest()
        validacao=json.loads((RUN_FONTE/'resumo_validacao_v3.json').read_text(encoding='utf-8'))
        assert len(df)==6000
        assert validacao.get('aprovado_para_treino') is True
        assert validacao.get('sha256_manifest')==HASH_MANIFEST

        arquivos=[]
        for nome in obrigatorios: arquivos.append(RUN_FONTE/nome)
        for split in ['train','val','test']:
            esperado=set(df.loc[df.split==split,'id'])
            imagens={p.stem for p in (RUN_FONTE/'images'/split).glob('*.png')}
            labels={p.stem for p in (RUN_FONTE/'labels'/split).glob('*.txt')}
            assert imagens==esperado,f'Conjunto de imagens divergente em {split}'
            assert labels==esperado,f'Conjunto de labels divergente em {split}'
            arquivos.extend(sorted((RUN_FONTE/'images'/split).glob('*.png')))
            arquivos.extend(sorted((RUN_FONTE/'labels'/split).glob('*.txt')))
        assert len(arquivos)==12004
        print(json.dumps({
            'status':'RUN_FONTE_APROVADO_PARA_EMPACOTAMENTO','run_id':RUN_ID,
            'modo_fonte':MODO_FONTE,
            'sha256_manifest':HASH_MANIFEST,'amostras':len(df),'arquivos_no_pacote':len(arquivos)
        },indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 2. Construção local do TAR imutável

        PNG e TAR já são formatos adequados para transferência; não usamos gzip porque ele
        acrescentaria processamento com pouco ganho sobre PNGs já comprimidos.
        ''') ,
        code(r'''
        RECONSTRUIR_PACOTE_LOCAL=False
        if PACOTE_LOCAL.exists() and not RECONSTRUIR_PACOTE_LOCAL:
            if not CONTRATO_LOCAL.exists():
                raise RuntimeError('Pacote local existe sem contrato. Revise o alvo antes de reconstruir.')
            contrato=json.loads(CONTRATO_LOCAL.read_text(encoding='utf-8'))
            assert contrato['run_id']==RUN_ID and contrato['sha256_manifest']==HASH_MANIFEST
            hash_atual=sha256_com_progresso(PACOTE_LOCAL,'Verificando TAR local existente')
            assert hash_atual==contrato['sha256_pacote'],'TAR local diverge do contrato.'
            print('Pacote local válido reutilizado.')
        else:
            temporario=PACOTE_LOCAL.with_suffix('.tar.partial')
            if temporario.exists(): temporario.unlink()
            log_etapa(f'Criando TAR local com {len(arquivos)} arquivos')
            inicio=time.perf_counter()
            with tarfile.open(temporario,'w') as tar:
                for arquivo in tqdm(arquivos,desc='Empacotando dataset',unit='arq'):
                    relativo=arquivo.relative_to(RUN_FONTE)
                    tar.add(
                        arquivo,arcname=(Path(RUN_ID)/relativo).as_posix(),recursive=False,
                        filter=metadado_tar_deterministico
                    )
            temporario.replace(PACOTE_LOCAL)
            hash_pacote=sha256_com_progresso(PACOTE_LOCAL,'Calculando SHA-256 do TAR')
            base_contrato={
                'versao':'yolo_dataset_package_v3','run_id':RUN_ID,
                'sha256_manifest':HASH_MANIFEST,'arquivo':PACOTE_LOCAL.name,
                'sha256_pacote':hash_pacote,'tamanho_bytes':PACOTE_LOCAL.stat().st_size,
                'membros':len(arquivos),'formato':'tar_sem_compressao'
            }
            base_contrato['sha256_contrato']=hashlib.sha256(
                json.dumps(base_contrato,sort_keys=True,separators=(',',':')).encode()
            ).hexdigest()
            CONTRATO_LOCAL.write_text(json.dumps(base_contrato,indent=2),encoding='utf-8')
            contrato=base_contrato
            log_etapa(f'TAR concluído em {time.perf_counter()-inicio:.1f}s')
        print(json.dumps(contrato,indent=2,ensure_ascii=False))
        print('Publicação ainda bloqueada; execute primeiro o gate local abaixo.')
        ''') ,
        md(r'''
        ## 3. Gate local do pacote
        ''') ,
        code(r'''
        contrato=json.loads(CONTRATO_LOCAL.read_text(encoding='utf-8'))
        assert contrato['run_id']==RUN_ID
        assert contrato['sha256_manifest']==HASH_MANIFEST
        assert contrato['membros']==12004
        assert PACOTE_LOCAL.stat().st_size==contrato['tamanho_bytes']
        with tarfile.open(PACOTE_LOCAL,'r') as tar:
            membros=tar.getmembers()
        assert len(membros)==contrato['membros']
        assert all(not m.issym() and not m.islnk() for m in membros)
        assert all(Path(m.name).parts[0]==RUN_ID for m in membros)

        hashes_esperados={}
        for linha in df.itertuples():
            hashes_esperados[f'{RUN_ID}/images/{linha.split}/{linha.id}.png']=linha.sha256_imagem
            hashes_esperados[f'{RUN_ID}/labels/{linha.split}/{linha.id}.txt']=linha.sha256_label
        assert len(hashes_esperados)==12000

        problemas_hash=[];hashes_verificados=0
        with tarfile.open(PACOTE_LOCAL,'r') as tar:
            membros_tar=tar.getmembers()
            nomes_tar={m.name for m in membros_tar}
            assert set(hashes_esperados)<=nomes_tar,'Arquivos do manifest ausentes no TAR.'
            alvos=[m for m in membros_tar if m.name in hashes_esperados]
            assert len(alvos)==12000
            for membro in tqdm(alvos,desc='Verificando conteúdo do TAR',unit='arq'):
                nome=membro.name;esperado=hashes_esperados[nome]
                extraido=tar.extractfile(membro)
                if extraido is None:
                    problemas_hash.append(f'{nome}: não é arquivo regular')
                    continue
                h=hashlib.sha256()
                for bloco in iter(lambda:extraido.read(8*1024*1024),b''):
                    h.update(bloco)
                hashes_verificados+=1
                if h.hexdigest()!=esperado:
                    problemas_hash.append(f'{nome}: SHA-256 divergente do manifest')
        assert not problemas_hash,problemas_hash[:20]
        assert hashes_verificados==12000
        print(json.dumps({
            'status':'PACOTE_LOCAL_APROVADO_PARA_PUBLICACAO',
            'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
            'sha256_pacote':contrato['sha256_pacote'],
            'tamanho_bytes':contrato['tamanho_bytes'],'membros':len(membros),
            'hashes_de_conteudo_verificados':hashes_verificados
        },indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 4. Publicação e verificação no Drive

        Altere a chave somente após conferir o gate local. A publicação usa um arquivo parcial
        e só o renomeia depois da cópia completa.
        ''') ,
        code(r'''
        PUBLICAR_PACOTE_NO_DRIVE=False
        if not PUBLICAR_PACOTE_NO_DRIVE:
            print('Publicação bloqueada. Confira o gate e altere a chave para True.')
        else:
            PACOTES_DRIVE.mkdir(parents=True,exist_ok=True)
            destino=PACOTES_DRIVE/PACOTE_LOCAL.name
            destino_parcial=destino.with_suffix(destino.suffix+'.partial')
            contrato_destino=PACOTES_DRIVE/'contrato_pacote_yolo_v3.json'
            if destino.exists():
                hash_remoto=sha256_com_progresso(destino,'Verificando TAR já existente no Drive')
                assert hash_remoto==contrato['sha256_pacote'],(
                    'Já existe pacote remoto divergente. Não será sobrescrito automaticamente.'
                )
                print('TAR remoto idêntico reutilizado.')
            else:
                if destino_parcial.exists(): destino_parcial.unlink()
                copiar_arquivo_com_progresso(PACOTE_LOCAL,destino_parcial,'Publicando TAR no Drive')
                hash_parcial=sha256_com_progresso(destino_parcial,'Verificando TAR publicado')
                assert hash_parcial==contrato['sha256_pacote']
                destino_parcial.replace(destino)
            shutil.copy2(CONTRATO_LOCAL,contrato_destino)
            hash_final=sha256_com_progresso(destino,'Verificação final do TAR no Drive')
            assert hash_final==contrato['sha256_pacote']
            contrato_remoto=json.loads(contrato_destino.read_text(encoding='utf-8'))
            assert contrato_remoto==contrato
            print(json.dumps({
                'status':'BACKUP_PACOTE_YOLO_APROVADO','run_id':RUN_ID,
                'sha256_pacote':hash_final,'tamanho_bytes':destino.stat().st_size,
                'membros':contrato['membros']
            },indent=2,ensure_ascii=False))
        ''') ,
    ]
    return write_notebook('07_pacote_dataset_para_gpu_v3.ipynb', cells)


def build_08():
    cells = [
        md(r'''
        # 08 — Seleção controlada YOLOv8n-OBB × YOLO11n-OBB (v3)

        Este notebook escolhe a família YOLO usando **somente treino e validação**. Teste
        sintético e ASTA não são lidos. Os dois modelos usam o mesmo protocolo, a mesma seed e
        os mesmos dados. A recomendação local precisa ser revisada antes de ser publicada.

        Execute em runtime GPU. A primeira célula interrompe imediatamente se CUDA não estiver
        disponível, antes de montar o Drive ou restaurar o dataset.
        ''') ,
        code(r'''
        import json, platform, sys, torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                'GPU obrigatória. No Colab, selecione Ambiente de execução > Alterar tipo de '
                'ambiente > GPU e execute novamente a partir desta célula.'
            )
        props=torch.cuda.get_device_properties(0)
        livre,total=torch.cuda.mem_get_info(0)
        relatorio_gpu={
            'status':'GPU_APROVADA','gpu':torch.cuda.get_device_name(0),
            'cuda':torch.version.cuda,'torch':torch.__version__,
            'vram_total_gb':round(total/1024**3,2),'vram_livre_gb':round(livre/1024**3,2),
            'python':sys.version.split()[0],'plataforma':platform.platform()
        }
        assert total>=8*1024**3,'GPU com menos de 8 GiB não foi aprovada para este protocolo.'
        print(json.dumps(relatorio_gpu,indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 1. Ambiente fixado e restauração pelo TAR

        O pacote contém somente o dataset aprovado e seus contratos. A restauração transfere
        um único arquivo do Drive, verifica SHA-256 e extrai localmente.
        ''') ,
        code(r'''
        %pip install -q ultralytics==8.4.127 opencv-python-headless pandas pyyaml tqdm

        from google.colab import drive
        from pathlib import Path
        import hashlib, json, math, os, shutil, tarfile, tempfile, time
        import cv2
        import numpy as np
        import pandas as pd
        import yaml
        from tqdm.auto import tqdm

        def log_etapa(mensagem):
            print(f"[{time.strftime('%H:%M:%S')}] {mensagem}",flush=True)

        def sha256_arquivo(caminho,bloco=8*1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as arq:
                for parte in iter(lambda:arq.read(bloco),b''):h.update(parte)
            return h.hexdigest()

        def sha256_com_progresso(caminho,descricao,bloco=8*1024*1024):
            caminho=Path(caminho);total=caminho.stat().st_size;h=hashlib.sha256()
            with caminho.open('rb') as arq,tqdm(total=total,desc=descricao,unit='B',unit_scale=True) as barra:
                for parte in iter(lambda:arq.read(bloco),b''):
                    h.update(parte);barra.update(len(parte))
            return h.hexdigest()

        def copiar_arquivo_com_progresso(origem,destino,descricao,bloco=8*1024*1024):
            origem,destino=Path(origem),Path(destino);destino.parent.mkdir(parents=True,exist_ok=True)
            total=origem.stat().st_size
            with origem.open('rb') as entrada,destino.open('wb') as saida, \
                 tqdm(total=total,desc=descricao,unit='B',unit_scale=True) as barra:
                for parte in iter(lambda:entrada.read(bloco),b''):
                    saida.write(parte);barra.update(len(parte))

        def copiar_arvore_com_progresso(origem,destino,descricao):
            origem,destino=Path(origem),Path(destino)
            arquivos=sorted(p for p in origem.rglob('*') if p.is_file())
            for arquivo in tqdm(arquivos,desc=descricao,unit='arq'):
                alvo=destino/arquivo.relative_to(origem);alvo.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(arquivo,alvo)

        drive.mount('/content/drive',force_remount=True)
        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        RUN_ID=(BACKUP_DIR/'data/synthetic_runs/LATEST_RUN_V3.txt').read_text(encoding='utf-8').strip()
        PACOTES_DRIVE=BACKUP_DIR/'data/yolo_packages'/RUN_ID
        contrato_pacote=json.loads((PACOTES_DRIVE/'contrato_pacote_yolo_v3.json').read_text(encoding='utf-8'))
        assert contrato_pacote['run_id']==RUN_ID
        pacote_remoto=PACOTES_DRIVE/contrato_pacote['arquivo']
        pacote_local=Path('/content')/contrato_pacote['arquivo']

        if not pacote_local.exists():
            copiar_arquivo_com_progresso(pacote_remoto,pacote_local,'Restaurando TAR do dataset')
        hash_pacote=sha256_com_progresso(pacote_local,'Verificando TAR local')
        assert hash_pacote==contrato_pacote['sha256_pacote']

        raiz_runs=Path('/content/data/synthetic_runs');raiz_runs.mkdir(parents=True,exist_ok=True)
        RUN=raiz_runs/RUN_ID
        if RUN.exists():
            assert (RUN/'manifest_geracao_v3.csv').is_file()
            assert sha256_arquivo(RUN/'manifest_geracao_v3.csv')==contrato_pacote['sha256_manifest']
            print('RUN local íntegro reutilizado.')
        else:
            temp=Path(tempfile.mkdtemp(prefix='extracao_yolo_',dir='/content'))
            try:
                with tarfile.open(pacote_local,'r') as tar:
                    membros=tar.getmembers()
                    assert len(membros)==contrato_pacote['membros']
                    for membro in membros:
                        partes=Path(membro.name).parts
                        assert partes and partes[0]==RUN_ID and '..' not in partes
                        assert not membro.issym() and not membro.islnk()
                    for membro in tqdm(membros,desc='Extraindo dataset local',unit='arq'):
                        tar.extract(membro,path=temp,filter='data')
                shutil.move(str(temp/RUN_ID),str(RUN))
            finally:
                if temp.exists():shutil.rmtree(temp)

        df=pd.read_csv(RUN/'manifest_geracao_v3.csv')
        HASH_MANIFEST=sha256_arquivo(RUN/'manifest_geracao_v3.csv')
        validacao=json.loads((RUN/'resumo_validacao_v3.json').read_text(encoding='utf-8'))
        assert len(df)==6000 and HASH_MANIFEST==contrato_pacote['sha256_manifest']
        assert validacao.get('aprovado_para_treino') is True
        for split in ['train','val','test']:
            esperado=set(df.loc[df.split==split,'id'])
            assert {p.stem for p in (RUN/'images'/split).glob('*.png')}==esperado
            assert {p.stem for p in (RUN/'labels'/split).glob('*.txt')}==esperado

        dataset_yaml=RUN/'dataset_yolo_selection_v3.yaml'
        dataset_yaml.write_text(yaml.safe_dump({
            'path':str(RUN.resolve()),'train':'images/train','val':'images/val','test':'images/test',
            'names':{0:'satellite_streak'}
        },sort_keys=False),encoding='utf-8')
        print(json.dumps({
            'status':'DATASET_GPU_LOCAL_APROVADO','run_id':RUN_ID,
            'sha256_manifest':HASH_MANIFEST,'amostras':len(df),
            'pacote_sha256':hash_pacote
        },indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 2. Contrato pré-registrado da seleção

        A seleção usa uma seed e somente validação. O vencedor será treinado depois em mais duas
        seeds. Teste sintético e ASTA permanecem proibidos nesta etapa.
        ''') ,
        code(r'''
        DRIVE_EXP=BACKUP_DIR/'experiments/yolo_family_selection_v3'/RUN_ID
        DRIVE_EXP.mkdir(parents=True,exist_ok=True)
        CONFIG_SELECTION={
            'protocolo':'yolo_family_selection_v1','run_id':RUN_ID,
            'sha256_manifest':HASH_MANIFEST,'ultralytics':'8.4.127',
            'modelos':{'yolov8n_obb':'yolov8n-obb.pt','yolo11n_obb':'yolo11n-obb.pt'},
            'seed_selecao':20260823,'seeds_finais':[20260823,20260824,20260825],
            'treino':{
                'epochs':100,'patience':20,'imgsz':512,'batch':16,'optimizer':'AdamW',
                'lr0':0.001,'lrf':0.01,'cos_lr':True,'weight_decay':0.0005,
                'warmup_epochs':3.0,'deterministic':True,'amp':True,'workers':2,
                'mosaic':0.0,'mixup':0.0,'copy_paste':0.0,
                'hsv_h':0.0,'hsv_s':0.0,'hsv_v':0.0,'degrees':0.0,
                'translate':0.0,'scale':0.0,'shear':0.0,'perspective':0.0,
                'flipud':0.0,'fliplr':0.0
            },
            'predicao_val':{'conf_min':0.001,'iou_nms':0.7,'max_det':100},
            'thresholds_confianca':[.01,.02,.03,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95],
            'matching':{'iou':0.30,'angulo':15.0,'distancia':14.0,'cobertura':0.50,'modo':'combinado'},
            'regra_escolha':{
                'delta_f1_material':0.02,'delta_fp_frames_material':0.01,
                'delta_map_material':0.01,'desempate_final':'yolo11n_obb'
            },
            'restricoes':{
                'teste_sintetico_lido':False,'asta_lido':False,
                'modelo_perdedor_nao_sera_avaliado_no_teste':True
            }
        }
        texto_config=json.dumps(CONFIG_SELECTION,sort_keys=True,separators=(',',':'))
        HASH_CONFIG_SELECTION=hashlib.sha256(texto_config.encode()).hexdigest()
        CONFIG_SELECTION['sha256_config_selecao']=HASH_CONFIG_SELECTION
        config_path=RUN/'protocolo_selecao_yolo_v3.json'
        config_path.write_text(json.dumps(CONFIG_SELECTION,indent=2),encoding='utf-8')
        remoto=DRIVE_EXP/config_path.name
        if remoto.exists():
            existente=json.loads(remoto.read_text(encoding='utf-8'))
            assert existente==CONFIG_SELECTION,'Já existe protocolo remoto divergente; seleção bloqueada.'
        else:
            shutil.copy2(config_path,remoto)
        print(json.dumps(CONFIG_SELECTION,indent=2,ensure_ascii=False))
        print('Contrato de seleção congelado antes do treino.')
        ''') ,
        md(r'''
        ## 3. Treino comparativo retomável

        Cada família é treinada com a mesma configuração. Um checkpoint é copiado ao Drive a
        cada cinco épocas. Resultados concluídos e compatíveis são reutilizados.
        ''') ,
        code(r'''
        EXECUTAR_SELECAO=False
        if not EXECUTAR_SELECAO:
            print('Treino bloqueado. Confira GPU, dataset e contrato; depois altere a chave para True.')
        else:
            import ultralytics
            from ultralytics import YOLO
            assert ultralytics.__version__=='8.4.127',f'Versão inesperada: {ultralytics.__version__}'
            LOCAL_PROJECT=Path('/content/yolo_family_selection_v3')/RUN_ID
            LOCAL_PROJECT.mkdir(parents=True,exist_ok=True)
            registros_treino=[]

            def callback_checkpoint(destino,modelo_chave,hash_peso_inicial):
                def salvar(trainer):
                    epoca=int(trainer.epoch)+1
                    if epoca%5!=0 and not trainer.stop:return
                    destino.mkdir(parents=True,exist_ok=True)
                    for caminho in [Path(trainer.last),Path(trainer.best),Path(trainer.save_dir)/'results.csv',Path(trainer.save_dir)/'args.yaml']:
                        if caminho.is_file():shutil.copy2(caminho,destino/caminho.name)
                    estado={'modelo':modelo_chave,'epoca':epoca,'sha256_config_selecao':HASH_CONFIG_SELECTION,
                            'sha256_peso_inicial':hash_peso_inicial}
                    (destino/'estado_checkpoint.json').write_text(json.dumps(estado,indent=2),encoding='utf-8')
                return salvar

            for modelo_chave,checkpoint in CONFIG_SELECTION['modelos'].items():
                remoto_modelo=DRIVE_EXP/'treinos'/modelo_chave
                conclusao_remota=remoto_modelo/'conclusao_treino.json'
                if conclusao_remota.exists():
                    registro=json.loads(conclusao_remota.read_text(encoding='utf-8'))
                    assert registro['sha256_config_selecao']==HASH_CONFIG_SELECTION
                    assert (remoto_modelo/'weights/best.pt').is_file()
                    registros_treino.append(registro)
                    print(modelo_chave,'já concluído e reutilizado.')
                    continue

                modelo_inicial=YOLO(checkpoint)
                peso_inicial=Path(checkpoint)
                assert peso_inicial.is_file(),f'Peso inicial não encontrado: {checkpoint}'
                hash_peso_inicial=sha256_arquivo(peso_inicial)
                checkpoint_remoto=remoto_modelo/'checkpoint'
                ultimo_remoto=checkpoint_remoto/'last.pt'
                nome_execucao=f'{modelo_chave}_seed_{CONFIG_SELECTION["seed_selecao"]}'
                destino_local=LOCAL_PROJECT/nome_execucao

                if ultimo_remoto.exists():
                    pesos_local=destino_local/'weights';pesos_local.mkdir(parents=True,exist_ok=True)
                    for nome in ['last.pt','best.pt']:
                        if (checkpoint_remoto/nome).exists():shutil.copy2(checkpoint_remoto/nome,pesos_local/nome)
                    for nome in ['results.csv','args.yaml']:
                        if (checkpoint_remoto/nome).exists():shutil.copy2(checkpoint_remoto/nome,destino_local/nome)
                    modelo=YOLO(pesos_local/'last.pt')
                    modelo.add_callback('on_model_save',callback_checkpoint(checkpoint_remoto,modelo_chave,hash_peso_inicial))
                    print('Retomando',modelo_chave,'do checkpoint remoto.')
                    modelo.train(resume=True,device=0,workers=CONFIG_SELECTION['treino']['workers'])
                else:
                    modelo=modelo_inicial
                    modelo.add_callback('on_model_save',callback_checkpoint(checkpoint_remoto,modelo_chave,hash_peso_inicial))
                    args={**CONFIG_SELECTION['treino'],'data':str(dataset_yaml),'seed':CONFIG_SELECTION['seed_selecao'],
                          'device':0,'project':str(LOCAL_PROJECT),'name':nome_execucao,'exist_ok':True,
                          'pretrained':True,'val':True,'plots':True,'save':True,'save_period':5,
                          'cache':False,'verbose':True}
                    modelo.train(**args)

                pasta_resultado=Path(modelo.trainer.save_dir)
                best=pasta_resultado/'weights/best.pt';last=pasta_resultado/'weights/last.pt'
                assert best.is_file() and last.is_file() and (pasta_resultado/'results.csv').is_file()
                copiar_arvore_com_progresso(pasta_resultado,remoto_modelo,f'Backup final {modelo_chave}')
                registro={
                    'modelo':modelo_chave,'checkpoint_inicial':checkpoint,
                    'sha256_peso_inicial':hash_peso_inicial,'seed':CONFIG_SELECTION['seed_selecao'],
                    'sha256_config_selecao':HASH_CONFIG_SELECTION,
                    'sha256_best_pt':sha256_arquivo(best),'sha256_last_pt':sha256_arquivo(last),
                    'epocas_registradas':int(len(pd.read_csv(pasta_resultado/'results.csv'))),
                    'concluido':True
                }
                (remoto_modelo/'conclusao_treino.json').write_text(json.dumps(registro,indent=2),encoding='utf-8')
                registros_treino.append(registro)
                print(json.dumps(registro,indent=2,ensure_ascii=False))
            print('Treinos de seleção concluídos:',[r['modelo'] for r in registros_treino])
        ''') ,
        md(r'''
        ## 4. Avaliação comum na validação

        O confidence threshold é calibrado separadamente para cada família na mesma grade. O
        matching combinado usa exatamente os limiares geométricos congelados no baseline Hough.
        ''') ,
        code(r'''
        EXECUTAR_AVALIACAO_VALIDACAO=False
        if not EXECUTAR_AVALIACAO_VALIDACAO:
            print('Avaliação bloqueada. Execute somente depois dos dois treinos concluídos.')
        else:
            from ultralytics import YOLO
            import gc,torch

            # Uma execução interrompida pode deixar o gerador/modelo anterior referenciado.
            for nome_residual in ['resultados','resultado','modelo','metricas','caixa']:
                globals().pop(nome_residual,None)
            gc.collect();torch.cuda.empty_cache()

            def parametros_linha(p0,p1):
                p0,p1=np.asarray(p0,dtype=float),np.asarray(p1,dtype=float)
                vetor=p1-p0;comp=float(np.linalg.norm(vetor));direcao=vetor/comp if comp>0 else np.array([1.,0.])
                return (p0+p1)/2,direcao,comp

            def erro_angular(d1,d2):
                return float(np.degrees(np.arccos(np.clip(abs(np.dot(d1,d2)),-1,1))))

            def distancia_ponto_linha(ponto,centro,direcao):
                v=np.asarray(ponto)-np.asarray(centro)
                return float(np.linalg.norm(v-np.dot(v,direcao)*direcao))

            def cobertura_projetada(p0a,p1a,p0b,p1b):
                cb,db,lb=parametros_linha(p0b,p1b)
                if lb<=0:return 0.0
                t=sorted([np.dot(np.asarray(p0a)-cb,db),np.dot(np.asarray(p1a)-cb,db)])
                inter=max(0,min(t[1],lb/2)-max(t[0],-lb/2))
                return float(inter/lb)

            def centerline(poligono):
                p=np.asarray(poligono,dtype=float)
                d01=np.linalg.norm(p[0]-p[1]);d12=np.linalg.norm(p[1]-p[2])
                if d01>=d12:return (p[0]+p[3])/2,(p[1]+p[2])/2
                return (p[0]+p[1])/2,(p[2]+p[3])/2

            def iou_poligonos(a,b):
                a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
                area_a=abs(float(cv2.contourArea(a)));area_b=abs(float(cv2.contourArea(b)))
                inter=float(cv2.intersectConvexConvex(a,b)[0]);uniao=area_a+area_b-inter
                return inter/uniao if uniao>0 else 0.0

            def ler_gt(linha):
                tokens=(RUN/'labels'/linha.split/(linha.id+'.txt')).read_text(encoding='utf-8').split()
                if not tokens:return None
                valores=np.asarray([float(x) for x in tokens[1:9]]).reshape(4,2)
                return valores*np.array([512.,512.])

            def metrica_predicao(pred,gt):
                p0,p1=centerline(pred);g0,g1=centerline(gt)
                cp,dp,_=parametros_linha(p0,p1);cg,dg,_=parametros_linha(g0,g1)
                return {'iou':iou_poligonos(pred,gt),'angulo':erro_angular(dp,dg),
                        'distancia':distancia_ponto_linha(cp,cg,dg),
                        'cobertura':cobertura_projetada(p0,p1,g0,g1)}

            def escolher_match(mets,lim):
                candidatos=[]
                for indice,m in enumerate(mets):
                    miou=m['iou']>=lim['iou']
                    mgeo=m['angulo']<=lim['angulo'] and m['distancia']<=lim['distancia'] and m['cobertura']>=lim['cobertura']
                    if not (miou or mgeo):continue
                    si=m['iou']/lim['iou'] if miou else -np.inf
                    sg=((1-m['angulo']/lim['angulo'])+(1-m['distancia']/lim['distancia'])+m['cobertura']) if mgeo else -np.inf
                    candidatos.append((max(si,sg),indice,m))
                return max(candidatos,key=lambda x:x[0]) if candidatos else None

            def avaliar_threshold(dados,predicoes,threshold):
                saida=[];lim=CONFIG_SELECTION['matching']
                for linha in dados.itertuples():
                    grupo=predicoes[(predicoes.id==linha.id)&(predicoes.conf>=threshold)]
                    polys=[g[['x1','y1','x2','y2','x3','y3','x4','y4']].to_numpy(float).reshape(4,2) for _,g in grupo.iterrows()]
                    gt=ler_gt(linha)
                    base={'id':linha.id,'tipo':linha.tipo,'subtipo':linha.subtipo,
                          'sha256_fundo':linha.sha256_fundo,'n_predicoes':len(polys)}
                    if gt is None:
                        base.update({'tp':0,'fp':len(polys),'fn':0,'acertou':False})
                    else:
                        mets=[metrica_predicao(p,gt) for p in polys];match=escolher_match(mets,lim)
                        base.update({'tp':int(match is not None),'fp':max(0,len(polys)-int(match is not None)),
                                     'fn':int(match is None),'acertou':match is not None})
                    saida.append(base)
                resultado=pd.DataFrame(saida);tp=int(resultado.tp.sum());fp=int(resultado.fp.sum());fn=int(resultado.fn.sum())
                p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f1=2*p*r/(p+r) if p+r else 0
                neg=resultado[resultado.tipo=='negativo']
                resumo={'threshold':float(threshold),'TP':tp,'FP':fp,'FN':fn,'precisao':p,'recall':r,'f1':f1,
                        'fp_por_imagem_negativa':float(neg.fp.mean()),
                        'taxa_frames_negativos_com_fp':float((neg.fp>0).mean())}
                return resultado,resumo

            def escolher_familia(tabela,regra):
                assert set(tabela.modelo)=={'yolov8n_obb','yolo11n_obb'} and len(tabela)==2
                a,b=tabela.sort_values('modelo').iloc[0],tabela.sort_values('modelo').iloc[1]
                if abs(a.f1-b.f1)>=regra['delta_f1_material']:
                    return (a if a.f1>b.f1 else b),'maior F1 combinado com diferença material'
                if abs(a.taxa_frames_negativos_com_fp-b.taxa_frames_negativos_com_fp)>=regra['delta_fp_frames_material']:
                    return (a if a.taxa_frames_negativos_com_fp<b.taxa_frames_negativos_com_fp else b),'menor taxa de frames negativos com FP'
                if abs(a.map50_95_obb-b.map50_95_obb)>=regra['delta_map_material']:
                    return (a if a.map50_95_obb>b.map50_95_obb else b),'maior mAP50-95 OBB'
                return tabela[tabela.modelo==regra['desempate_final']].iloc[0],'desempate pré-registrado por maturidade/eficiência'

            df_val=df[df.split=='val'].copy().sort_values('id')
            ids_val_esperados=set(df_val.id.astype(str))
            pasta_imagens_val=RUN/'images/val'
            assert pasta_imagens_val.is_dir()
            assert {p.stem for p in pasta_imagens_val.glob('*.png')}==ids_val_esperados
            candidatos=[];ARTEFATOS_AVALIACAO=RUN/'yolo_selection_validation';ARTEFATOS_AVALIACAO.mkdir(exist_ok=True)
            for modelo_chave in CONFIG_SELECTION['modelos']:
                gc.collect();torch.cuda.empty_cache()
                remoto_modelo=DRIVE_EXP/'treinos'/modelo_chave
                conclusao=json.loads((remoto_modelo/'conclusao_treino.json').read_text(encoding='utf-8'))
                best_remoto=remoto_modelo/'weights/best.pt'
                assert sha256_arquivo(best_remoto)==conclusao['sha256_best_pt']
                modelo=YOLO(best_remoto)
                registros=[];tempos=[];ids_val_observados=[]
                # Passar uma lista de 900 caminhos pode virar um lote único no Ultralytics,
                # mesmo com stream=True. Um diretório + batch=1 mantém uso de VRAM limitado.
                resultados=modelo.predict(source=str(pasta_imagens_val),stream=True,batch=1,imgsz=CONFIG_SELECTION['treino']['imgsz'],
                    conf=CONFIG_SELECTION['predicao_val']['conf_min'],iou=CONFIG_SELECTION['predicao_val']['iou_nms'],
                    max_det=CONFIG_SELECTION['predicao_val']['max_det'],device=0,verbose=False)
                for resultado in tqdm(resultados,total=len(ids_val_esperados),desc=f'Predições {modelo_chave}',unit='img'):
                    id_amostra=Path(resultado.path).stem;ids_val_observados.append(id_amostra)
                    tempos.append(float(resultado.speed.get('inference',np.nan)))
                    if resultado.obb is None:continue
                    polys=resultado.obb.xyxyxyxy.cpu().numpy();confs=resultado.obb.conf.cpu().numpy()
                    for poly,conf in zip(polys,confs):
                        plano=poly.reshape(-1)
                        registros.append({'id':id_amostra,'conf':float(conf),
                                          **{f'{eixo}{i+1}':float(plano[2*i+(0 if eixo=="x" else 1)]) for i in range(4) for eixo in ['x','y']}})
                assert len(ids_val_observados)==len(ids_val_esperados)
                assert len(set(ids_val_observados))==len(ids_val_observados)
                assert set(ids_val_observados)==ids_val_esperados
                del resultados,resultado
                gc.collect();torch.cuda.empty_cache()
                pred=pd.DataFrame(registros,columns=['id','conf','x1','y1','x2','y2','x3','y3','x4','y4'])
                pred.to_csv(ARTEFATOS_AVALIACAO/f'predicoes_val_{modelo_chave}.csv',index=False)
                grades=[];resultados_por_threshold={}
                for threshold in CONFIG_SELECTION['thresholds_confianca']:
                    por_imagem,resumo=avaliar_threshold(df_val,pred,threshold)
                    grades.append(resumo);resultados_por_threshold[threshold]=por_imagem
                grade=pd.DataFrame(grades).sort_values(['f1','precisao','threshold'],ascending=[False,False,False])
                melhor=grade.iloc[0].to_dict();threshold=float(melhor['threshold'])
                grade.to_csv(ARTEFATOS_AVALIACAO/f'grade_threshold_val_{modelo_chave}.csv',index=False)
                resultados_por_threshold[threshold].to_csv(ARTEFATOS_AVALIACAO/f'resultado_val_{modelo_chave}.csv',index=False)
                metricas=modelo.val(data=str(dataset_yaml),split='val',imgsz=CONFIG_SELECTION['treino']['imgsz'],
                                    batch=CONFIG_SELECTION['treino']['batch'],device=0,workers=2,plots=False,verbose=False,
                                    project=str(ARTEFATOS_AVALIACAO/'ultralytics'),name=modelo_chave,exist_ok=True)
                caixa=metricas.box
                candidato={'modelo':modelo_chave,**melhor,'map50_95_obb':float(caixa.map),'map50_obb':float(caixa.map50),
                           'inferencia_ms_media':float(np.nanmean(tempos)),'sha256_best_pt':conclusao['sha256_best_pt']}
                candidatos.append(candidato)
                del metricas,caixa,modelo
                gc.collect();torch.cuda.empty_cache()

            tabela=pd.DataFrame(candidatos).sort_values('modelo').reset_index(drop=True)
            regra=CONFIG_SELECTION['regra_escolha']
            vencedor,motivo=escolher_familia(tabela,regra)
            comparacao_path=ARTEFATOS_AVALIACAO/'comparacao_modelos_val_v3.csv'
            tabela.to_csv(comparacao_path,index=False)
            recomendacao={'protocolo':'yolo_family_selection_v1','run_id':RUN_ID,
                          'sha256_manifest':HASH_MANIFEST,'sha256_config_selecao':HASH_CONFIG_SELECTION,
                          'vencedor_recomendado':vencedor.modelo,'motivo':motivo,
                          'threshold_confianca':float(vencedor.threshold),
                          'sha256_best_pt':vencedor.sha256_best_pt,
                          'sha256_tabela_comparacao':sha256_arquivo(comparacao_path),
                          'teste_sintetico_lido':False,'asta_lido':False}
            (ARTEFATOS_AVALIACAO/'recomendacao_local_v3.json').write_text(json.dumps(recomendacao,indent=2),encoding='utf-8')
            copiar_arvore_com_progresso(ARTEFATOS_AVALIACAO,DRIVE_EXP/'avaliacao_val','Backup avaliação de validação')
            print(tabela[['modelo','threshold','precisao','recall','f1','taxa_frames_negativos_com_fp','map50_95_obb','inferencia_ms_media']])
            print(json.dumps(recomendacao,indent=2,ensure_ascii=False))
            print('Recomendação local calculada. Publicação do vencedor ainda bloqueada.')
        ''') ,
        md(r'''
        ## 5. Gate humano e publicação do vencedor

        Só altere a chave depois de revisar a tabela completa. A publicação não toca no teste e
        apenas transforma a recomendação determinística em contrato oficial de seleção.
        ''') ,
        code(r'''
        APROVAR_SELECAO_YOLO=False
        recomendacao_path=RUN/'yolo_selection_validation/recomendacao_local_v3.json'
        if not APROVAR_SELECAO_YOLO:
            print('Seleção não publicada. Compartilhe a tabela e a recomendação para revisão.')
        else:
            recomendacao=json.loads(recomendacao_path.read_text(encoding='utf-8'))
            assert recomendacao['sha256_manifest']==HASH_MANIFEST
            assert recomendacao['sha256_config_selecao']==HASH_CONFIG_SELECTION
            comparacao_path=RUN/'yolo_selection_validation/comparacao_modelos_val_v3.csv'
            assert sha256_arquivo(comparacao_path)==recomendacao['sha256_tabela_comparacao']
            best_remoto=DRIVE_EXP/'treinos'/recomendacao['vencedor_recomendado']/'weights/best.pt'
            assert sha256_arquivo(best_remoto)==recomendacao['sha256_best_pt']
            contrato={**recomendacao,'aprovado':True,'seeds_finais':CONFIG_SELECTION['seeds_finais'],
                      'regra_escolha':CONFIG_SELECTION['regra_escolha']}
            texto=json.dumps(contrato,sort_keys=True,separators=(',',':'))
            contrato['sha256_contrato_selecao']=hashlib.sha256(texto.encode()).hexdigest()
            contrato_path=DRIVE_EXP/'contrato_selecao_yolo_v3.json'
            if contrato_path.exists():
                assert json.loads(contrato_path.read_text(encoding='utf-8'))==contrato,'Contrato remoto divergente.'
            else:
                contrato_path.write_text(json.dumps(contrato,indent=2),encoding='utf-8')
            ponteiro=BACKUP_DIR/'experiments/yolo_family_selection_v3/LATEST_SELECTION_V3.json'
            ponteiro.parent.mkdir(parents=True,exist_ok=True)
            ponteiro.write_text(json.dumps({
                'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                'sha256_contrato_selecao':contrato['sha256_contrato_selecao'],
                'vencedor':contrato['vencedor_recomendado']
            },indent=2),encoding='utf-8')
            print(json.dumps(contrato,indent=2,ensure_ascii=False))
            print('SELEÇÃO YOLO PUBLICADA. Teste sintético e ASTA permanecem bloqueados.')
        ''') ,
    ]
    return write_notebook('08_selecao_yolov8n_yolo11n_validacao_v3.ipynb', cells)


def build_09():
    # Reusa somente as células estáveis de gate GPU e restauração do Notebook 08.
    # O 08 é sempre gerado antes do 09 no fluxo principal.
    nb08_path = OUT / '08_selecao_yolov8n_yolo11n_validacao_v3.ipynb'
    if not nb08_path.is_file():
        build_08()
    nb08 = json.loads(nb08_path.read_text(encoding='utf-8'))
    gpu_gate = nb08['cells'][1]
    restore_dataset = nb08['cells'][3]

    cells = [
        md(r'''
        # 09 — Treino final multiseed e seleção operacional por validação (v3)

        Este notebook treina a família já congelada, **YOLO11n-OBB**, nas seeds
        `20260823`, `20260824` e `20260825`. Ele não escolhe novamente a família e não lê
        predições do teste sintético nem dados ASTA.

        A seed `20260823` pode ser reutilizada do Notebook 08 somente porque arquitetura,
        peso inicial, dados, hiperparâmetros e seed são idênticos e todos os hashes são
        verificados. As outras duas seeds são treinos independentes e retomáveis.

        Ao final, os três `best.pt` são avaliados somente na validação. Um checkpoint
        operacional é congelado para uso externo, mas o teste sintético futuro avaliará os
        três checkpoints em um único evento e reportará variabilidade entre seeds. Nenhuma
        seed poderá ser escolhida depois de observar o teste.
        ''') ,
        gpu_gate,
        md(r'''
        ## 1. Ambiente fixado e restauração pelo TAR

        A restauração é idêntica à aprovada no Notebook 08: um único TAR, verificação de
        SHA-256 e extração local. Execute em GPU desde a primeira célula.
        ''') ,
        restore_dataset,
        md(r'''
        ## 2. Contrato final antes do treino

        Esta seção valida o contrato publicado no Notebook 08 e congela, antes dos novos
        treinos, as regras de early stopping, checkpoint, seleção operacional e teste.

        **Regras fechadas:**
        - cada seed usa `best.pt` produzido pelo fitness de validação do Ultralytics 8.4.127;
        - `patience=20`; teste e ASTA não participam do early stopping;
        - o checkpoint operacional é escolhido somente na validação por F1 combinado em
          `confidence=0,45`, com desempates pré-definidos;
        - o teste protegido avaliará os três checkpoints uma única vez e reportará a média,
          o desvio entre seeds e IC por bootstrap agrupado por fundo;
        - resultados de teste nunca selecionarão seed, threshold ou checkpoint.
        ''') ,
        code(r'''
        DRIVE_SELECTION=BACKUP_DIR/'experiments/yolo_family_selection_v3'/RUN_ID
        DRIVE_FINAL=BACKUP_DIR/'experiments/yolo_final_multiseed_v3'/RUN_ID
        DRIVE_FINAL.mkdir(parents=True,exist_ok=True)

        def verificar_hash_canonico(registro,campo_hash):
            copia=dict(registro);observado=copia.pop(campo_hash)
            texto=json.dumps(copia,sort_keys=True,separators=(',',':'))
            calculado=hashlib.sha256(texto.encode()).hexdigest()
            assert calculado==observado,f'Hash canônico divergente em {campo_hash}'
            return observado

        ponteiro=json.loads((BACKUP_DIR/'experiments/yolo_family_selection_v3/LATEST_SELECTION_V3.json').read_text(encoding='utf-8'))
        contrato_selecao=json.loads((DRIVE_SELECTION/'contrato_selecao_yolo_v3.json').read_text(encoding='utf-8'))
        config_selecao=json.loads((DRIVE_SELECTION/'protocolo_selecao_yolo_v3.json').read_text(encoding='utf-8'))
        HASH_CONTRATO_SELECTION=verificar_hash_canonico(contrato_selecao,'sha256_contrato_selecao')
        HASH_CONFIG_SELECTION=verificar_hash_canonico(config_selecao,'sha256_config_selecao')
        assert ponteiro['run_id']==RUN_ID and ponteiro['sha256_manifest']==HASH_MANIFEST
        assert ponteiro['sha256_contrato_selecao']==HASH_CONTRATO_SELECTION
        assert contrato_selecao['aprovado'] is True
        assert contrato_selecao['vencedor_recomendado']=='yolo11n_obb'
        assert float(contrato_selecao['threshold_confianca'])==0.45
        assert contrato_selecao['teste_sintetico_lido'] is False and contrato_selecao['asta_lido'] is False
        assert config_selecao['ultralytics']=='8.4.127'

        CONFIG_FINAL={
            'protocolo':'yolo_final_multiseed_v1','run_id':RUN_ID,
            'sha256_manifest':HASH_MANIFEST,
            'sha256_contrato_selecao':HASH_CONTRATO_SELECTION,
            'sha256_config_selecao':HASH_CONFIG_SELECTION,
            'familia':'yolo11n_obb','checkpoint_inicial':'yolo11n-obb.pt',
            'seeds':[20260823,20260824,20260825],
            'reutilizar_seed_selecao_se_identica':True,
            'treino':config_selecao['treino'],
            'early_stopping':{
                'conjunto':'val','patience':20,
                'checkpoint_por_seed':'best.pt do fitness interno de validação do Ultralytics 8.4.127',
                'teste_sintetico_usado':False,'asta_usado':False
            },
            'predicao_val':config_selecao['predicao_val'],
            'threshold_confianca_fixo':float(contrato_selecao['threshold_confianca']),
            'matching':config_selecao['matching'],
            'regra_checkpoint_operacional':[
                'maior_f1_combinado_val','menor_taxa_frames_negativos_com_fp_val',
                'maior_map50_95_obb_val','menor_seed'
            ],
            'plano_teste_protegido':{
                'evento_unico':True,'avaliar_tres_checkpoints':True,
                'resultado_primario':'média das métricas das três seeds',
                'variabilidade_treino':'desvio-padrão e faixa entre seeds',
                'incerteza_amostral':'bootstrap por sha256_fundo da média entre seeds',
                'selecionar_seed_com_teste':False,'recalibrar_threshold_com_teste':False,
                'checkpoint_operacional_para_asta':'escolhido somente na validação',
                'asta_permanece_intocado':True
            }
        }
        assert CONFIG_FINAL['treino']['patience']==CONFIG_FINAL['early_stopping']['patience']==20
        assert CONFIG_FINAL['seeds']==contrato_selecao['seeds_finais']
        texto_final=json.dumps(CONFIG_FINAL,sort_keys=True,separators=(',',':'))
        HASH_CONFIG_FINAL=hashlib.sha256(texto_final.encode()).hexdigest()
        CONFIG_FINAL['sha256_config_final']=HASH_CONFIG_FINAL
        config_final_path=RUN/'protocolo_treino_final_multiseed_v3.json'
        config_final_path.write_text(json.dumps(CONFIG_FINAL,indent=2),encoding='utf-8')
        remoto=DRIVE_FINAL/config_final_path.name
        if remoto.exists():
            assert json.loads(remoto.read_text(encoding='utf-8'))==CONFIG_FINAL,'Protocolo final remoto divergente.'
        else:
            shutil.copy2(config_final_path,remoto)
        print(json.dumps(CONFIG_FINAL,indent=2,ensure_ascii=False))
        print('PROTOCOLO FINAL MULTISEED PUBLICADO. Teste sintético e ASTA seguem bloqueados.')
        ''') ,
        md(r'''
        ## 3. Treinos independentes e retomáveis

        Checkpoints parciais são publicados a cada cinco épocas. Uma seed concluída é
        reutilizada somente se o registro, o hash da configuração e o hash do `best.pt`
        coincidirem. Alterar hiperparâmetros exige um novo protocolo e outro diretório.
        ''') ,
        code(r'''
        EXECUTAR_TREINOS_FINAIS=False
        if not EXECUTAR_TREINOS_FINAIS:
            print('Treinos bloqueados. Compartilhe primeiro a saída da seção 2; depois altere a chave para True.')
        else:
            import ultralytics
            from ultralytics import YOLO
            assert ultralytics.__version__=='8.4.127',f'Versão inesperada: {ultralytics.__version__}'
            LOCAL_PROJECT=Path('/content/yolo_final_multiseed_v3')/RUN_ID
            LOCAL_PROJECT.mkdir(parents=True,exist_ok=True)
            registros=[]

            selecao_treino=DRIVE_SELECTION/'treinos/yolo11n_obb'
            conclusao_selecao=json.loads((selecao_treino/'conclusao_treino.json').read_text(encoding='utf-8'))
            assert conclusao_selecao['seed']==20260823
            assert conclusao_selecao['sha256_config_selecao']==HASH_CONFIG_SELECTION
            assert conclusao_selecao['sha256_best_pt']==contrato_selecao['sha256_best_pt']
            assert sha256_arquivo(selecao_treino/'weights/best.pt')==conclusao_selecao['sha256_best_pt']
            HASH_PESO_INICIAL_ESPERADO=conclusao_selecao['sha256_peso_inicial']

            def callback_checkpoint(destino,seed):
                def salvar(trainer):
                    epoca=int(trainer.epoch)+1
                    if epoca%5!=0 and not trainer.stop:return
                    destino.mkdir(parents=True,exist_ok=True)
                    candidatos=[Path(trainer.last),Path(trainer.best),Path(trainer.save_dir)/'results.csv',Path(trainer.save_dir)/'args.yaml']
                    for caminho in candidatos:
                        if caminho.is_file():shutil.copy2(caminho,destino/caminho.name)
                    estado={'seed':int(seed),'epoca':epoca,'sha256_config_final':HASH_CONFIG_FINAL,
                            'sha256_peso_inicial':HASH_PESO_INICIAL_ESPERADO}
                    (destino/'estado_checkpoint.json').write_text(json.dumps(estado,indent=2),encoding='utf-8')
                return salvar

            for seed in CONFIG_FINAL['seeds']:
                chave=f'seed_{seed}'
                remoto_seed=DRIVE_FINAL/'treinos'/chave
                conclusao_path=remoto_seed/'conclusao_treino_final.json'
                if conclusao_path.exists():
                    registro=json.loads(conclusao_path.read_text(encoding='utf-8'))
                    assert registro['sha256_config_final']==HASH_CONFIG_FINAL and registro['seed']==seed
                    best_path=BACKUP_DIR/registro['best_pt_relativo_drive']
                    assert best_path.is_file() and sha256_arquivo(best_path)==registro['sha256_best_pt']
                    registros.append(registro);print(chave,'já concluída e reutilizada.');continue

                remoto_seed.mkdir(parents=True,exist_ok=True)
                if seed==20260823:
                    best=selecao_treino/'weights/best.pt';last=selecao_treino/'weights/last.pt'
                    registro={
                        'protocolo':'yolo_final_multiseed_v1','familia':'yolo11n_obb','seed':seed,
                        'sha256_config_final':HASH_CONFIG_FINAL,'checkpoint_inicial':'yolo11n-obb.pt',
                        'sha256_peso_inicial':HASH_PESO_INICIAL_ESPERADO,
                        'sha256_best_pt':sha256_arquivo(best),'sha256_last_pt':sha256_arquivo(last),
                        'best_pt_relativo_drive':str(best.relative_to(BACKUP_DIR)).replace('\\','/'),
                        'epocas_registradas':int(conclusao_selecao['epocas_registradas']),
                        'origem':'reutilizado_do_notebook_08_com_invariantes_verificados','concluido':True
                    }
                    conclusao_path.write_text(json.dumps(registro,indent=2),encoding='utf-8')
                    registros.append(registro);print(json.dumps(registro,indent=2,ensure_ascii=False));continue

                checkpoint_remoto=remoto_seed/'checkpoint'
                ultimo_remoto=checkpoint_remoto/'last.pt'
                nome_execucao=f'yolo11n_obb_seed_{seed}'
                destino_local=LOCAL_PROJECT/nome_execucao
                if ultimo_remoto.exists():
                    estado=json.loads((checkpoint_remoto/'estado_checkpoint.json').read_text(encoding='utf-8'))
                    assert estado['seed']==seed and estado['sha256_config_final']==HASH_CONFIG_FINAL
                    pesos_local=destino_local/'weights';pesos_local.mkdir(parents=True,exist_ok=True)
                    for nome in ['last.pt','best.pt']:
                        if (checkpoint_remoto/nome).exists():shutil.copy2(checkpoint_remoto/nome,pesos_local/nome)
                    for nome in ['results.csv','args.yaml']:
                        if (checkpoint_remoto/nome).exists():shutil.copy2(checkpoint_remoto/nome,destino_local/nome)
                    modelo=YOLO(pesos_local/'last.pt')
                    modelo.add_callback('on_model_save',callback_checkpoint(checkpoint_remoto,seed))
                    print('Retomando',chave,'do checkpoint remoto.')
                    modelo.train(resume=True,device=0,workers=CONFIG_FINAL['treino']['workers'])
                else:
                    modelo=YOLO(CONFIG_FINAL['checkpoint_inicial'])
                    peso_inicial=Path(CONFIG_FINAL['checkpoint_inicial'])
                    assert peso_inicial.is_file()
                    assert sha256_arquivo(peso_inicial)==HASH_PESO_INICIAL_ESPERADO
                    modelo.add_callback('on_model_save',callback_checkpoint(checkpoint_remoto,seed))
                    args={**CONFIG_FINAL['treino'],'data':str(dataset_yaml),'seed':seed,'device':0,
                          'project':str(LOCAL_PROJECT),'name':nome_execucao,'exist_ok':True,
                          'pretrained':True,'val':True,'plots':True,'save':True,'save_period':5,
                          'cache':False,'verbose':True}
                    modelo.train(**args)

                pasta_resultado=Path(modelo.trainer.save_dir)
                best=pasta_resultado/'weights/best.pt';last=pasta_resultado/'weights/last.pt'
                assert best.is_file() and last.is_file() and (pasta_resultado/'results.csv').is_file()
                copiar_arvore_com_progresso(pasta_resultado,remoto_seed,f'Backup final {chave}')
                best_remoto=remoto_seed/'weights/best.pt';last_remoto=remoto_seed/'weights/last.pt'
                registro={
                    'protocolo':'yolo_final_multiseed_v1','familia':'yolo11n_obb','seed':seed,
                    'sha256_config_final':HASH_CONFIG_FINAL,'checkpoint_inicial':'yolo11n-obb.pt',
                    'sha256_peso_inicial':HASH_PESO_INICIAL_ESPERADO,
                    'sha256_best_pt':sha256_arquivo(best_remoto),'sha256_last_pt':sha256_arquivo(last_remoto),
                    'best_pt_relativo_drive':str(best_remoto.relative_to(BACKUP_DIR)).replace('\\','/'),
                    'epocas_registradas':int(len(pd.read_csv(remoto_seed/'results.csv'))),
                    'origem':'treino_final_independente','concluido':True
                }
                conclusao_path.write_text(json.dumps(registro,indent=2),encoding='utf-8')
                registros.append(registro);print(json.dumps(registro,indent=2,ensure_ascii=False))
            print('TREINOS MULTISEED CONCLUÍDOS:',[r['seed'] for r in registros])
        ''') ,
        md(r'''
        ## 4. Gate local dos três checkpoints

        Execute esta seção antes da avaliação de validação. Compartilhe o JSON produzido.
        ''') ,
        code(r'''
        registros=[];erros=[]
        for seed in CONFIG_FINAL['seeds']:
            caminho=DRIVE_FINAL/'treinos'/f'seed_{seed}'/'conclusao_treino_final.json'
            if not caminho.exists():
                erros.append(f'ausente:{seed}');continue
            registro=json.loads(caminho.read_text(encoding='utf-8'))
            if registro.get('sha256_config_final')!=HASH_CONFIG_FINAL:erros.append(f'config:{seed}')
            best=BACKUP_DIR/registro['best_pt_relativo_drive']
            if not best.is_file() or sha256_arquivo(best)!=registro.get('sha256_best_pt'):erros.append(f'best:{seed}')
            registros.append(registro)
        resumo_treinos={
            'status':'TREINOS_MULTISEED_APROVADOS' if not erros and len(registros)==3 else 'REPROVADO',
            'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,'sha256_config_final':HASH_CONFIG_FINAL,
            'seeds_encontradas':sorted(r['seed'] for r in registros),
            'epocas_por_seed':{str(r['seed']):r['epocas_registradas'] for r in registros},
            'sha256_best_por_seed':{str(r['seed']):r['sha256_best_pt'] for r in registros},
            'erros':erros
        }
        print(json.dumps(resumo_treinos,indent=2,ensure_ascii=False))
        assert resumo_treinos['status']=='TREINOS_MULTISEED_APROVADOS'
        ''') ,
        md(r'''
        ## 5. Avaliação comum somente na validação

        Esta célula usa o threshold `0,45` já congelado; não recalibra uma grade por seed.
        Predições são processadas em streaming com `batch=1` para limitar VRAM. A recomendação
        de checkpoint operacional ainda precisa de aprovação humana na seção seguinte.
        ''') ,
        code(r'''
        EXECUTAR_AVALIACAO_FINAL_VAL=False
        if not EXECUTAR_AVALIACAO_FINAL_VAL:
            print('Avaliação bloqueada. Execute somente após o gate TREINOS_MULTISEED_APROVADOS.')
        else:
            from ultralytics import YOLO
            import gc,torch
            gc.collect();torch.cuda.empty_cache()

            def parametros_linha(p0,p1):
                p0,p1=np.asarray(p0,dtype=float),np.asarray(p1,dtype=float)
                v=p1-p0;comp=float(np.linalg.norm(v));direcao=v/comp if comp>0 else np.array([1.,0.])
                return (p0+p1)/2,direcao,comp
            def centerline(p):
                p=np.asarray(p,dtype=float);d01=np.linalg.norm(p[0]-p[1]);d12=np.linalg.norm(p[1]-p[2])
                return ((p[0]+p[3])/2,(p[1]+p[2])/2) if d01>=d12 else ((p[0]+p[1])/2,(p[2]+p[3])/2)
            def iou_poligonos(a,b):
                a,b=np.asarray(a,np.float32),np.asarray(b,np.float32);aa=abs(float(cv2.contourArea(a)));ab=abs(float(cv2.contourArea(b)))
                inter=float(cv2.intersectConvexConvex(a,b)[0]);uniao=aa+ab-inter
                return inter/uniao if uniao>0 else 0.0
            def metricas_match(pred,gt):
                p0,p1=centerline(pred);g0,g1=centerline(gt);cp,dp,_=parametros_linha(p0,p1);cg,dg,lg=parametros_linha(g0,g1)
                ang=float(np.degrees(np.arccos(np.clip(abs(np.dot(dp,dg)),-1,1))))
                v=cp-cg;dist=float(np.linalg.norm(v-np.dot(v,dg)*dg))
                t=sorted([np.dot(p0-cg,dg),np.dot(p1-cg,dg)]);inter=max(0,min(t[1],lg/2)-max(t[0],-lg/2));cob=float(inter/lg) if lg>0 else 0.0
                return {'iou':iou_poligonos(pred,gt),'angulo':ang,'distancia':dist,'cobertura':cob}
            def ler_gt(linha):
                tokens=(RUN/'labels'/linha.split/(linha.id+'.txt')).read_text(encoding='utf-8').split()
                return None if not tokens else np.asarray([float(x) for x in tokens[1:9]]).reshape(4,2)*np.array([512.,512.])
            def aceito(m,lim):
                return m['iou']>=lim['iou'] or (m['angulo']<=lim['angulo'] and m['distancia']<=lim['distancia'] and m['cobertura']>=lim['cobertura'])
            def avaliar(dados,predicoes,threshold):
                saida=[];lim=CONFIG_FINAL['matching']
                for linha in dados.itertuples():
                    grupo=predicoes[(predicoes.id==linha.id)&(predicoes.conf>=threshold)]
                    polys=[g[['x1','y1','x2','y2','x3','y3','x4','y4']].to_numpy(float).reshape(4,2) for _,g in grupo.iterrows()]
                    gt=ler_gt(linha);match=False
                    if gt is not None:match=any(aceito(metricas_match(p,gt),lim) for p in polys)
                    saida.append({'id':linha.id,'tipo':linha.tipo,'subtipo':linha.subtipo,'sha256_fundo':linha.sha256_fundo,
                                  'tp':int(gt is not None and match),'fp':len(polys)-int(gt is not None and match),
                                  'fn':int(gt is not None and not match),'n_predicoes':len(polys)})
                out=pd.DataFrame(saida);tp=int(out.tp.sum());fp=int(out.fp.sum());fn=int(out.fn.sum())
                p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f1=2*p*r/(p+r) if p+r else 0
                neg=out[out.tipo=='negativo']
                return out,{'TP':tp,'FP':fp,'FN':fn,'precisao':p,'recall':r,'f1':f1,
                            'fp_por_imagem_negativa':float(neg.fp.mean()),
                            'taxa_frames_negativos_com_fp':float((neg.fp>0).mean())}

            df_val=df[df.split=='val'].copy().sort_values('id');ids_esperados=set(df_val.id.astype(str))
            pasta_val=RUN/'images/val';assert {p.stem for p in pasta_val.glob('*.png')}==ids_esperados
            ARTEFATOS_VAL=RUN/'yolo_final_validation';ARTEFATOS_VAL.mkdir(exist_ok=True)
            linhas=[]
            for registro in registros:
                seed=int(registro['seed']);best=BACKUP_DIR/registro['best_pt_relativo_drive']
                assert sha256_arquivo(best)==registro['sha256_best_pt']
                gc.collect();torch.cuda.empty_cache();modelo=YOLO(best)
                preds=[];tempos=[];ids=[]
                resultados=modelo.predict(source=str(pasta_val),stream=True,batch=1,imgsz=CONFIG_FINAL['treino']['imgsz'],
                    conf=CONFIG_FINAL['predicao_val']['conf_min'],iou=CONFIG_FINAL['predicao_val']['iou_nms'],
                    max_det=CONFIG_FINAL['predicao_val']['max_det'],device=0,verbose=False)
                for resultado in tqdm(resultados,total=len(ids_esperados),desc=f'Val seed {seed}',unit='img'):
                    id_amostra=Path(resultado.path).stem;ids.append(id_amostra);tempos.append(float(resultado.speed.get('inference',np.nan)))
                    if resultado.obb is None:continue
                    for poly,conf in zip(resultado.obb.xyxyxyxy.cpu().numpy(),resultado.obb.conf.cpu().numpy()):
                        plano=poly.reshape(-1);preds.append({'id':id_amostra,'conf':float(conf),
                            **{f'{e}{i+1}':float(plano[2*i+(0 if e=='x' else 1)]) for i in range(4) for e in ['x','y']}})
                assert len(ids)==len(ids_esperados) and len(set(ids))==len(ids) and set(ids)==ids_esperados
                pred=pd.DataFrame(preds,columns=['id','conf','x1','y1','x2','y2','x3','y3','x4','y4'])
                pred.to_csv(ARTEFATOS_VAL/f'predicoes_val_seed_{seed}.csv',index=False)
                por_imagem,resumo=avaliar(df_val,pred,CONFIG_FINAL['threshold_confianca_fixo'])
                por_imagem.to_csv(ARTEFATOS_VAL/f'resultado_val_seed_{seed}.csv',index=False)
                metricas=modelo.val(data=str(dataset_yaml),split='val',imgsz=CONFIG_FINAL['treino']['imgsz'],
                    batch=CONFIG_FINAL['treino']['batch'],device=0,workers=2,plots=False,verbose=False,
                    project=str(ARTEFATOS_VAL/'ultralytics'),name=f'seed_{seed}',exist_ok=True)
                linhas.append({'seed':seed,**resumo,'map50_95_obb':float(metricas.box.map),
                               'map50_obb':float(metricas.box.map50),'inferencia_ms_media':float(np.nanmean(tempos)),
                               'sha256_best_pt':registro['sha256_best_pt']})
                del resultados,resultado,metricas,modelo;gc.collect();torch.cuda.empty_cache()

            comparacao=pd.DataFrame(linhas).sort_values('seed').reset_index(drop=True)
            assert set(comparacao.seed)==set(CONFIG_FINAL['seeds']) and len(comparacao)==3
            ordenada=comparacao.sort_values(['f1','taxa_frames_negativos_com_fp','map50_95_obb','seed'],ascending=[False,True,False,True])
            campeao=ordenada.iloc[0]
            comparacao_path=ARTEFATOS_VAL/'comparacao_seeds_val_v3.csv';comparacao.to_csv(comparacao_path,index=False)
            recomendacao={
                'protocolo':'yolo_final_multiseed_v1','run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                'sha256_config_final':HASH_CONFIG_FINAL,'threshold_confianca':CONFIG_FINAL['threshold_confianca_fixo'],
                'seeds_avaliadas':CONFIG_FINAL['seeds'],'checkpoint_operacional_seed':int(campeao.seed),
                'checkpoint_operacional_sha256':campeao.sha256_best_pt,
                'regra_checkpoint_operacional':CONFIG_FINAL['regra_checkpoint_operacional'],
                'sha256_tabela_validacao':sha256_arquivo(comparacao_path),
                'teste_sintetico_lido':False,'asta_lido':False
            }
            (ARTEFATOS_VAL/'recomendacao_checkpoint_operacional_v3.json').write_text(json.dumps(recomendacao,indent=2),encoding='utf-8')
            copiar_arvore_com_progresso(ARTEFATOS_VAL,DRIVE_FINAL/'avaliacao_val','Backup avaliação multiseed em validação')
            print(comparacao[['seed','precisao','recall','f1','taxa_frames_negativos_com_fp','map50_95_obb','inferencia_ms_media']])
            print(json.dumps(recomendacao,indent=2,ensure_ascii=False))
            print('RECOMENDAÇÃO LOCAL PRONTA. Teste e ASTA continuam bloqueados.')
        ''') ,
        md(r'''
        ## 6. Gate humano e contrato dos modelos finais

        Só publique após revisar a tabela das três seeds. Este gate congela os três hashes,
        o checkpoint operacional e todo o plano do teste. Ele ainda não executa o teste.
        ''') ,
        code(r'''
        APROVAR_MODELOS_FINAIS=False
        recomendacao_path=RUN/'yolo_final_validation/recomendacao_checkpoint_operacional_v3.json'
        if not APROVAR_MODELOS_FINAIS:
            print('Contrato final não publicado. Compartilhe a tabela e a recomendação para revisão.')
        else:
            recomendacao=json.loads(recomendacao_path.read_text(encoding='utf-8'))
            assert recomendacao['sha256_manifest']==HASH_MANIFEST and recomendacao['sha256_config_final']==HASH_CONFIG_FINAL
            tabela_path=RUN/'yolo_final_validation/comparacao_seeds_val_v3.csv'
            assert sha256_arquivo(tabela_path)==recomendacao['sha256_tabela_validacao']
            checkpoints=[]
            for registro in registros:
                best=BACKUP_DIR/registro['best_pt_relativo_drive']
                assert sha256_arquivo(best)==registro['sha256_best_pt']
                checkpoints.append({'seed':registro['seed'],'sha256_best_pt':registro['sha256_best_pt'],
                    'best_pt_relativo_drive':registro['best_pt_relativo_drive']})
            assert len(checkpoints)==3 and len({x['seed'] for x in checkpoints})==3
            assert recomendacao['checkpoint_operacional_sha256'] in {x['sha256_best_pt'] for x in checkpoints}
            contrato={**recomendacao,'aprovado':True,'checkpoints':sorted(checkpoints,key=lambda x:x['seed']),
                      'plano_teste_protegido':CONFIG_FINAL['plano_teste_protegido']}
            texto=json.dumps(contrato,sort_keys=True,separators=(',',':'))
            contrato['sha256_contrato_modelos_finais']=hashlib.sha256(texto.encode()).hexdigest()
            destino=DRIVE_FINAL/'contrato_modelos_finais_v3.json'
            if destino.exists():assert json.loads(destino.read_text(encoding='utf-8'))==contrato,'Contrato final remoto divergente.'
            else:destino.write_text(json.dumps(contrato,indent=2),encoding='utf-8')
            ponteiro=BACKUP_DIR/'experiments/yolo_final_multiseed_v3/LATEST_FINAL_MODELS_V3.json'
            ponteiro.parent.mkdir(parents=True,exist_ok=True)
            ponteiro.write_text(json.dumps({'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                'sha256_contrato_modelos_finais':contrato['sha256_contrato_modelos_finais'],
                'checkpoint_operacional_seed':contrato['checkpoint_operacional_seed']},indent=2),encoding='utf-8')
            print(json.dumps(contrato,indent=2,ensure_ascii=False))
            print('MODELOS FINAIS PUBLICADOS. Notebook de teste pode ser gerado; ASTA permanece bloqueado.')
        ''') ,
    ]
    return write_notebook('09_treino_final_multiseed_yolo11n_validacao_v3.ipynb', cells)


def build_10():
    # Reusa somente o gate de GPU e a restauração íntegra pelo TAR já aprovados no Notebook 08.
    nb08_path = OUT / '08_selecao_yolov8n_yolo11n_validacao_v3.ipynb'
    if not nb08_path.is_file():
        build_08()
    nb08 = json.loads(nb08_path.read_text(encoding='utf-8'))
    gpu_gate = nb08['cells'][1]
    restore_dataset = nb08['cells'][3]

    cells = [
        md(r'''
        # 10 — Teste sintético YOLO protegido e comparação com Hough (v3)

        Este notebook abre o teste sintético **uma única vez**, depois de família, pesos,
        threshold, matching e regras estatísticas terem sido congelados no Notebook 09.
        Os três checkpoints YOLO11n-OBB são avaliados; o resultado primário é a média entre
        seeds. Nenhuma seed, threshold ou regra poderá ser alterada após observar o teste.

        O evento é retomável: uma interrupção pode continuar o mesmo `event_id`, reutilizando
        somente seeds já concluídas e verificadas por hash. Um evento finalizado jamais executa
        inferência novamente. ASTA permanece externo e intocado neste notebook.
        ''') ,
        gpu_gate,
        md(r'''
        ## 1. Ambiente fixado e restauração íntegra pelo TAR

        Execute em GPU. Esta restauração é idêntica à dos Notebooks 08–09 e verifica o pacote,
        o manifest e os conjuntos exatos de imagens/labels antes de qualquer inferência.
        ''') ,
        restore_dataset,
        md(r'''
        ## 2. Contratos finais e gate pré-teste

        Esta seção não executa o modelo no teste. Ela confirma o contrato imutável dos três
        checkpoints, o resultado Hough já congelado, a composição exata do split e a ausência
        de um evento YOLO final anterior.
        ''') ,
        code(r'''
        from datetime import datetime, timezone

        EXPECTED_FINAL_CONTRACT_SHA256='0a8d3537969dcb033dc8cdfa1a9d63dc24de4cc98e212c42fab62621126890a7'
        EXPECTED_FINAL_CONFIG_SHA256='457b70c8e96176827bf4a3f755963e6c6c3e607bec7d8133233518b26be27870'
        EXPECTED_HOUGH_CONFIG_SHA256='14ef0ea56a2e86bf1581225c4b9a0733d482834e5f7d3fab18b3eaf76f03a2e5'
        EXPECTED_HOUGH_RESULT_SHA256='5b476de5a6015b6c38758a43e8c46719050bc036d28362d5f7d43f13e8ffa3b8'

        DRIVE_FINAL=BACKUP_DIR/'experiments/yolo_final_multiseed_v3'/RUN_ID
        DRIVE_TEST=BACKUP_DIR/'experiments/yolo_protected_test_v3'/RUN_ID
        DRIVE_RUN=BACKUP_DIR/'data/synthetic_runs'/RUN_ID
        CONTRATO_PATH=DRIVE_FINAL/'contrato_modelos_finais_v3.json'
        CONFIG_FINAL_PATH=DRIVE_FINAL/'protocolo_treino_final_multiseed_v3.json'
        PONTEIRO_PATH=BACKUP_DIR/'experiments/yolo_final_multiseed_v3/LATEST_FINAL_MODELS_V3.json'
        LOCK_PATH=DRIVE_TEST/'TRAVA_EVENTO_TESTE_YOLO_V3.json'

        def verificar_hash_canonico(registro,campo_hash):
            copia=dict(registro);observado=copia.pop(campo_hash)
            texto=json.dumps(copia,sort_keys=True,separators=(',',':'))
            calculado=hashlib.sha256(texto.encode()).hexdigest()
            assert calculado==observado,f'Hash canônico divergente em {campo_hash}'
            return observado

        contrato=json.loads(CONTRATO_PATH.read_text(encoding='utf-8'))
        CONFIG_FINAL_TEST=json.loads(CONFIG_FINAL_PATH.read_text(encoding='utf-8'))
        ponteiro=json.loads(PONTEIRO_PATH.read_text(encoding='utf-8'))
        HASH_CONTRATO_FINAL=verificar_hash_canonico(contrato,'sha256_contrato_modelos_finais')
        HASH_CONFIG_FINAL=verificar_hash_canonico(CONFIG_FINAL_TEST,'sha256_config_final')
        assert HASH_CONTRATO_FINAL==EXPECTED_FINAL_CONTRACT_SHA256
        assert HASH_CONFIG_FINAL==EXPECTED_FINAL_CONFIG_SHA256==contrato['sha256_config_final']
        assert contrato['aprovado'] is True and contrato['run_id']==RUN_ID
        assert contrato['sha256_manifest']==HASH_MANIFEST
        assert contrato['threshold_confianca']==0.45
        assert contrato['seeds_avaliadas']==[20260823,20260824,20260825]
        assert contrato['teste_sintetico_lido'] is False and contrato['asta_lido'] is False
        assert contrato['plano_teste_protegido']['evento_unico'] is True
        assert contrato['plano_teste_protegido']['avaliar_tres_checkpoints'] is True
        assert contrato['plano_teste_protegido']['selecionar_seed_com_teste'] is False
        assert contrato['plano_teste_protegido']['recalibrar_threshold_com_teste'] is False
        assert CONFIG_FINAL_TEST['threshold_confianca_fixo']==contrato['threshold_confianca']==0.45
        assert CONFIG_FINAL_TEST['matching']=={'iou':0.3,'angulo':15.0,'distancia':14.0,'cobertura':0.5,'modo':'combinado'}
        assert ponteiro['run_id']==RUN_ID
        assert ponteiro['sha256_manifest']==HASH_MANIFEST
        assert ponteiro['sha256_contrato_modelos_finais']==HASH_CONTRATO_FINAL
        assert ponteiro['checkpoint_operacional_seed']==contrato['checkpoint_operacional_seed']==20260824

        MODELOS_LOCAL=Path('/content/modelos_finais_yolo_v3')/RUN_ID
        MODELOS_LOCAL.mkdir(parents=True,exist_ok=True)
        checkpoints=[]
        for item in contrato['checkpoints']:
            seed=int(item['seed']);remoto=BACKUP_DIR/item['best_pt_relativo_drive']
            assert remoto.is_file() and sha256_arquivo(remoto)==item['sha256_best_pt']
            local=MODELOS_LOCAL/f'best_seed_{seed}.pt'
            if not local.exists():shutil.copy2(remoto,local)
            assert sha256_arquivo(local)==item['sha256_best_pt']
            checkpoints.append({**item,'seed':seed,'local':str(local)})
        assert [x['seed'] for x in checkpoints]==[20260823,20260824,20260825]

        hough_dir=DRIVE_RUN/'hough_v3'
        hough_config=json.loads((hough_dir/'config_congelada_v3.json').read_text(encoding='utf-8'))
        hough_lock=json.loads((hough_dir/'TRAVA_TESTE_FINAL.json').read_text(encoding='utf-8'))
        hough_result_path=hough_dir/'resultado_test_v3.csv'
        assert hough_config['sha256_config']==EXPECTED_HOUGH_CONFIG_SHA256
        hough_sem_hash=dict(hough_config);hough_hash_observado=hough_sem_hash.pop('sha256_config')
        hough_hash_calculado=hashlib.sha256(json.dumps(hough_sem_hash,sort_keys=True).encode()).hexdigest()
        assert hough_hash_calculado==hough_hash_observado
        assert hough_lock['sha256_config']==EXPECTED_HOUGH_CONFIG_SHA256
        assert hough_lock['sha256_resultado']==EXPECTED_HOUGH_RESULT_SHA256
        assert sha256_arquivo(hough_result_path)==EXPECTED_HOUGH_RESULT_SHA256
        assert hough_config['sha256_manifest']==HASH_MANIFEST==hough_lock['sha256_manifest']

        df_test=df[df.split=='test'].copy().sort_values('id').reset_index(drop=True)
        ids_test=set(df_test.id.astype(str));pasta_test=RUN/'images/test'
        assert len(df_test)==900 and (df_test.tipo=='positivo').sum()==450 and (df_test.tipo=='negativo').sum()==450
        assert df_test.sha256_fundo.nunique()==40
        assert {p.stem for p in pasta_test.glob('*.png')}==ids_test
        assert {p.stem for p in (RUN/'labels/test').glob('*.txt')}==ids_test
        fundos={s:set(df.loc[df.split==s,'sha256_fundo']) for s in ['train','val','test']}
        assert fundos['train'].isdisjoint(fundos['val']) and fundos['train'].isdisjoint(fundos['test'])
        assert fundos['val'].isdisjoint(fundos['test'])

        lock_existente=json.loads(LOCK_PATH.read_text(encoding='utf-8')) if LOCK_PATH.exists() else None
        if lock_existente is None:
            status_pre='TESTE_YOLO_PRONTO_PARA_EVENTO_UNICO'
        elif lock_existente.get('status')=='concluido':
            status_pre='BLOQUEADO_TESTE_YOLO_JA_CONCLUIDO'
        else:
            status_pre='EVENTO_YOLO_INCOMPLETO_EXIGE_RETOMADA_CONTROLADA'
        gate_pre={
            'status':status_pre,'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
            'sha256_contrato_modelos_finais':HASH_CONTRATO_FINAL,
            'sha256_config_final':HASH_CONFIG_FINAL,
            'seeds':[x['seed'] for x in checkpoints],'imagens_test':len(df_test),
            'positivas_test':int((df_test.tipo=='positivo').sum()),
            'negativas_test':int((df_test.tipo=='negativo').sum()),
            'fundos_test':int(df_test.sha256_fundo.nunique()),
            'sha256_hough_config':hough_config['sha256_config'],
            'sha256_hough_resultado':hough_lock['sha256_resultado'],
            'evento_existente':lock_existente
        }
        print(json.dumps(gate_pre,indent=2,ensure_ascii=False))
        if status_pre=='TESTE_YOLO_PRONTO_PARA_EVENTO_UNICO':
            print('Gate pré-teste aprovado. Não habilite o evento antes de compartilhar este relatório.')
        ''') ,
        md(r'''
        ## 3. Funções congeladas de matching e estatística

        As funções abaixo reproduzem o matching combinado fixado no Notebook 09. Há um único
        GT por imagem positiva; previsões adicionais contam como falsos positivos. O bootstrap
        reamostra os 40 fundos e calcula a média de F1 das três seeds em cada réplica.
        ''') ,
        code(r'''
        def parametros_linha(p0,p1):
            p0,p1=np.asarray(p0,dtype=float),np.asarray(p1,dtype=float)
            v=p1-p0;comp=float(np.linalg.norm(v));direcao=v/comp if comp>0 else np.array([1.,0.])
            return (p0+p1)/2,direcao,comp

        def centerline(p):
            p=np.asarray(p,dtype=float);d01=np.linalg.norm(p[0]-p[1]);d12=np.linalg.norm(p[1]-p[2])
            return ((p[0]+p[3])/2,(p[1]+p[2])/2) if d01>=d12 else ((p[0]+p[1])/2,(p[2]+p[3])/2)

        def iou_poligonos(a,b):
            a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
            aa=abs(float(cv2.contourArea(a)));ab=abs(float(cv2.contourArea(b)))
            inter=float(cv2.intersectConvexConvex(a,b)[0]);uniao=aa+ab-inter
            return inter/uniao if uniao>0 else 0.0

        def metricas_match(pred,gt):
            p0,p1=centerline(pred);g0,g1=centerline(gt)
            cp,dp,_=parametros_linha(p0,p1);cg,dg,lg=parametros_linha(g0,g1)
            ang=float(np.degrees(np.arccos(np.clip(abs(np.dot(dp,dg)),-1,1))))
            v=cp-cg;dist=float(np.linalg.norm(v-np.dot(v,dg)*dg))
            t=sorted([np.dot(p0-cg,dg),np.dot(p1-cg,dg)])
            inter=max(0,min(t[1],lg/2)-max(t[0],-lg/2));cob=float(inter/lg) if lg>0 else 0.0
            return {'iou':iou_poligonos(pred,gt),'angulo':ang,'distancia':dist,'cobertura':cob}

        def ler_gt(linha):
            tokens=(RUN/'labels'/linha.split/(linha.id+'.txt')).read_text(encoding='utf-8').split()
            return None if not tokens else np.asarray([float(x) for x in tokens[1:9]]).reshape(4,2)*np.array([512.,512.])

        def score_match(m,lim):
            miou=m['iou']>=lim['iou']
            mgeo=(m['angulo']<=lim['angulo'] and m['distancia']<=lim['distancia'] and m['cobertura']>=lim['cobertura'])
            if not (miou or mgeo):return None
            siou=m['iou']/lim['iou'] if miou else -np.inf
            sgeo=((1-m['angulo']/lim['angulo'])+(1-m['distancia']/lim['distancia'])+m['cobertura']) if mgeo else -np.inf
            return max(siou,sgeo)

        def avaliar_predicoes(dados,predicoes,threshold):
            saida=[];lim_match={k:CONFIG_FINAL_TEST['matching'][k] for k in ['iou','angulo','distancia','cobertura']}
            for linha in dados.itertuples():
                grupo=predicoes[(predicoes.id==linha.id)&(predicoes.conf>=threshold)].copy()
                candidatos=[]
                for _,g in grupo.iterrows():
                    poly=g[['x1','y1','x2','y2','x3','y3','x4','y4']].to_numpy(float).reshape(4,2)
                    candidatos.append((poly,float(g.conf)))
                gt=ler_gt(linha);aceitos=[]
                if gt is not None:
                    for poly,conf in candidatos:
                        met=metricas_match(poly,gt);score=score_match(met,lim_match)
                        if score is not None:aceitos.append((score,conf,met))
                melhor=max(aceitos,key=lambda x:(x[0],x[1])) if aceitos else None
                match=melhor is not None
                saida.append({
                    'id':linha.id,'split':linha.split,'tipo':linha.tipo,'subtipo':linha.subtipo,
                    'fundo_origem':linha.fundo_origem,'sha256_fundo':linha.sha256_fundo,
                    'tp':int(gt is not None and match),'fp':max(0,len(candidatos)-int(gt is not None and match)),
                    'fn':int(gt is not None and not match),'n_predicoes':len(candidatos),
                    'conf_match':melhor[1] if melhor else np.nan,
                    'iou_match':melhor[2]['iou'] if melhor else np.nan,
                    'angulo_match':melhor[2]['angulo'] if melhor else np.nan,
                    'distancia_match':melhor[2]['distancia'] if melhor else np.nan,
                    'cobertura_match':melhor[2]['cobertura'] if melhor else np.nan
                })
            return pd.DataFrame(saida)

        def metricas_contagens(tp,fp,fn):
            p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
            return p,r,2*p*r/(p+r) if p+r else 0.0

        def resumir_resultado(resultado):
            tp=int(resultado.tp.sum());fp=int(resultado.fp.sum());fn=int(resultado.fn.sum())
            p,r,f1=metricas_contagens(tp,fp,fn);neg=resultado[resultado.tipo=='negativo']
            return {'TP':tp,'FP':fp,'FN':fn,'precisao':p,'recall':r,'f1':f1,
                    'fp_por_imagem_negativa':float(neg.fp.mean()),
                    'taxa_frames_negativos_com_fp':float((neg.fp>0).mean())}

        assert abs(metricas_match(np.array([[0,0],[100,0],[100,4],[0,4]],float),np.array([[0,0],[100,0],[100,4],[0,4]],float))['angulo'])<1e-9
        print('Funções geométricas e estatísticas congeladas.')
        ''') ,
        md(r'''
        ## 4. Evento único, retomável e protegido

        Primeiro execute esta célula com as duas chaves `False`. Depois de revisar e compartilhar
        o gate pré-teste, altere somente `EXECUTAR_TESTE_PROTEGIDO=True`. Use a chave de retomada
        apenas se o mesmo evento tiver sido interrompido e revisado; seeds já concluídas por hash
        serão reutilizadas. Um evento concluído é recusado incondicionalmente.
        ''') ,
        code(r'''
        EXECUTAR_TESTE_PROTEGIDO=False
        RETOMAR_EVENTO_INCOMPLETO=False

        if not EXECUTAR_TESTE_PROTEGIDO:
            print('Evento único bloqueado. Compartilhe primeiro o gate pré-teste da seção 2.')
        else:
            import gc,os,uuid,ultralytics
            from ultralytics import YOLO
            assert ultralytics.__version__=='8.4.127'
            DRIVE_TEST.mkdir(parents=True,exist_ok=True)
            def atualizar_lock(evento_atual):
                temporario=LOCK_PATH.with_name(f"{LOCK_PATH.stem}.{evento_atual['event_id']}.tmp")
                temporario.write_text(json.dumps(evento_atual,indent=2),encoding='utf-8')
                os.replace(temporario,LOCK_PATH)
            agora=datetime.now(timezone.utc).isoformat()
            if LOCK_PATH.exists():
                evento=json.loads(LOCK_PATH.read_text(encoding='utf-8'))
                assert evento['sha256_contrato_modelos_finais']==HASH_CONTRATO_FINAL
                assert evento['sha256_config_final']==HASH_CONFIG_FINAL
                assert evento['sha256_manifest']==HASH_MANIFEST
                if evento.get('status')=='concluido':
                    raise RuntimeError('Teste YOLO final já concluído. Nova inferência é proibida.')
                if not RETOMAR_EVENTO_INCOMPLETO:
                    raise RuntimeError('Há evento incompleto. Revise a trava e habilite somente RETOMAR_EVENTO_INCOMPLETO.')
            else:
                evento={'status':'em_execucao','event_id':str(uuid.uuid4()),'iniciado_em_utc':agora,
                        'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                        'sha256_contrato_modelos_finais':HASH_CONTRATO_FINAL,
                        'sha256_config_final':HASH_CONFIG_FINAL,
                        'seeds_previstas':[x['seed'] for x in checkpoints],'seeds_concluidas':[]}
                LOCK_PATH.parent.mkdir(parents=True,exist_ok=True)
                with LOCK_PATH.open('x',encoding='utf-8') as arq:json.dump(evento,arq,indent=2)

            EVENT_DIR=DRIVE_TEST/f"evento_{evento['event_id']}";EVENT_DIR.mkdir(parents=True,exist_ok=True)
            registros_teste=[]
            for item in checkpoints:
                seed=int(item['seed']);SEED_DRIVE=EVENT_DIR/f'seed_{seed}'
                registro_path=SEED_DRIVE/'conclusao_teste_seed.json'
                if registro_path.exists():
                    registro=json.loads(registro_path.read_text(encoding='utf-8'))
                    assert registro['sha256_contrato_modelos_finais']==HASH_CONTRATO_FINAL
                    assert registro['sha256_checkpoint']==item['sha256_best_pt']
                    for nome,campo in [('predicoes_test.csv','sha256_predicoes'),('resultado_test.csv','sha256_resultado')]:
                        assert sha256_arquivo(SEED_DRIVE/nome)==registro[campo]
                    registros_teste.append(registro);print(f'Seed {seed} já concluída neste evento e reutilizada.')
                    continue

                LOCAL_SEED=Path('/content/yolo_protected_test_v3')/evento['event_id']/f'seed_{seed}'
                LOCAL_SEED.mkdir(parents=True,exist_ok=True)
                gc.collect();torch.cuda.empty_cache();modelo=YOLO(item['local'])
                preds=[];tempos=[];ids=[]
                resultados=modelo.predict(source=str(pasta_test),stream=True,batch=1,
                    imgsz=CONFIG_FINAL_TEST['treino']['imgsz'],conf=CONFIG_FINAL_TEST['predicao_val']['conf_min'],
                    iou=CONFIG_FINAL_TEST['predicao_val']['iou_nms'],max_det=CONFIG_FINAL_TEST['predicao_val']['max_det'],
                    device=0,verbose=False)
                for resultado in tqdm(resultados,total=len(ids_test),desc=f'TESTE seed {seed}',unit='img'):
                    id_amostra=Path(resultado.path).stem;ids.append(id_amostra)
                    tempos.append(float(resultado.speed.get('inference',np.nan)))
                    if resultado.obb is None:continue
                    for poly,conf in zip(resultado.obb.xyxyxyxy.cpu().numpy(),resultado.obb.conf.cpu().numpy()):
                        plano=poly.reshape(-1);preds.append({'id':id_amostra,'conf':float(conf),
                            **{f'{e}{i+1}':float(plano[2*i+(0 if e=='x' else 1)]) for i in range(4) for e in ['x','y']}})
                assert len(ids)==len(ids_test) and len(set(ids))==len(ids) and set(ids)==ids_test
                pred=pd.DataFrame(preds,columns=['id','conf','x1','y1','x2','y2','x3','y3','x4','y4'])
                pred_path=LOCAL_SEED/'predicoes_test.csv';pred.to_csv(pred_path,index=False)
                por_imagem=avaliar_predicoes(df_test,pred,CONFIG_FINAL_TEST['threshold_confianca_fixo'])
                resultado_path=LOCAL_SEED/'resultado_test.csv';por_imagem.to_csv(resultado_path,index=False)
                resumo=resumir_resultado(por_imagem)
                metricas=modelo.val(data=str(dataset_yaml),split='test',imgsz=CONFIG_FINAL_TEST['treino']['imgsz'],
                    batch=CONFIG_FINAL_TEST['treino']['batch'],device=0,workers=CONFIG_FINAL_TEST['treino']['workers'],
                    plots=False,verbose=False,project=str(LOCAL_SEED/'ultralytics'),name='test',exist_ok=True)
                registro={'protocolo':'yolo_test_protegido_v3','event_id':evento['event_id'],'seed':seed,
                    'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                    'sha256_contrato_modelos_finais':HASH_CONTRATO_FINAL,
                    'sha256_config_final':HASH_CONFIG_FINAL,
                    'sha256_checkpoint':item['sha256_best_pt'],'threshold_confianca':CONFIG_FINAL_TEST['threshold_confianca_fixo'],
                    **resumo,'map50_obb':float(metricas.box.map50),'map50_95_obb':float(metricas.box.map),
                    'inferencia_ms_media':float(np.nanmean(tempos)),
                    'sha256_predicoes':sha256_arquivo(pred_path),'sha256_resultado':sha256_arquivo(resultado_path),
                    'concluido_em_utc':datetime.now(timezone.utc).isoformat()}
                registro_local=LOCAL_SEED/'conclusao_teste_seed.json'
                registro_local.write_text(json.dumps(registro,indent=2),encoding='utf-8')
                SEED_DRIVE.mkdir(parents=True,exist_ok=True)
                arquivos_seed=sorted(p for p in LOCAL_SEED.rglob('*') if p.is_file() and p!=registro_local)
                for arquivo in tqdm(arquivos_seed,desc=f'Backup teste seed {seed}',unit='arq'):
                    alvo=SEED_DRIVE/arquivo.relative_to(LOCAL_SEED);alvo.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(arquivo,alvo)
                # O registro é o commit da seed e só pode existir depois de todos os artefatos.
                shutil.copy2(registro_local,registro_path)
                assert sha256_arquivo(SEED_DRIVE/'predicoes_test.csv')==registro['sha256_predicoes']
                assert sha256_arquivo(SEED_DRIVE/'resultado_test.csv')==registro['sha256_resultado']
                registros_teste.append(registro)
                evento['seeds_concluidas']=sorted({*evento.get('seeds_concluidas',[]),seed})
                atualizar_lock(evento)
                del resultados,resultado,metricas,modelo;gc.collect();torch.cuda.empty_cache()

            assert sorted(r['seed'] for r in registros_teste)==[20260823,20260824,20260825]
            CONS_LOCAL=Path('/content/yolo_protected_test_v3')/evento['event_id']/'consolidado'
            CONS_LOCAL.mkdir(parents=True,exist_ok=True)
            tabelas=[]
            for registro in registros_teste:
                tab=pd.read_csv(EVENT_DIR/f"seed_{registro['seed']}"/'resultado_test.csv');tab['seed']=registro['seed'];tabelas.append(tab)
            resultados_yolo=pd.concat(tabelas,ignore_index=True)
            assert len(resultados_yolo)==2700 and set(resultados_yolo.id)==ids_test
            comparacao_seeds=pd.DataFrame(registros_teste).sort_values('seed')
            colunas_metricas=['precisao','recall','f1','fp_por_imagem_negativa','taxa_frames_negativos_com_fp','map50_obb','map50_95_obb','inferencia_ms_media']
            agregado=comparacao_seeds[colunas_metricas].agg(['mean','std','min','max']).T.reset_index(names='metrica')

            positivos=resultados_yolo[resultados_yolo.tipo=='positivo']
            recall_seed_subtipo=(positivos.groupby(['seed','subtipo'])[['tp','fn']].sum().reset_index())
            recall_seed_subtipo['recall']=recall_seed_subtipo.tp/(recall_seed_subtipo.tp+recall_seed_subtipo.fn)
            recall_subtipo=recall_seed_subtipo.groupby('subtipo').recall.agg(['mean','std','min','max']).reset_index()
            negativos=resultados_yolo[resultados_yolo.tipo=='negativo'].copy()
            neg_seed_subtipo=negativos.groupby(['seed','subtipo']).agg(
                fp_por_imagem=('fp','mean'),taxa_frames_com_fp=('fp',lambda s:float((s>0).mean()))).reset_index()
            neg_subtipo=neg_seed_subtipo.groupby('subtipo')[['fp_por_imagem','taxa_frames_com_fp']].agg(['mean','std','min','max'])
            geometria=(positivos[positivos.tp==1].groupby('seed')[['angulo_match','distancia_match','cobertura_match','iou_match']].mean())

            hough=pd.read_csv(hough_result_path).sort_values('id').reset_index(drop=True)
            assert set(hough.id.astype(str))==ids_test and len(hough)==900
            htp=int(hough.tp_combinado.sum());hfp=int(hough.fp_combinado.sum());hfn=int(hough.fn_combinado.sum())
            hp,hr,hf1=metricas_contagens(htp,hfp,hfn);hneg=hough[hough.tipo=='negativo']
            resumo_hough={'metodo':'Hough congelado','precisao':hp,'recall':hr,'f1':hf1,
                'fp_por_imagem_negativa':float(hneg.fp_combinado.mean()),
                'taxa_frames_negativos_com_fp':float((hneg.fp_combinado>0).mean()),'map50_95_obb':None}
            medias=comparacao_seeds[colunas_metricas].mean()
            resumo_yolo={'metodo':'YOLO11n-OBB média 3 seeds','precisao':float(medias.precisao),
                'recall':float(medias.recall),'f1':float(medias.f1),
                'fp_por_imagem_negativa':float(medias.fp_por_imagem_negativa),
                'taxa_frames_negativos_com_fp':float(medias.taxa_frames_negativos_com_fp),
                'map50_95_obb':float(medias.map50_95_obb)}
            comparacao_metodos=pd.DataFrame([resumo_hough,resumo_yolo])

            fundos_ids=sorted(df_test.sha256_fundo.unique());assert len(fundos_ids)==40
            yolo_fundos={int(seed):g.groupby('sha256_fundo')[['tp','fp','fn']].sum()
                         for seed,g in resultados_yolo.groupby('seed')}
            hough_fundos=hough.groupby('sha256_fundo')[['tp_combinado','fp_combinado','fn_combinado']].sum()
            assert all(set(t.index)==set(fundos_ids) for t in yolo_fundos.values()) and set(hough_fundos.index)==set(fundos_ids)
            rng=np.random.default_rng(20260823);boots=[]
            for _ in tqdm(range(2000),desc='Bootstrap YOLO × Hough por fundo',unit='rep'):
                amostra=rng.choice(fundos_ids,size=len(fundos_ids),replace=True);f1_seeds=[]
                for seed in sorted(yolo_fundos):
                    s=yolo_fundos[seed].loc[amostra].sum();f1_seeds.append(metricas_contagens(s.tp,s.fp,s.fn)[2])
                hs=hough_fundos.loc[amostra].sum();hf=metricas_contagens(hs.tp_combinado,hs.fp_combinado,hs.fn_combinado)[2]
                ym=float(np.mean(f1_seeds));boots.append({'f1_yolo_media_seeds':ym,'f1_hough':hf,'delta_f1_yolo_menos_hough':ym-hf})
            bootstrap=pd.DataFrame(boots)
            intervalos={c:[float(x) for x in np.percentile(bootstrap[c],[2.5,97.5])] for c in bootstrap.columns}

            comparacao_seeds.to_csv(CONS_LOCAL/'comparacao_seeds_test_v3.csv',index=False)
            agregado.to_csv(CONS_LOCAL/'variabilidade_seeds_test_v3.csv',index=False)
            recall_subtipo.to_csv(CONS_LOCAL/'recall_por_subtipo_test_v3.csv',index=False)
            neg_subtipo.to_csv(CONS_LOCAL/'negativos_por_subtipo_test_v3.csv')
            geometria.to_csv(CONS_LOCAL/'geometria_matches_test_v3.csv')
            comparacao_metodos.to_csv(CONS_LOCAL/'comparacao_yolo_hough_test_v3.csv',index=False)
            bootstrap.to_csv(CONS_LOCAL/'bootstrap_yolo_hough_por_fundo_v3.csv',index=False)
            resumo_final={'protocolo':'yolo_test_protegido_v3','status':'TESTE_YOLO_PROTEGIDO_CONCLUIDO',
                'event_id':evento['event_id'],'run_id':RUN_ID,'sha256_manifest':HASH_MANIFEST,
                'sha256_contrato_modelos_finais':HASH_CONTRATO_FINAL,
                'sha256_config_final':HASH_CONFIG_FINAL,
                'seeds':[20260823,20260824,20260825],
                'threshold_confianca':CONFIG_FINAL_TEST['threshold_confianca_fixo'],
                'resultado_primario_yolo_media_seeds':resumo_yolo,
                'variabilidade_f1':{'media':float(comparacao_seeds.f1.mean()),'desvio_padrao_amostral':float(comparacao_seeds.f1.std()),
                    'min':float(comparacao_seeds.f1.min()),'max':float(comparacao_seeds.f1.max())},
                'hough_congelado':resumo_hough,'bootstrap_por_fundo':{'seed':20260823,'replicacoes':2000,
                    'n_fundos':40,'ic95_percentil':intervalos},
                'sha256_tabela_seeds':sha256_arquivo(CONS_LOCAL/'comparacao_seeds_test_v3.csv'),
                'sha256_comparacao_yolo_hough':sha256_arquivo(CONS_LOCAL/'comparacao_yolo_hough_test_v3.csv'),
                'sha256_bootstrap':sha256_arquivo(CONS_LOCAL/'bootstrap_yolo_hough_por_fundo_v3.csv'),
                'selecionou_seed_com_teste':False,'recalibrou_threshold_com_teste':False,'asta_lido':False}
            texto=json.dumps(resumo_final,sort_keys=True,separators=(',',':'))
            resumo_final['sha256_resultado_final']=hashlib.sha256(texto.encode()).hexdigest()
            resumo_path=CONS_LOCAL/'resumo_final_yolo_test_v3.json';resumo_path.write_text(json.dumps(resumo_final,indent=2),encoding='utf-8')
            copiar_arvore_com_progresso(CONS_LOCAL,EVENT_DIR/'consolidado','Backup consolidação teste YOLO')
            assert sha256_arquivo(EVENT_DIR/'consolidado/resumo_final_yolo_test_v3.json')==sha256_arquivo(resumo_path)
            evento.update({'status':'concluido','concluido_em_utc':datetime.now(timezone.utc).isoformat(),
                'seeds_concluidas':[20260823,20260824,20260825],
                'sha256_resultado_final':resumo_final['sha256_resultado_final'],
                'resumo_relativo':str((EVENT_DIR/'consolidado/resumo_final_yolo_test_v3.json').relative_to(BACKUP_DIR))})
            atualizar_lock(evento)
            print(comparacao_seeds[['seed','precisao','recall','f1','taxa_frames_negativos_com_fp','map50_95_obb','inferencia_ms_media']])
            print(comparacao_metodos)
            print(json.dumps(resumo_final,indent=2,ensure_ascii=False))
            print('TESTE YOLO PROTEGIDO CONCLUÍDO E TRAVADO. ASTA permanece intocado.')
        ''') ,
        md(r'''
        ## 5. Verificação pós-evento (somente leitura)

        Esta célula não executa inferência. Depois do evento, confirma a trava final, o hash do
        resumo e imprime o relatório persistido no Drive.
        ''') ,
        code(r'''
        if not LOCK_PATH.exists():
            print('Evento ainda não executado.')
        else:
            trava_final=json.loads(LOCK_PATH.read_text(encoding='utf-8'))
            print(json.dumps(trava_final,indent=2,ensure_ascii=False))
            if trava_final.get('status')=='concluido':
                resumo_remoto=BACKUP_DIR/trava_final['resumo_relativo']
                resumo=json.loads(resumo_remoto.read_text(encoding='utf-8'))
                copia=dict(resumo);observado=copia.pop('sha256_resultado_final')
                calculado=hashlib.sha256(json.dumps(copia,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert calculado==observado==trava_final['sha256_resultado_final']
                assert resumo['sha256_contrato_modelos_finais']==EXPECTED_FINAL_CONTRACT_SHA256
                assert resumo['sha256_config_final']==EXPECTED_FINAL_CONFIG_SHA256
                assert resumo['selecionou_seed_com_teste'] is False and resumo['recalibrou_threshold_com_teste'] is False
                assert resumo['asta_lido'] is False
                print(json.dumps(resumo,indent=2,ensure_ascii=False))
                print('BACKUP TESTE YOLO PROTEGIDO APROVADO.')
        ''') ,
    ]
    return write_notebook('10_teste_sintetico_yolo_protegido_comparacao_hough_v3.ipynb', cells)


def build_11():
    cells = [
        md(r'''
        # 11 — Auditoria externa ASTA antes da inferência (v3)

        Este notebook audita o conjunto real ASTA sem executar YOLO, Hough ou qualquer outro
        detector. A auditoria confirma a origem, o inventário exato, os pares imagem–máscara,
        dimensões, hashes, legibilidade, valores das máscaras, componentes conectados e casos
        que precisam de revisão visual antes de converter máscaras em objetos geométricos.

        **Regra de proteção:** `data/raw/asta_real` é somente leitura. Checkpoints e relatórios
        são gravados exclusivamente em `experiments/asta_external_v3/audit`. O protocolo de
        tiles, agrupamento de componentes, matching e métricas será congelado somente depois
        desta auditoria; portanto, este notebook não abre o teste externo.

        Execute em **CPU**. A etapa pesada percorre cerca de 18 GB, mostra progresso e é
        retomável por checkpoint. Não copie a pasta inteira para `/content`: apenas um par é
        temporariamente preparado no disco local por vez.
        ''') ,
        md(r'''
        ## 0. Ambiente, Drive e caminhos protegidos

        Esta seção monta o Drive e declara utilitários. Nenhum arquivo bruto é alterado.
        ''') ,
        code(r'''
        !pip install -q remotezip

        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)

        from pathlib import Path
        from datetime import datetime, timezone
        from remotezip import RemoteZip
        from tqdm.auto import tqdm
        from PIL import Image, ImageDraw
        import cv2, hashlib, json, math, os, shutil, time, warnings, zipfile
        import numpy as np
        import pandas as pd
        import requests

        # 10.560² = 111.513.600 pixels. Mantemos um teto explícito em vez de desativar
        # globalmente a proteção contra imagens desproporcionalmente grandes.
        Image.MAX_IMAGE_PIXELS = 200_000_000

        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        ASTA_RAW=BACKUP_DIR/'data/raw/asta_real'
        AUDIT_DRIVE=BACKUP_DIR/'experiments/asta_external_v3/audit'
        AUDIT_LOCAL=Path('/content/asta_audit_v3')
        STAGING=AUDIT_LOCAL/'staging_um_par'
        PREVIEWS_DRIVE=AUDIT_DRIVE/'previews_checkpoint'
        RECORD_ID='11642424'
        DOI='10.5281/zenodo.11642424'
        URL_ZIP='https://zenodo.org/records/11642424/files/Processed.zip?download=1'
        URL_API='https://zenodo.org/api/records/11642424'
        EXPECTED_ZIP_MD5='37c6c2096c74ce22778d3baad056d0b9'
        AREA_MINIMA=15
        CHECKPOINT_EVERY=5

        assert ASTA_RAW.is_dir(),f'Pasta ASTA ausente no Drive: {ASTA_RAW}'
        AUDIT_DRIVE.mkdir(parents=True,exist_ok=True)
        AUDIT_LOCAL.mkdir(parents=True,exist_ok=True)
        STAGING.mkdir(parents=True,exist_ok=True)
        PREVIEWS_DRIVE.mkdir(parents=True,exist_ok=True)

        def agora():return datetime.now().strftime('%H:%M:%S')
        def sha256_arquivo(caminho,chunk=8*1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as arq:
                for bloco in iter(lambda:arq.read(chunk),b''):h.update(bloco)
            return h.hexdigest()
        def sha256_canonico(objeto):
            texto=json.dumps(objeto,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            return hashlib.sha256(texto.encode('utf-8')).hexdigest()
        def gravar_json_atomico(caminho,objeto):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            temporario=caminho.with_name(caminho.name+'.tmp')
            temporario.write_text(json.dumps(objeto,indent=2,ensure_ascii=False),encoding='utf-8')
            os.replace(temporario,caminho)
        def gravar_csv_atomico(caminho,tabela):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            temporario=caminho.with_name(caminho.name+'.tmp')
            tabela.to_csv(temporario,index=False)
            os.replace(temporario,caminho)

        print(f'[{agora()}] ASTA localizado. Runtime CPU é suficiente.')
        print('Fonte bruta protegida (somente leitura):',ASTA_RAW)
        print('Saídas da auditoria:',AUDIT_DRIVE)
        ''') ,
        md(r'''
        ## 1. Inventário oficial e gate leve

        O índice do `Processed.zip` é consultado por requisições parciais; o arquivo de 18 GB
        não é baixado. O gate exige os 178 pares oficiais, compara os nomes com o Drive e valida
        o registro Zenodo. Ainda não decodifica as imagens nem calcula seus SHA-256.
        ''') ,
        code(r'''
        def nome_imagem_para_mascara(nome):
            sufixo='.fits_full.png'
            if not nome.endswith(sufixo):raise ValueError(f'Nome de imagem inesperado: {nome}')
            return nome[:-len(sufixo)]+'_mask.png'

        print(f'[{agora()}] Lendo índice oficial do Zenodo...')
        with RemoteZip(URL_ZIP) as zf:
            membros_oficiais=zf.namelist()
        imagens_oficiais=sorted(Path(n).name for n in membros_oficiais if n.endswith('_full.png'))
        mascaras_oficiais=sorted(Path(n).name for n in membros_oficiais if n.endswith('_mask.png'))
        set_imagens_oficiais,set_mascaras_oficiais=set(imagens_oficiais),set(mascaras_oficiais)
        pares_oficiais=[]
        for nome_img in imagens_oficiais:
            nome_mask=nome_imagem_para_mascara(nome_img)
            if nome_mask in set_mascaras_oficiais:pares_oficiais.append((nome_img,nome_mask))

        resposta=requests.get(URL_API,timeout=60);resposta.raise_for_status();zenodo=resposta.json()
        arquivo_zip=next(f for f in zenodo['files'] if f['key']=='Processed.zip')
        checksum_zip=arquivo_zip['checksum'].split(':',1)[-1].lower()
        licenca=(zenodo.get('metadata',{}).get('license') or {}).get('id')

        nomes_locais={p.name for p in ASTA_RAW.iterdir() if p.is_file()}
        imagens_locais={n for n in nomes_locais if n.endswith('_full.png')}
        mascaras_locais={n for n in nomes_locais if n.endswith('_mask.png')}
        faltantes_imagem=sorted(set_imagens_oficiais-imagens_locais)
        faltantes_mascara=sorted(set_mascaras_oficiais-mascaras_locais)
        extras_imagem=sorted(imagens_locais-set_imagens_oficiais)
        extras_mascara=sorted(mascaras_locais-set_mascaras_oficiais)

        erros=[];avisos=[]
        if len(imagens_oficiais)!=178 or len(mascaras_oficiais)!=178 or len(pares_oficiais)!=178:
            erros.append('O índice oficial não contém exatamente 178 pares.')
        if checksum_zip!=EXPECTED_ZIP_MD5:erros.append('MD5 oficial do Processed.zip divergiu do registro esperado.')
        if faltantes_imagem:erros.append(f'{len(faltantes_imagem)} imagens oficiais ausentes no Drive.')
        if faltantes_mascara:erros.append(f'{len(faltantes_mascara)} máscaras oficiais ausentes no Drive.')
        if extras_imagem:erros.append(f'{len(extras_imagem)} imagens *_full.png não pertencem ao índice oficial.')
        if extras_mascara:erros.append(f'{len(extras_mascara)} máscaras *_mask.png não pertencem ao índice oficial.')
        if licenca and licenca.lower()!='cc-by-4.0':avisos.append(f'Licença informada pela API: {licenca!r}; revisar atribuição.')

        amostra_path=ASTA_RAW/'amostra_utilizada.csv'
        if amostra_path.is_file():
            amostra=pd.read_csv(amostra_path)
            colunas={'arquivo_imagem','arquivo_mascara'}
            if not colunas.issubset(amostra.columns):
                avisos.append('amostra_utilizada.csv não possui as duas colunas históricas esperadas.')
            else:
                pares_amostra=set(map(tuple,amostra[['arquivo_imagem','arquivo_mascara']].astype(str).to_numpy()))
                if pares_amostra!=set(pares_oficiais):avisos.append('amostra_utilizada.csv não coincide exatamente com o índice oficial atual.')
        else:avisos.append('amostra_utilizada.csv histórico não foi encontrado; o manifest v3 será a fonte rastreável.')

        inventario_tamanho=[
            {'imagem':img,'mascara':mask,'bytes_imagem':(ASTA_RAW/img).stat().st_size,
             'bytes_mascara':(ASTA_RAW/mask).stat().st_size,
             'mtime_ns_imagem':(ASTA_RAW/img).stat().st_mtime_ns,
             'mtime_ns_mascara':(ASTA_RAW/mask).stat().st_mtime_ns}
            for img,mask in pares_oficiais
        ]
        fingerprint_inventario=sha256_canonico(inventario_tamanho)
        contrato_inventario={
            'protocolo':'asta_inventory_audit_v3','record_id':RECORD_ID,'doi':DOI,
            'url_zip':URL_ZIP,'zip_md5_oficial':checksum_zip,
            'zip_tamanho_bytes_oficial':int(arquivo_zip['size']),'licenca_api':licenca,
            'pares_oficiais':len(pares_oficiais),'imagens_drive':len(imagens_locais),
            'mascaras_drive':len(mascaras_locais),'faltantes_imagem':faltantes_imagem,
            'faltantes_mascara':faltantes_mascara,'extras_imagem':extras_imagem,
            'extras_mascara':extras_mascara,'fingerprint_nomes_tamanhos':fingerprint_inventario,
            'raw_somente_leitura':True,'yolo_executado':False,'hough_executado':False,
            'erros':erros,'avisos':avisos
        }
        contrato_inventario['sha256_contrato_inventario']=sha256_canonico(contrato_inventario)
        print(json.dumps(contrato_inventario,indent=2,ensure_ascii=False))
        if erros:raise RuntimeError('Gate de inventário ASTA reprovado; não execute a auditoria pesada.')
        print('INVENTARIO_ASTA_APROVADO_PARA_AUDITORIA_PESADA')
        print('Compartilhe este relatório antes de habilitar a seção 2.')
        ''') ,
        md(r'''
        ## 2. Auditoria pesada, retomável e sem inferência

        Primeiro execute com `EXECUTAR_AUDITORIA_PESADA=False`. Após conferir o gate leve,
        altere somente essa chave para `True`. Cada par é copiado para uma pasta temporária,
        auditado e removido; checkpoints são publicados a cada cinco pares. Uma queda da sessão
        não obriga a repetir os pares já confirmados sob o mesmo fingerprint de inventário.

        As folhas mostram uma miniatura da imagem com a máscara em vermelho. Componentes
        conectados são **sinais para revisão**, não equivalem automaticamente ao número de
        trilhas: uma trilha pode ser interrompida e uma imagem pode conter várias trilhas.
        ''') ,
        code(r'''
        EXECUTAR_AUDITORIA_PESADA=False

        CHECKPOINT_MANIFEST=AUDIT_DRIVE/'checkpoint_manifest_auditoria_asta_v3.csv'
        CHECKPOINT_COMPONENTES=AUDIT_DRIVE/'checkpoint_componentes_mascaras_asta_v3.csv'
        CHECKPOINT_CONTRATO=AUDIT_DRIVE/'checkpoint_contrato_auditoria_asta_v3.json'

        def copiar_para_stage_com_hash(origem,destino,tentativas=3,chunk=8*1024*1024):
            ultimo=None
            for tentativa in range(1,tentativas+1):
                try:
                    if destino.exists():destino.unlink()
                    h=hashlib.sha256()
                    with origem.open('rb') as entrada,destino.open('wb') as saida:
                        for bloco in iter(lambda:entrada.read(chunk),b''):
                            saida.write(bloco);h.update(bloco)
                    if destino.stat().st_size!=origem.stat().st_size:raise IOError('tamanho local divergente após cópia')
                    return h.hexdigest()
                except (OSError,IOError) as exc:
                    ultimo=exc
                    if destino.exists():destino.unlink()
                    if tentativa<tentativas:time.sleep(2**tentativa)
            raise IOError(f'Falha ao preparar {origem.name} após {tentativas} tentativas: {ultimo}')

        def orientacao_componente(labels,indice,stat):
            x,y,w,h,area=[int(v) for v in stat]
            recorte=labels[y:y+h,x:x+w]
            yy,xx=np.where(recorte==indice)
            if len(xx)>200_000:
                passo=math.ceil(len(xx)/200_000);xx=xx[::passo];yy=yy[::passo]
            if len(xx)<2:return np.nan
            pts=np.column_stack([xx.astype(np.float32)+x,yy.astype(np.float32)+y]).reshape(-1,1,2)
            vx,vy,_,_=cv2.fitLine(pts,cv2.DIST_L2,0,0.01,0.01).reshape(-1)
            return float(np.degrees(np.arctan2(vy,vx))%180.0)

        def salvar_preview(caminho_img,mask,nome_saida,titulo):
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(caminho_img) as pil:
                    modo=pil.mode
                    cinza=pil.convert('L');cinza.thumbnail((512,512),Image.Resampling.LANCZOS)
                    base=np.asarray(cinza,dtype=np.uint8)
            mini_mask=cv2.resize((mask>0).astype(np.uint8),(base.shape[1],base.shape[0]),interpolation=cv2.INTER_NEAREST)
            rgb=np.repeat(base[:,:,None],3,axis=2);ativo=mini_mask.astype(bool)
            rgb[ativo,0]=255;rgb[ativo,1]=(rgb[ativo,1]*0.25).astype(np.uint8);rgb[ativo,2]=(rgb[ativo,2]*0.25).astype(np.uint8)
            quadro=Image.new('RGB',(512,548),'white');quadro.paste(Image.fromarray(rgb),(0,36))
            ImageDraw.Draw(quadro).text((8,10),titulo[:74],fill='black')
            temporario=AUDIT_LOCAL/(nome_saida+'.tmp.jpg');quadro.save(temporario,quality=88,optimize=True)
            destino=PREVIEWS_DRIVE/nome_saida;shutil.copy2(temporario,destino);temporario.unlink()
            return modo,base

        def auditar_par(nome_img,nome_mask):
            origem_img,origem_mask=ASTA_RAW/nome_img,ASTA_RAW/nome_mask
            local_img,local_mask=STAGING/nome_img,STAGING/nome_mask
            hash_img=copiar_para_stage_com_hash(origem_img,local_img)
            hash_mask=copiar_para_stage_com_hash(origem_mask,local_mask)
            try:
                with Image.open(local_img) as pil_img:
                    largura_img,altura_img=pil_img.size;modo_img=pil_img.mode;formato_img=pil_img.format
                with Image.open(local_mask) as pil_mask:
                    largura_mask,altura_mask=pil_mask.size;modo_mask=pil_mask.mode;formato_mask=pil_mask.format
                mask_original=cv2.imread(str(local_mask),cv2.IMREAD_UNCHANGED)
                if mask_original is None:raise IOError('cv2 não conseguiu ler a máscara')
                canais_mask=1 if mask_original.ndim==2 else int(mask_original.shape[2])
                dtype_mask=str(mask_original.dtype)
                if mask_original.ndim==2:mask=mask_original
                elif mask_original.shape[2]==4:mask=cv2.cvtColor(mask_original,cv2.COLOR_BGRA2GRAY)
                else:mask=cv2.cvtColor(mask_original,cv2.COLOR_BGR2GRAY)
                if mask.shape!=(altura_mask,largura_mask):raise ValueError('shape da máscara diverge do cabeçalho PIL')
                if mask.dtype==np.uint8:
                    hist=cv2.calcHist([mask],[0],None,[256],[0,256]).ravel();valores_mask=[int(i) for i,v in enumerate(hist) if v>0]
                else:valores_mask=[int(x) for x in np.unique(mask)]
                binaria=np.where(mask>0,255,0).astype(np.uint8)
                n_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(binaria,connectivity=8)
                componentes=[]
                for indice in range(1,n_labels):
                    x,y,w,h,area=[int(v) for v in stats[indice]]
                    if area<AREA_MINIMA:continue
                    componentes.append({
                        'id_asta':Path(nome_img).name.replace('.fits_full.png',''),'arquivo_mascara':nome_mask,
                        'componente_indice':indice,'area_pixels':area,'x':x,'y':y,'largura_bbox':w,'altura_bbox':h,
                        'centro_x':float(centroids[indice][0]),'centro_y':float(centroids[indice][1]),
                        'angulo_graus_0_180':orientacao_componente(labels,indice,stats[indice]),
                        'toca_borda':bool(x==0 or y==0 or x+w==largura_mask or y+h==altura_mask)
                    })
                id_asta=Path(nome_img).name.replace('.fits_full.png','')
                preview_nome=id_asta+'.jpg'
                _,mini=salvar_preview(local_img,mask,preview_nome,f'{id_asta} | componentes >= {AREA_MINIMA}: {len(componentes)}')
                p=np.percentile(mini,[1,50,99,99.8]).astype(float)
                linha={
                    'id_asta':id_asta,'arquivo_imagem':nome_img,'arquivo_mascara':nome_mask,
                    'bytes_imagem':origem_img.stat().st_size,'bytes_mascara':origem_mask.stat().st_size,
                    'sha256_imagem':hash_img,'sha256_mascara':hash_mask,
                    'largura_imagem':largura_img,'altura_imagem':altura_img,'modo_imagem':modo_img,'formato_imagem':formato_img,
                    'largura_mascara':largura_mask,'altura_mascara':altura_mask,'modo_mascara':modo_mask,'formato_mascara':formato_mask,
                    'dtype_mascara':dtype_mask,'canais_mascara':canais_mask,
                    'dimensoes_coincidem':bool((largura_img,altura_img)==(largura_mask,altura_mask)),
                    'valores_mascara':'|'.join(map(str,valores_mask)),'mascara_pixels_positivos':int(cv2.countNonZero(binaria)),
                    'mascara_fracao_positiva':float(cv2.countNonZero(binaria)/binaria.size),
                    'componentes_totais':int(n_labels-1),'componentes_relevantes':len(componentes),
                    'componentes_pequenos':int((n_labels-1)-len(componentes)),
                    'componentes_relevantes_tocam_borda':int(sum(c['toca_borda'] for c in componentes)),
                    'imagem_p1_miniatura':p[0],'imagem_p50_miniatura':p[1],
                    'imagem_p99_miniatura':p[2],'imagem_p998_miniatura':p[3],
                    'imagem_desvio_miniatura':float(mini.std()),'preview_relativo':str(preview_nome)
                }
                del labels,stats,centroids,binaria,mask,mask_original,mini
                return linha,componentes
            finally:
                if local_img.exists():local_img.unlink()
                if local_mask.exists():local_mask.unlink()

        def publicar_checkpoint(linhas,componentes):
            gravar_csv_atomico(CHECKPOINT_MANIFEST,pd.DataFrame(linhas).sort_values('id_asta'))
            comp_df=pd.DataFrame(componentes)
            if len(comp_df):comp_df=comp_df.sort_values(['id_asta','componente_indice'])
            gravar_csv_atomico(CHECKPOINT_COMPONENTES,comp_df)
            gravar_json_atomico(CHECKPOINT_CONTRATO,{
                'protocolo':'asta_audit_checkpoint_v3','fingerprint_nomes_tamanhos':fingerprint_inventario,
                'pares_concluidos':len(linhas),'atualizado_em_utc':datetime.now(timezone.utc).isoformat(),
                'raw_somente_leitura':True,'yolo_executado':False,'hough_executado':False
            })

        if not EXECUTAR_AUDITORIA_PESADA:
            print('Auditoria pesada bloqueada. Compartilhe primeiro o gate leve da seção 1.')
        else:
            contrato_checkpoint=json.loads(CHECKPOINT_CONTRATO.read_text(encoding='utf-8')) if CHECKPOINT_CONTRATO.exists() else None
            if contrato_checkpoint:
                assert contrato_checkpoint['fingerprint_nomes_tamanhos']==fingerprint_inventario,'Inventário mudou; checkpoint não pode ser reutilizado.'
            linhas_existentes=pd.read_csv(CHECKPOINT_MANIFEST).to_dict('records') if CHECKPOINT_MANIFEST.exists() else []
            if CHECKPOINT_COMPONENTES.exists() and CHECKPOINT_COMPONENTES.stat().st_size:
                try:componentes_existentes=pd.read_csv(CHECKPOINT_COMPONENTES).to_dict('records')
                except pd.errors.EmptyDataError:componentes_existentes=[]
            else:componentes_existentes=[]
            por_id={str(x['id_asta']):x for x in linhas_existentes}
            componentes_por_id={}
            for c in componentes_existentes:componentes_por_id.setdefault(str(c['id_asta']),[]).append(c)
            bytes_total=sum(x['bytes_imagem']+x['bytes_mascara'] for x in inventario_tamanho)
            bytes_novos=0;novos_desde_checkpoint=0;inicio=time.time()
            barra=tqdm(pares_oficiais,total=len(pares_oficiais),desc='Auditoria ASTA',unit='par')
            for nome_img,nome_mask in barra:
                id_asta=nome_img.replace('.fits_full.png','')
                preview_ok=(PREVIEWS_DRIVE/(id_asta+'.jpg')).is_file()
                if id_asta in por_id and preview_ok:
                    barra.set_postfix(concluidos=len(por_id),retomados='sim');continue
                linha,componentes=auditar_par(nome_img,nome_mask)
                por_id[id_asta]=linha;componentes_por_id[id_asta]=componentes
                bytes_novos+=linha['bytes_imagem']+linha['bytes_mascara'];novos_desde_checkpoint+=1
                decorrido=max(time.time()-inicio,1e-9);taxa=bytes_novos/decorrido
                restante=max(0,bytes_total-bytes_novos)
                barra.set_postfix(concluidos=len(por_id),GB=f'{bytes_novos/1e9:.2f}',MBps=f'{taxa/1e6:.1f}',eta_min=f'{restante/max(taxa,1)/60:.1f}')
                if novos_desde_checkpoint>=CHECKPOINT_EVERY:
                    todas_linhas=list(por_id.values());todos_componentes=[c for grupo in componentes_por_id.values() for c in grupo]
                    publicar_checkpoint(todas_linhas,todos_componentes);novos_desde_checkpoint=0
                    print(f'[{agora()}] Checkpoint publicado: {len(todas_linhas)}/{len(pares_oficiais)} pares.')
            linhas_auditadas=list(por_id.values());componentes_auditados=[c for grupo in componentes_por_id.values() for c in grupo]
            publicar_checkpoint(linhas_auditadas,componentes_auditados)
            print(f'[{agora()}] Auditoria automática concluída: {len(linhas_auditadas)} pares.')
        ''') ,
        md(r'''
        ## 3. Consolidação, folhas visuais e backup verificável

        Esta seção também não executa modelos. Ela exige os 178 pares auditados, produz folhas
        de revisão com todas as imagens, registra alertas objetivos e publica artefatos finais
        com SHA-256. O resultado automático **não aprova** ainda a conversão das máscaras.
        ''') ,
        code(r'''
        if not CHECKPOINT_MANIFEST.exists():
            print('Auditoria pesada ainda não executada.')
        else:
            manifest_asta=pd.read_csv(CHECKPOINT_MANIFEST).sort_values('id_asta').reset_index(drop=True)
            if CHECKPOINT_COMPONENTES.exists() and CHECKPOINT_COMPONENTES.stat().st_size:
                try:componentes_asta=pd.read_csv(CHECKPOINT_COMPONENTES)
                except pd.errors.EmptyDataError:componentes_asta=pd.DataFrame()
            else:componentes_asta=pd.DataFrame()
            erros=[];avisos=[]
            inventario_atual=[
                {'imagem':img,'mascara':mask,'bytes_imagem':(ASTA_RAW/img).stat().st_size,
                 'bytes_mascara':(ASTA_RAW/mask).stat().st_size,
                 'mtime_ns_imagem':(ASTA_RAW/img).stat().st_mtime_ns,
                 'mtime_ns_mascara':(ASTA_RAW/mask).stat().st_mtime_ns}
                for img,mask in pares_oficiais
            ]
            if sha256_canonico(inventario_atual)!=fingerprint_inventario:erros.append('Inventário raw mudou durante a auditoria.')
            if len(manifest_asta)!=178:erros.append(f'Esperados 178 pares auditados; encontrados {len(manifest_asta)}.')
            if manifest_asta.id_asta.nunique()!=len(manifest_asta):erros.append('IDs ASTA duplicados no manifest.')
            if not manifest_asta.dimensoes_coincidem.astype(bool).all():erros.append('Há pares com dimensões imagem/máscara divergentes.')
            if manifest_asta.sha256_imagem.duplicated().any():avisos.append('Há imagens com SHA-256 duplicado; revisar proveniência.')
            if manifest_asta.sha256_mascara.duplicated().any():avisos.append('Há máscaras com SHA-256 duplicado; revisar conteúdo.')
            vazias=int((manifest_asta.mascara_pixels_positivos==0).sum())
            multicomponentes=int((manifest_asta.componentes_relevantes>1).sum())
            sem_componente_relevante=int((manifest_asta.componentes_relevantes==0).sum())
            toca_borda=int((manifest_asta.componentes_relevantes_tocam_borda>0).sum())
            if vazias:avisos.append(f'{vazias} máscaras vazias exigem revisão.')
            if sem_componente_relevante:avisos.append(f'{sem_componente_relevante} máscaras sem componente >= {AREA_MINIMA} exigem revisão.')
            if multicomponentes:avisos.append(f'{multicomponentes} máscaras possuem múltiplos componentes relevantes; isso não define sozinho o número de trilhas.')

            FOLHAS_LOCAL=AUDIT_LOCAL/'folhas_auditoria_asta_v3';FOLHAS_LOCAL.mkdir(parents=True,exist_ok=True)
            previews=[]
            for linha in manifest_asta.itertuples():
                caminho=PREVIEWS_DRIVE/linha.preview_relativo
                if not caminho.is_file():erros.append(f'Preview ausente: {caminho.name}')
                else:previews.append((linha.id_asta,caminho,int(linha.componentes_relevantes)))
            for antigo in FOLHAS_LOCAL.glob('folha_*.jpg'):antigo.unlink()
            por_folha=12;colunas=4;largura_item,altura_item=384,411
            for inicio_indice in range(0,len(previews),por_folha):
                lote=previews[inicio_indice:inicio_indice+por_folha];linhas_grade=math.ceil(len(lote)/colunas)
                folha=Image.new('RGB',(colunas*largura_item,linhas_grade*altura_item),(235,235,235))
                for pos,(id_asta,caminho,n_comp) in enumerate(lote):
                    with Image.open(caminho) as prev:
                        prev=prev.convert('RGB');prev.thumbnail((largura_item,altura_item-24),Image.Resampling.LANCZOS)
                        x=(pos%colunas)*largura_item+(largura_item-prev.width)//2
                        y=(pos//colunas)*altura_item+24;folha.paste(prev,(x,y))
                    ImageDraw.Draw(folha).text(((pos%colunas)*largura_item+5,(pos//colunas)*altura_item+5),f'{id_asta} | comp={n_comp}',fill='black')
                folha.save(FOLHAS_LOCAL/f'folha_{inicio_indice//por_folha+1:02d}.jpg',quality=88,optimize=True)

            FINAL_LOCAL=AUDIT_LOCAL/'final';FINAL_LOCAL.mkdir(parents=True,exist_ok=True)
            manifest_path=FINAL_LOCAL/'manifest_auditoria_asta_v3.csv';manifest_asta.to_csv(manifest_path,index=False)
            componentes_path=FINAL_LOCAL/'componentes_mascaras_asta_v3.csv';componentes_asta.to_csv(componentes_path,index=False)
            contrato_path=FINAL_LOCAL/'contrato_inventario_asta_v3.json';contrato_path.write_text(json.dumps(contrato_inventario,indent=2,ensure_ascii=False),encoding='utf-8')
            zip_path=FINAL_LOCAL/'folhas_auditoria_asta_v3.zip'
            with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
                for folha in sorted(FOLHAS_LOCAL.glob('folha_*.jpg')):zf.write(folha,arcname=folha.name)

            dimensoes=(manifest_asta.groupby(['largura_imagem','altura_imagem']).size().reset_index(name='quantidade').to_dict('records'))
            modos=manifest_asta.modo_imagem.value_counts().to_dict();valores=manifest_asta.valores_mascara.value_counts().to_dict()
            resumo={
                'protocolo':'asta_external_audit_v3','status':'AUDITORIA_AUTOMATICA_ASTA_CONCLUIDA_AGUARDA_REVISAO_VISUAL',
                'record_id':RECORD_ID,'doi':DOI,'pares_auditados':len(manifest_asta),
                'fingerprint_nomes_tamanhos':fingerprint_inventario,
                'sha256_contrato_inventario':contrato_inventario['sha256_contrato_inventario'],
                'dimensoes_imagem':dimensoes,'modos_imagem':modos,'valores_mascara':valores,
                'mascaras_vazias':vazias,'sem_componente_relevante':sem_componente_relevante,
                'multiplos_componentes_relevantes':multicomponentes,'componentes_tocando_borda':toca_borda,
                'area_minima_componente':AREA_MINIMA,'erros':erros,'avisos':avisos,
                'raw_somente_leitura':True,'yolo_executado':False,'hough_executado':False,
                'protocolo_tiles_congelado':False,'conversao_mascara_objetos_congelada':False,
                'revisao_visual_aprovada':False
            }
            resumo['sha256_manifest_auditoria']=sha256_arquivo(manifest_path)
            resumo['sha256_componentes']=sha256_arquivo(componentes_path)
            resumo['sha256_folhas_zip']=sha256_arquivo(zip_path)
            resumo_path=FINAL_LOCAL/'resumo_auditoria_asta_v3.json';resumo_path.write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf-8')
            registro_hashes={p.name:sha256_arquivo(p) for p in sorted(FINAL_LOCAL.iterdir()) if p.is_file()}
            hashes_path=FINAL_LOCAL/'hashes_artefatos_auditoria_asta_v3.json';hashes_path.write_text(json.dumps(registro_hashes,indent=2),encoding='utf-8')
            if erros:
                print(json.dumps(resumo,indent=2,ensure_ascii=False));raise RuntimeError('Consolidação ASTA reprovada; não avançar.')
            FINAL_DRIVE=AUDIT_DRIVE/'final';FINAL_DRIVE.mkdir(parents=True,exist_ok=True)
            for arquivo in tqdm(sorted(FINAL_LOCAL.iterdir()),desc='Publicando auditoria ASTA',unit='arq'):
                if arquivo.is_file():shutil.copy2(arquivo,FINAL_DRIVE/arquivo.name)
            for nome,hash_esperado in registro_hashes.items():assert sha256_arquivo(FINAL_DRIVE/nome)==hash_esperado
            assert sha256_arquivo(FINAL_DRIVE/'hashes_artefatos_auditoria_asta_v3.json')==sha256_arquivo(hashes_path)
            print(json.dumps(resumo,indent=2,ensure_ascii=False))
            print('BACKUP_AUDITORIA_ASTA_AUTOMATICA_APROVADO')
            print('Envie o relatório acima e folhas_auditoria_asta_v3.zip. Nenhuma inferência foi executada.')
        ''') ,
        md(r'''
        ## 4. Verificação pós-backup, somente leitura

        Pode ser executada após reconexão. Confirma os hashes dos artefatos finais sem reler os
        18 GB e sem executar detectores. A revisão visual e o protocolo externo continuam
        pendentes mesmo quando este gate passa.
        ''') ,
        code(r'''
        FINAL_DRIVE=AUDIT_DRIVE/'final'
        hashes_remoto=FINAL_DRIVE/'hashes_artefatos_auditoria_asta_v3.json'
        resumo_remoto=FINAL_DRIVE/'resumo_auditoria_asta_v3.json'
        if not hashes_remoto.is_file() or not resumo_remoto.is_file():
            print('Backup final da auditoria ainda não existe.')
        else:
            hashes=json.loads(hashes_remoto.read_text(encoding='utf-8'))
            problemas=[]
            for nome,esperado in tqdm(sorted(hashes.items()),desc='Verificando artefatos ASTA',unit='arq'):
                caminho=FINAL_DRIVE/nome
                if not caminho.is_file() or sha256_arquivo(caminho)!=esperado:problemas.append(nome)
            resumo_final=json.loads(resumo_remoto.read_text(encoding='utf-8'))
            assert resumo_final['status']=='AUDITORIA_AUTOMATICA_ASTA_CONCLUIDA_AGUARDA_REVISAO_VISUAL'
            assert resumo_final['raw_somente_leitura'] is True
            assert resumo_final['yolo_executado'] is False and resumo_final['hough_executado'] is False
            assert resumo_final['protocolo_tiles_congelado'] is False
            assert resumo_final['revisao_visual_aprovada'] is False
            if problemas:raise RuntimeError(f'Artefatos divergentes: {problemas}')
            print(json.dumps(resumo_final,indent=2,ensure_ascii=False))
            print('BACKUP_AUDITORIA_ASTA_AUTOMATICA_VERIFICADO_SEM_INFERENCIA')
        ''') ,
    ]
    return write_notebook('11_auditoria_externa_asta_sem_inferencia_v3.ipynb', cells)


def build_12():
    cells = [
        md(r'''
        # 12 — Protocolo externo ASTA: tiles e centerline sem inferência (v3)

        Este notebook transforma a auditoria aprovada do ASTA em um contrato de avaliação
        externo, **antes** da primeira predição. Ele não carrega pesos, não chama YOLO e não
        executa o detector Hough nas imagens.

        O ASTA fornece máscaras semânticas, não OBBs por instância. Por isso o endpoint externo
        principal será uma centerline obtida por skeletonização da máscara. Essa escolha evita
        inventar instâncias em cruzamentos e evita tratar segmentos desconectados como satélites
        diferentes. YOLO e Hough serão comparados mais tarde depois de convertidos para
        centerlines globais sob o mesmo contrato.

        Execute em **CPU**. A validação das 178 máscaras é retomável, mostra progresso e usa
        somente uma máscara local por vez. `data/raw/asta_real` continua somente leitura.
        ''') ,
        md(r'''
        ## 0. Ambiente, caminhos e funções de integridade

        Esta seção monta o Drive e declara os hashes já aprovados. Nenhum detector é importado.
        ''') ,
        code(r'''
        %pip install -q scikit-image

        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)

        from pathlib import Path
        from datetime import datetime, timezone
        from tqdm.auto import tqdm
        from PIL import Image, ImageDraw
        from skimage.morphology import skeletonize
        import gc, hashlib, json, math, os, shutil, time, zipfile
        import cv2
        import numpy as np
        import pandas as pd

        Image.MAX_IMAGE_PIXELS=200_000_000

        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        ASTA_RAW=BACKUP_DIR/'data/raw/asta_real'
        AUDIT_FINAL=BACKUP_DIR/'experiments/asta_external_v3/audit/final'
        AUDIT_PREVIEWS=BACKUP_DIR/'experiments/asta_external_v3/audit/previews_checkpoint'
        PROTOCOL_DRIVE=BACKUP_DIR/'experiments/asta_external_v3/protocol'
        DRAFT_DRIVE=PROTOCOL_DRIVE/'draft_review'
        FINAL_DRIVE=PROTOCOL_DRIVE/'final'
        LOCAL=Path('/content/asta_protocol_v3')
        STAGING=LOCAL/'staging_uma_mascara'
        GEOM_PREVIEWS_DRIVE=PROTOCOL_DRIVE/'skeleton_previews_checkpoint'

        EXPECTED_AUDIT_MANIFEST_SHA256='91bfa07ebffceaa46271a254d3862348ce2b242a06ed44fc4c07efeeb58722ca'
        EXPECTED_COMPONENTS_SHA256='cb0089cb07731e5fb233c112f315bae28207b8dad58cec0f733f829565846059'
        EXPECTED_SHEETS_SHA256='2dd4a0b2b35a19bd0549c25ed65e4c01d472aafcfae88fbf24da9fd598dc40e9'
        EXPECTED_INVENTORY_CONTRACT_SHA256='ca4609ef651bd30cef611bd394c539868b10f2e8547676e4d00af7e271c14c06'
        EXPECTED_FINAL_MODELS_CONTRACT_SHA256='0a8d3537969dcb033dc8cdfa1a9d63dc24de4cc98e212c42fab62621126890a7'
        EXPECTED_FINAL_CONFIG_SHA256='457b70c8e96176827bf4a3f755963e6c6c3e607bec7d8133233518b26be27870'
        EXPECTED_OPERATIONAL_WEIGHT_SHA256='a2b64773f32868ac952d81c33b331f1759b2250a3a04f49d5ab105f7bb79dd0c'
        EXPECTED_HOUGH_CONFIG_SHA256='14ef0ea56a2e86bf1581225c4b9a0733d482834e5f7d3fab18b3eaf76f03a2e5'
        RUN_ID='synthetic_v3_c41d5091de56'
        CHECKPOINT_EVERY=5

        for pasta in [LOCAL,STAGING,GEOM_PREVIEWS_DRIVE,PROTOCOL_DRIVE]:pasta.mkdir(parents=True,exist_ok=True)
        assert ASTA_RAW.is_dir() and AUDIT_FINAL.is_dir() and AUDIT_PREVIEWS.is_dir()

        def agora():return datetime.now().strftime('%H:%M:%S')
        def sha256_arquivo(caminho,chunk=8*1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as arq:
                for bloco in iter(lambda:arq.read(chunk),b''):h.update(bloco)
            return h.hexdigest()
        def sha256_canonico(objeto):
            texto=json.dumps(objeto,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            return hashlib.sha256(texto.encode('utf-8')).hexdigest()
        def verificar_hash_canonico(registro,campo,ensure_ascii=False):
            copia=dict(registro);observado=copia.pop(campo)
            texto=json.dumps(copia,sort_keys=True,separators=(',',':'),ensure_ascii=ensure_ascii)
            assert hashlib.sha256(texto.encode('utf-8')).hexdigest()==observado,f'Hash canônico divergente: {campo}'
            return observado
        def gravar_json_atomico(caminho,objeto):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            temporario=caminho.with_name(caminho.name+'.tmp')
            temporario.write_text(json.dumps(objeto,indent=2,ensure_ascii=False),encoding='utf-8')
            os.replace(temporario,caminho)
        def gravar_csv_atomico(caminho,tabela):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            temporario=caminho.with_name(caminho.name+'.tmp');tabela.to_csv(temporario,index=False)
            os.replace(temporario,caminho)

        print(f'[{agora()}] Runtime CPU preparado. ASTA raw permanece somente leitura.')
        ''') ,
        md(r'''
        ## 1. Gate dos artefatos aprovados

        Confirma auditoria, CSVs, modelo operacional escolhido na validação e configuração Hough
        congelada. Os pesos são apenas re-hashados no Drive; nenhum modelo é carregado.
        ''') ,
        code(r'''
        hashes_auditoria=json.loads((AUDIT_FINAL/'hashes_artefatos_auditoria_asta_v3.json').read_text(encoding='utf-8'))
        problemas=[]
        for nome,esperado in tqdm(sorted(hashes_auditoria.items()),desc='Verificando auditoria ASTA',unit='arq'):
            caminho=AUDIT_FINAL/nome
            if not caminho.is_file() or sha256_arquivo(caminho)!=esperado:problemas.append(nome)
        if problemas:raise RuntimeError(f'Artefatos da auditoria divergentes: {problemas}')

        manifest_path=AUDIT_FINAL/'manifest_auditoria_asta_v3.csv'
        componentes_path=AUDIT_FINAL/'componentes_mascaras_asta_v3.csv'
        folhas_path=AUDIT_FINAL/'folhas_auditoria_asta_v3.zip'
        resumo_auditoria=json.loads((AUDIT_FINAL/'resumo_auditoria_asta_v3.json').read_text(encoding='utf-8'))
        contrato_inventario=json.loads((AUDIT_FINAL/'contrato_inventario_asta_v3.json').read_text(encoding='utf-8'))
        assert sha256_arquivo(manifest_path)==EXPECTED_AUDIT_MANIFEST_SHA256
        assert sha256_arquivo(componentes_path)==EXPECTED_COMPONENTS_SHA256
        assert sha256_arquivo(folhas_path)==EXPECTED_SHEETS_SHA256
        assert contrato_inventario['sha256_contrato_inventario']==EXPECTED_INVENTORY_CONTRACT_SHA256
        assert resumo_auditoria['raw_somente_leitura'] is True
        assert resumo_auditoria['yolo_executado'] is False and resumo_auditoria['hough_executado'] is False

        manifest=pd.read_csv(manifest_path).sort_values('id_asta').reset_index(drop=True)
        componentes=pd.read_csv(componentes_path)
        assert len(manifest)==178 and manifest.id_asta.nunique()==178
        assert len(componentes)==int(manifest.componentes_relevantes.sum())==291
        assert manifest.sha256_imagem.nunique()==178 and manifest.sha256_mascara.nunique()==178
        assert manifest.dimensoes_coincidem.astype(bool).all()
        vazios=manifest.loc[manifest.mascara_pixels_positivos==0,'id_asta'].astype(str).tolist()
        assert vazios==['ML1_20200328_002539_red']

        final_dir=BACKUP_DIR/'experiments/yolo_final_multiseed_v3'/RUN_ID
        contrato_modelos=json.loads((final_dir/'contrato_modelos_finais_v3.json').read_text(encoding='utf-8'))
        config_final=json.loads((final_dir/'protocolo_treino_final_multiseed_v3.json').read_text(encoding='utf-8'))
        assert verificar_hash_canonico(contrato_modelos,'sha256_contrato_modelos_finais',ensure_ascii=True)==EXPECTED_FINAL_MODELS_CONTRACT_SHA256
        assert verificar_hash_canonico(config_final,'sha256_config_final',ensure_ascii=True)==EXPECTED_FINAL_CONFIG_SHA256
        assert contrato_modelos['checkpoint_operacional_seed']==20260824
        assert contrato_modelos['checkpoint_operacional_sha256']==EXPECTED_OPERATIONAL_WEIGHT_SHA256
        operacional=next(x for x in contrato_modelos['checkpoints'] if int(x['seed'])==20260824)
        peso_operacional=BACKUP_DIR/operacional['best_pt_relativo_drive']
        assert peso_operacional.is_file() and sha256_arquivo(peso_operacional)==EXPECTED_OPERATIONAL_WEIGHT_SHA256

        hough_config_path=BACKUP_DIR/'data/synthetic_runs'/RUN_ID/'hough_v3/config_congelada_v3.json'
        hough_config=json.loads(hough_config_path.read_text(encoding='utf-8'))
        assert hough_config['sha256_config']==EXPECTED_HOUGH_CONFIG_SHA256

        gate_entradas={
            'status':'ENTRADAS_PROTOCOLO_ASTA_APROVADAS','pares':len(manifest),
            'componentes_relevantes':len(componentes),'mascara_vazia':vazios[0],
            'sha256_manifest_auditoria':EXPECTED_AUDIT_MANIFEST_SHA256,
            'sha256_componentes':EXPECTED_COMPONENTS_SHA256,
            'sha256_contrato_modelos_finais':EXPECTED_FINAL_MODELS_CONTRACT_SHA256,
            'checkpoint_operacional_seed':20260824,
            'sha256_checkpoint_operacional':EXPECTED_OPERATIONAL_WEIGHT_SHA256,
            'sha256_hough_config':EXPECTED_HOUGH_CONFIG_SHA256,
            'raw_somente_leitura':True,'yolo_executado':False,'hough_executado':False
        }
        print(json.dumps(gate_entradas,indent=2,ensure_ascii=False))
        print('Compartilhe este gate se qualquer hash ou contagem divergir. Caso aprovado, prossiga.')
        ''') ,
        md(r'''
        ## 2. Protocolo candidato e testes determinísticos

        O grid regular usa 512 × 512, overlap de 128 e padding apenas à direita/abaixo.
        Percentis são calculados no frame válido antes do recorte; padding da imagem recebe a
        mediana normalizada e padding da máscara recebe zero. Predições cujo centro cair fora
        do frame original são descartadas.

        O endpoint primário é F1 de centerline com tolerância de 14 pixels, reportado macro por
        frame positivo e pooled por comprimento. OBB-IoU não é endpoint primário no ASTA porque
        a anotação oficial é uma máscara semântica.
        ''') ,
        code(r'''
        def plano_eixo(tamanho,tile,stride):
            quantidade=math.ceil(max(0,tamanho-tile)/stride)+1
            inicios=[i*stride for i in range(quantidade)]
            tamanho_padded=inicios[-1]+tile
            return {'inicios':inicios,'quantidade':quantidade,'tamanho_padded':tamanho_padded,
                    'padding_final':tamanho_padded-tamanho}

        eixo_x=plano_eixo(10560,512,384);eixo_y=plano_eixo(10560,512,384)
        assert eixo_x['quantidade']==eixo_y['quantidade']==28
        assert eixo_x['padding_final']==eixo_y['padding_final']==320
        cobertura=np.zeros(eixo_x['tamanho_padded'],dtype=np.uint8)
        for inicio in eixo_x['inicios']:cobertura[inicio:inicio+512]=1
        assert cobertura[:10560].all()

        # Testes mínimos da representação oficial: vazio permanece vazio e cruzamento permanece
        # uma única máscara semântica, sem tentativa de inventar duas instâncias OBB.
        teste_vazio=np.zeros((64,64),dtype=bool)
        teste_cruz=np.zeros((64,64),dtype=np.uint8)
        cv2.line(teste_cruz,(4,4),(59,59),1,3);cv2.line(teste_cruz,(4,59),(59,4),1,3)
        sk_vazio=skeletonize(teste_vazio);sk_cruz=skeletonize(teste_cruz>0)
        assert int(sk_vazio.sum())==0 and int(sk_cruz.sum())>80

        CONFIG_PROTOCOLO={
            'protocolo':'asta_external_protocol_candidate_v3','versao':'3.0',
            'fonte':{'record_id':'11642424','doi':'10.5281/zenodo.11642424','licenca':'cc-by-4.0',
                     'pares':178,'largura':10560,'altura':10560,
                     'sha256_manifest_auditoria':EXPECTED_AUDIT_MANIFEST_SHA256,
                     'sha256_componentes':EXPECTED_COMPONENTS_SHA256,
                     'sha256_contrato_inventario':EXPECTED_INVENTORY_CONTRACT_SHA256},
            'tiles':{'tamanho':512,'overlap':128,'stride':384,'grid_x':28,'grid_y':28,
                     'tiles_por_frame':784,'tiles_total':139552,'padding_direita':320,
                     'padding_abaixo':320,'grid_regular':True,
                     'descartar_predicao_centro_fora_frame':True},
            'preprocessamento':{'entrada':'uint8_grayscale','escopo_percentis':'frame_inteiro_valido',
                     'percentil_baixo':1.0,'percentil_alto':99.8,'calculo':'histograma_exato_uint8',
                     'saida':'uint8_0_255','padding_imagem':'mediana_normalizada_do_frame',
                     'padding_mascara':0,'mesmo_para_yolo_e_hough':True,
                     'normalizacao_fotometrica':False},
            'ground_truth':{'fonte':'mascara_semantica_oficial','representacao':'skeleton_centerline',
                     'algoritmo':'skimage.morphology.skeletonize_2d','componentes_nao_sao_instancias':True,
                     'mascara_vazia_id':'ML1_20200328_002539_red','mascara_vazia_papel':'negativo_real'},
            'modelo_yolo':{'familia':'yolo11n_obb','seed_operacional':20260824,
                     'sha256_best_pt':EXPECTED_OPERATIONAL_WEIGHT_SHA256,'imgsz':512,
                     'threshold_confianca':0.45,'iou_nms':0.7,'max_det':100,
                     'obb_para_centerline':'eixo_maior_da_obb'},
            'hough':{'sha256_config_congelada':EXPECTED_HOUGH_CONFIG_SHA256,
                     'recalibracao_no_asta':False},
            'merging_global':{'angulo_max_graus':5.0,'distancia_perpendicular_max_px':14.0,
                     'gap_axial_max_px':128.0,'grafo_componentes_conexos':True,
                     'agregacao_yolo':'ajuste_tls_ponderado_por_confianca',
                     'agregacao_hough':'ajuste_tls_ponderado_por_comprimento'},
            'metricas':{'primaria':'f1_centerline_macro_frames_positivos','tolerancia_centerline_px':14.0,
                     'secundarias':['precisao_recall_f1_centerline_pooled','cobertura_gt_por_frame',
                                    'comprimento_fp_por_megapixel_negativo','taxa_frames_com_fp',
                                    'tempo_preprocessamento','tempo_inferencia','tempo_merging','tempo_total_frame'],
                     'obb_iou_primario':False,'bootstrap_frames':2000,'bootstrap_seed':20260823},
            'restricoes':{'ajuste_por_resultado_asta':False,'recalibrar_threshold':False,
                     'selecionar_modelo_ou_seed':False,'yolo_executado':False,'hough_executado':False,
                     'inferencia_asta_executada':False}
        }
        CONFIG_PROTOCOLO['sha256_protocolo_candidato']=sha256_canonico(CONFIG_PROTOCOLO)
        candidato_path=LOCAL/'protocolo_asta_candidato_v3.json'
        candidato_path.write_text(json.dumps(CONFIG_PROTOCOLO,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(CONFIG_PROTOCOLO,indent=2,ensure_ascii=False))
        print('PROTOCOLO CANDIDATO CALCULADO, MAS AINDA NÃO PUBLICADO.')
        ''') ,
        md(r'''
        ## 3. Validação das centerlines oficiais, retomável

        Execute primeiro com `EXECUTAR_VALIDACAO_SKELETON=False`. Depois do gate de entradas,
        altere somente essa chave para `True`. A rotina copia uma máscara por vez, confere seu
        SHA-256, skeletoniza, salva um preview e remove o arquivo local. O checkpoint é gravado
        a cada cinco máscaras.

        O resultado é um ZIP dirigido: casos com junções, alta multiplicidade, máscara vazia,
        os dois exemplos críticos já conhecidos e uma amostra determinística de casos comuns.
        Nenhuma revisão manual das 178 imagens individualmente é exigida.
        ''') ,
        code(r'''
        EXECUTAR_VALIDACAO_SKELETON=False

        CHECKPOINT_CSV=PROTOCOL_DRIVE/'checkpoint_skeleton_asta_v3.csv'
        CHECKPOINT_JSON=PROTOCOL_DRIVE/'checkpoint_skeleton_asta_v3.json'
        FINAL_LOCAL=LOCAL/'draft_review';FINAL_LOCAL.mkdir(parents=True,exist_ok=True)

        def copiar_com_hash(origem,destino,tentativas=3,chunk=8*1024*1024):
            ultimo=None
            for tentativa in range(1,tentativas+1):
                try:
                    if destino.exists():destino.unlink()
                    h=hashlib.sha256()
                    with origem.open('rb') as entrada,destino.open('wb') as saida:
                        for bloco in iter(lambda:entrada.read(chunk),b''):
                            saida.write(bloco);h.update(bloco)
                    if destino.stat().st_size!=origem.stat().st_size:raise IOError('tamanho divergente')
                    return h.hexdigest()
                except (OSError,IOError) as exc:
                    ultimo=exc
                    if destino.exists():destino.unlink()
                    if tentativa<tentativas:time.sleep(2**tentativa)
            raise IOError(f'Falha após {tentativas} tentativas: {ultimo}')

        def auditar_skeleton(linha):
            origem=ASTA_RAW/linha.arquivo_mascara;local=STAGING/linha.arquivo_mascara
            hash_copia=copiar_com_hash(origem,local)
            if hash_copia!=linha.sha256_mascara:raise RuntimeError(f'Hash de máscara divergente: {linha.id_asta}')
            mask=cv2.imread(str(local),cv2.IMREAD_GRAYSCALE)
            if mask is None:raise RuntimeError(f'Máscara ilegível: {linha.id_asta}')
            if mask.shape!=(int(linha.altura_mascara),int(linha.largura_mascara)):
                raise RuntimeError(f'Dimensão divergente: {linha.id_asta}')
            binaria=mask>0;skeleton=skeletonize(binaria);sk_u8=skeleton.astype(np.uint8)
            vizinhos=cv2.filter2D(sk_u8,cv2.CV_16S,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-sk_u8.astype(np.int16)
            pixels_mask=int(binaria.sum());pixels_skeleton=int(sk_u8.sum())
            endpoints=int(((vizinhos==1)&skeleton).sum())
            junctions=int(((vizinhos>=3)&skeleton).sum())

            preview_origem=AUDIT_PREVIEWS/linha.preview_relativo
            if not preview_origem.is_file():raise RuntimeError(f'Preview da auditoria ausente: {linha.id_asta}')
            with Image.open(preview_origem) as pil:quadro=np.asarray(pil.convert('RGB')).copy()
            mini=cv2.resize(sk_u8.astype(np.float32),(512,512),interpolation=cv2.INTER_AREA)>0
            area=quadro[36:548,:512];area[mini]=np.array([0,255,255],dtype=np.uint8)
            destino_preview=GEOM_PREVIEWS_DRIVE/(linha.id_asta+'.jpg')
            Image.fromarray(quadro).save(destino_preview,quality=90,optimize=True)

            resultado={'id_asta':linha.id_asta,'arquivo_mascara':linha.arquivo_mascara,
                'sha256_mascara':linha.sha256_mascara,'mascara_pixels_positivos':pixels_mask,
                'componentes_relevantes':int(linha.componentes_relevantes),
                'skeleton_pixels':pixels_skeleton,'skeleton_endpoints_pixels':endpoints,
                'skeleton_junction_pixels':junctions,
                'skeleton_por_pixel_mascara':float(pixels_skeleton/pixels_mask) if pixels_mask else 0.0,
                'preview_skeleton':destino_preview.name}
            del mask,binaria,skeleton,sk_u8,vizinhos,quadro,area,mini;gc.collect()
            if local.exists():local.unlink()
            return resultado

        if not EXECUTAR_VALIDACAO_SKELETON:
            print('Validação skeleton bloqueada. Compartilhe primeiro o gate da seção 1.')
        else:
            existentes=[]
            if CHECKPOINT_JSON.is_file() and CHECKPOINT_CSV.is_file():
                contrato_cp=json.loads(CHECKPOINT_JSON.read_text(encoding='utf-8'))
                assert contrato_cp['sha256_manifest_auditoria']==EXPECTED_AUDIT_MANIFEST_SHA256
                assert contrato_cp['sha256_protocolo_candidato']==CONFIG_PROTOCOLO['sha256_protocolo_candidato']
                existentes=pd.read_csv(CHECKPOINT_CSV).to_dict('records')
            por_id={str(x['id_asta']):x for x in existentes}
            inicio=time.time()
            for indice,linha in enumerate(tqdm(manifest.itertuples(),total=len(manifest),desc='Skeleton ASTA',unit='mask'),1):
                preview_ok=(GEOM_PREVIEWS_DRIVE/(linha.id_asta+'.jpg')).is_file()
                if linha.id_asta in por_id and preview_ok:continue
                por_id[linha.id_asta]=auditar_skeleton(linha)
                if indice%CHECKPOINT_EVERY==0 or indice==len(manifest):
                    tabela_cp=pd.DataFrame(por_id.values()).sort_values('id_asta')
                    gravar_csv_atomico(CHECKPOINT_CSV,tabela_cp)
                    gravar_json_atomico(CHECKPOINT_JSON,{
                        'protocolo':'asta_skeleton_checkpoint_v3','processados':len(tabela_cp),
                        'sha256_manifest_auditoria':EXPECTED_AUDIT_MANIFEST_SHA256,
                        'sha256_protocolo_candidato':CONFIG_PROTOCOLO['sha256_protocolo_candidato'],
                        'atualizado_em_utc':datetime.now(timezone.utc).isoformat()})

            diagnostico=pd.DataFrame(por_id.values()).sort_values('id_asta').reset_index(drop=True)
            erros=[]
            if len(diagnostico)!=178 or diagnostico.id_asta.nunique()!=178:erros.append('Diagnóstico não contém 178 IDs únicos.')
            combinada=manifest[['id_asta','mascara_pixels_positivos','componentes_relevantes']].merge(
                diagnostico[['id_asta','mascara_pixels_positivos','skeleton_pixels']],
                on='id_asta',suffixes=('_manifest','_diag'))
            if not (combinada.mascara_pixels_positivos_manifest.astype(int)==combinada.mascara_pixels_positivos_diag.astype(int)).all():
                erros.append('Pixels positivos divergiram entre auditoria e validação skeleton.')
            positivos=combinada.mascara_pixels_positivos_manifest.astype(int)>0
            if not (combinada.loc[positivos,'skeleton_pixels']>0).all():erros.append('Máscara positiva sem skeleton.')
            if not (combinada.loc[~positivos,'skeleton_pixels']==0).all():erros.append('Máscara vazia gerou skeleton.')

            conhecidos=['ML1_20200328_002539_red','ML1_20220516_172703_red','ML1_20190806_172711_red']
            prioridade=[]
            prioridade+=conhecidos
            prioridade+=diagnostico.sort_values('skeleton_junction_pixels',ascending=False).loc[lambda x:x.skeleton_junction_pixels>0,'id_asta'].tolist()
            prioridade+=manifest.sort_values(['componentes_relevantes','mascara_pixels_positivos'],ascending=False).head(24).id_asta.tolist()
            rng=np.random.default_rng(20260823)
            comuns=sorted(set(manifest.id_asta.astype(str))-set(prioridade))
            amostra=rng.choice(comuns,size=min(24,len(comuns)),replace=False).tolist()
            ids_revisao=[]
            for id_asta in prioridade+amostra:
                if id_asta not in ids_revisao:ids_revisao.append(id_asta)
                if len(ids_revisao)>=80:break

            folhas=LOCAL/'folhas_revisao_skeleton';folhas.mkdir(parents=True,exist_ok=True)
            for antigo in folhas.glob('folha_*.jpg'):antigo.unlink()
            por_folha=12;colunas=4;largura_item,altura_item=384,411
            diag_idx=diagnostico.set_index('id_asta')
            for inicio_idx in range(0,len(ids_revisao),por_folha):
                lote=ids_revisao[inicio_idx:inicio_idx+por_folha];n_linhas=math.ceil(len(lote)/colunas)
                folha=Image.new('RGB',(colunas*largura_item,n_linhas*altura_item),(235,235,235))
                draw=ImageDraw.Draw(folha)
                for pos,id_asta in enumerate(lote):
                    caminho=GEOM_PREVIEWS_DRIVE/(id_asta+'.jpg')
                    with Image.open(caminho) as prev:
                        prev=prev.convert('RGB');prev.thumbnail((largura_item,altura_item-28),Image.Resampling.LANCZOS)
                        x=(pos%colunas)*largura_item+(largura_item-prev.width)//2
                        y=(pos//colunas)*altura_item+28;folha.paste(prev,(x,y))
                    d=diag_idx.loc[id_asta]
                    titulo=f'{id_asta} | comp={int(d.componentes_relevantes)} | j={int(d.skeleton_junction_pixels)}'
                    draw.text(((pos%colunas)*largura_item+5,(pos//colunas)*altura_item+5),titulo,fill='black')
                folha.save(folhas/f'folha_{inicio_idx//por_folha+1:02d}.jpg',quality=90,optimize=True)

            diagnostico_path=FINAL_LOCAL/'diagnostico_skeleton_asta_v3.csv';diagnostico.to_csv(diagnostico_path,index=False)
            zip_path=FINAL_LOCAL/'folhas_revisao_skeleton_asta_v3.zip'
            with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
                for folha in sorted(folhas.glob('folha_*.jpg')):zf.write(folha,arcname=folha.name)
            shutil.copy2(LOCAL/'protocolo_asta_candidato_v3.json',FINAL_LOCAL/'protocolo_asta_candidato_v3.json')
            resumo={
                'protocolo':'asta_protocol_review_v3','status':'VALIDACAO_SKELETON_PRONTA_PARA_REVISAO',
                'pares':len(diagnostico),'mascaras_positivas':int(positivos.sum()),
                'mascaras_vazias':int((~positivos).sum()),'id_mascara_vazia':'ML1_20200328_002539_red',
                'skeletons_positivos':int((diagnostico.skeleton_pixels>0).sum()),
                'frames_com_junction_pixels':int((diagnostico.skeleton_junction_pixels>0).sum()),
                'itens_revisao_dirigida':len(ids_revisao),'erros':erros,
                'sha256_manifest_auditoria':EXPECTED_AUDIT_MANIFEST_SHA256,
                'sha256_protocolo_candidato':CONFIG_PROTOCOLO['sha256_protocolo_candidato'],
                'raw_somente_leitura':True,'yolo_executado':False,'hough_executado':False,
                'tempo_segundos':round(time.time()-inicio,1)}
            resumo['sha256_diagnostico_skeleton']=sha256_arquivo(diagnostico_path)
            resumo['sha256_folhas_revisao']=sha256_arquivo(zip_path)
            resumo_path=FINAL_LOCAL/'resumo_revisao_protocolo_asta_v3.json'
            resumo_path.write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf-8')
            if erros:
                print(json.dumps(resumo,indent=2,ensure_ascii=False));raise RuntimeError('Validação skeleton reprovada.')

            registro={p.name:sha256_arquivo(p) for p in sorted(FINAL_LOCAL.iterdir()) if p.is_file()}
            hashes_path=FINAL_LOCAL/'hashes_rascunho_protocolo_asta_v3.json'
            hashes_path.write_text(json.dumps(registro,indent=2),encoding='utf-8')
            DRAFT_DRIVE.mkdir(parents=True,exist_ok=True)
            for arquivo in tqdm(sorted(FINAL_LOCAL.iterdir()),desc='Publicando rascunho ASTA sem inferência',unit='arq'):
                if arquivo.is_file():shutil.copy2(arquivo,DRAFT_DRIVE/arquivo.name)
            for nome,esperado in registro.items():assert sha256_arquivo(DRAFT_DRIVE/nome)==esperado
            assert sha256_arquivo(DRAFT_DRIVE/hashes_path.name)==sha256_arquivo(hashes_path)
            print(json.dumps(resumo,indent=2,ensure_ascii=False))
            print('BACKUP_RASCUNHO_PROTOCOLO_ASTA_APROVADO_SEM_INFERENCIA')
            print('Envie o resumo e folhas_revisao_skeleton_asta_v3.zip antes de publicar o contrato.')
        ''') ,
        md(r'''
        ## 4. Publicação do contrato após revisão humana

        Deixe `APROVAR_E_PUBLICAR_PROTOCOLO=False` até compartilhar o resumo e o ZIP da seção 3.
        Após a aprovação, altere somente essa chave. O contrato é criado de forma imutável; uma
        versão divergente não sobrescreve a anterior. Mesmo depois deste gate, nenhuma inferência
        é executada neste notebook.
        ''') ,
        code(r'''
        APROVAR_E_PUBLICAR_PROTOCOLO=False

        if not APROVAR_E_PUBLICAR_PROTOCOLO:
            print('Publicação bloqueada. Compartilhe primeiro o resumo e as folhas skeleton.')
        else:
            resumo_path=DRAFT_DRIVE/'resumo_revisao_protocolo_asta_v3.json'
            hashes_path=DRAFT_DRIVE/'hashes_rascunho_protocolo_asta_v3.json'
            assert resumo_path.is_file() and hashes_path.is_file()
            resumo_review=json.loads(resumo_path.read_text(encoding='utf-8'))
            hashes_review=json.loads(hashes_path.read_text(encoding='utf-8'))
            assert resumo_review['status']=='VALIDACAO_SKELETON_PRONTA_PARA_REVISAO'
            assert resumo_review['erros']==[] and resumo_review['pares']==178
            assert resumo_review['sha256_protocolo_candidato']==CONFIG_PROTOCOLO['sha256_protocolo_candidato']
            for nome,esperado in hashes_review.items():assert sha256_arquivo(DRAFT_DRIVE/nome)==esperado

            contrato={
                'protocolo':'asta_external_protocol_v3','status':'APROVADO_SEM_INFERENCIA',
                'aprovado':True,'aprovado_em_utc':datetime.now(timezone.utc).isoformat(),
                'configuracao':CONFIG_PROTOCOLO,
                'sha256_manifest_auditoria':EXPECTED_AUDIT_MANIFEST_SHA256,
                'sha256_componentes':EXPECTED_COMPONENTS_SHA256,
                'sha256_diagnostico_skeleton':resumo_review['sha256_diagnostico_skeleton'],
                'sha256_folhas_revisao':resumo_review['sha256_folhas_revisao'],
                'revisao_visual_csv_aprovada':True,'validacao_skeleton_aprovada':True,
                'raw_somente_leitura':True,'yolo_executado':False,'hough_executado':False,
                'inferencia_asta_executada':False
            }
            contrato['sha256_contrato_protocolo_asta']=sha256_canonico(contrato)
            FINAL_DRIVE.mkdir(parents=True,exist_ok=True)
            contrato_path=FINAL_DRIVE/'contrato_protocolo_asta_v3.json'
            if contrato_path.exists():
                existente=json.loads(contrato_path.read_text(encoding='utf-8'))
                if existente.get('sha256_contrato_protocolo_asta')!=contrato['sha256_contrato_protocolo_asta']:
                    raise RuntimeError('Já existe contrato ASTA diferente. Não sobrescrever.')
                contrato=existente
            else:
                with contrato_path.open('x',encoding='utf-8') as arq:json.dump(contrato,arq,indent=2,ensure_ascii=False)
            for nome in ['resumo_revisao_protocolo_asta_v3.json','diagnostico_skeleton_asta_v3.csv',
                         'folhas_revisao_skeleton_asta_v3.zip','protocolo_asta_candidato_v3.json']:
                origem=DRAFT_DRIVE/nome;destino=FINAL_DRIVE/nome
                if not destino.exists():shutil.copy2(origem,destino)
                assert sha256_arquivo(destino)==sha256_arquivo(origem)
            ponteiro={'protocolo':'asta_external_protocol_v3',
                      'sha256_contrato_protocolo_asta':contrato['sha256_contrato_protocolo_asta'],
                      'contrato_relativo':'experiments/asta_external_v3/protocol/final/contrato_protocolo_asta_v3.json',
                      'inferencia_asta_executada':False}
            gravar_json_atomico(PROTOCOL_DRIVE/'LATEST_ASTA_PROTOCOL_V3.json',ponteiro)
            print(json.dumps(contrato,indent=2,ensure_ascii=False))
            print('PROTOCOLO_ASTA_V3_PUBLICADO_SEM_INFERENCIA')
        ''') ,
        md(r'''
        ## 5. Verificação pós-backup

        Este gate pode ser executado após reconexão. Ele apenas verifica o contrato publicado.
        ''') ,
        code(r'''
        contrato_path=FINAL_DRIVE/'contrato_protocolo_asta_v3.json'
        if not contrato_path.is_file():
            print('Contrato final ainda não foi publicado.')
        else:
            contrato=json.loads(contrato_path.read_text(encoding='utf-8'))
            assert verificar_hash_canonico(contrato,'sha256_contrato_protocolo_asta')==contrato['sha256_contrato_protocolo_asta']
            assert contrato['status']=='APROVADO_SEM_INFERENCIA' and contrato['aprovado'] is True
            assert contrato['raw_somente_leitura'] is True
            assert contrato['yolo_executado'] is False and contrato['hough_executado'] is False
            assert contrato['inferencia_asta_executada'] is False
            print(json.dumps({
                'status':'BACKUP_PROTOCOLO_ASTA_V3_APROVADO_SEM_INFERENCIA',
                'sha256_contrato_protocolo_asta':contrato['sha256_contrato_protocolo_asta'],
                'sha256_manifest_auditoria':contrato['sha256_manifest_auditoria'],
                'sha256_diagnostico_skeleton':contrato['sha256_diagnostico_skeleton'],
                'yolo_executado':False,'hough_executado':False
            },indent=2,ensure_ascii=False))
        ''') ,
    ]
    return write_notebook('12_protocolo_externo_asta_tiles_centerline_sem_inferencia_v3.ipynb', cells)


def build_13():
    cells = [
        md(r'''
        # 13 — Evento externo protegido ASTA: YOLO × Hough (v3)

        Este notebook executa a primeira e única avaliação externa no ASTA. Ele está vinculado
        ao contrato imutável publicado pelo Notebook 12 e não permite recalibração por resultado.

        **Antes de executar qualquer célula, selecione GPU no Colab.** Primeiro execute somente
        até o gate `ASTA_PRONTO_PARA_EVENTO_EXTERNO_UNICO` e compartilhe o relatório. A célula de
        inferência permanece desativada por padrão.

        O processamento é retomável por frame e por método. Cada imagem e máscara é copiada para
        o disco local, conferida por SHA-256, processada e removida; o raw no Drive é somente leitura.
        ''') ,
        code(r'''
        %pip install -q ultralytics==8.4.127 opencv-python-headless pillow pandas tqdm scikit-image

        from google.colab import drive
        from pathlib import Path
        from datetime import datetime, timezone
        from tqdm.auto import tqdm
        from PIL import Image, ImageDraw
        from skimage.morphology import skeletonize
        import gc, hashlib, json, math, os, shutil, time, uuid, zipfile
        import cv2
        import numpy as np
        import pandas as pd
        import torch
        import ultralytics

        Image.MAX_IMAGE_PIXELS=200_000_000
        drive.mount('/content/drive',force_remount=True)

        if not torch.cuda.is_available():
            raise RuntimeError('GPU CUDA ausente. Troque o runtime para GPU antes de continuar.')
        livre,total=torch.cuda.mem_get_info()
        if livre/1024**3<10:
            raise RuntimeError(f'VRAM livre insuficiente: {livre/1024**3:.2f} GB. Reinicie o runtime GPU.')
        if ultralytics.__version__!='8.4.127':
            raise RuntimeError(f'Ultralytics divergente: {ultralytics.__version__}')
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        ASTA_RAW=BACKUP_DIR/'data/raw/asta_real'
        AUDIT_FINAL=BACKUP_DIR/'experiments/asta_external_v3/audit/final'
        PROTOCOL_FINAL=BACKUP_DIR/'experiments/asta_external_v3/protocol/final'
        PROTOCOL_ROOT=BACKUP_DIR/'experiments/asta_external_v3/protocol'
        INFERENCE_ROOT=BACKUP_DIR/'experiments/asta_external_v3/inference'
        FINAL_MODELS=BACKUP_DIR/'experiments/yolo_final_multiseed_v3/synthetic_v3_c41d5091de56'
        HOUGH_CONFIG_PATH=BACKUP_DIR/'data/synthetic_runs/synthetic_v3_c41d5091de56/hough_v3/config_congelada_v3.json'
        LOCAL=Path('/content/asta_external_event_v3')
        STAGING=LOCAL/'staging_frame'
        LOCAL_WEIGHT=LOCAL/'yolo11n_obb_seed_20260824_best.pt'
        LOCK_PATH=INFERENCE_ROOT/'TRAVA_EVENTO_EXTERNO_ASTA_V3.json'
        for pasta in [LOCAL,STAGING,INFERENCE_ROOT]:pasta.mkdir(parents=True,exist_ok=True)

        EXPECTED_PROTOCOL_SHA256='dc4c58381b1e3e7d2a24f7fdac8fc59276e53b9d900853c1191a497319e862bc'
        EXPECTED_AUDIT_MANIFEST_SHA256='91bfa07ebffceaa46271a254d3862348ce2b242a06ed44fc4c07efeeb58722ca'
        EXPECTED_SKELETON_DIAGNOSTIC_SHA256='cd6a2ccce5a65fee093157c03570259b33f830aa6e6f88286390514cb7bd8388'
        EXPECTED_OPERATIONAL_WEIGHT_SHA256='a2b64773f32868ac952d81c33b331f1759b2250a3a04f49d5ab105f7bb79dd0c'
        EXPECTED_HOUGH_CONFIG_SHA256='14ef0ea56a2e86bf1581225c4b9a0733d482834e5f7d3fab18b3eaf76f03a2e5'
        EXPECTED_FINAL_MODELS_CONTRACT_SHA256='0a8d3537969dcb033dc8cdfa1a9d63dc24de4cc98e212c42fab62621126890a7'
        BATCH_TILES_YOLO=16

        def agora():return datetime.now().strftime('%H:%M:%S')
        def sha256_arquivo(caminho,chunk=8*1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as arq:
                for bloco in iter(lambda:arq.read(chunk),b''):h.update(bloco)
            return h.hexdigest()
        def sha256_canonico(objeto):
            texto=json.dumps(objeto,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            return hashlib.sha256(texto.encode('utf-8')).hexdigest()
        def verificar_hash_canonico(registro,campo,ensure_ascii=False):
            copia=dict(registro);observado=copia.pop(campo)
            texto=json.dumps(copia,sort_keys=True,separators=(',',':'),ensure_ascii=ensure_ascii)
            calculado=hashlib.sha256(texto.encode('utf-8')).hexdigest()
            if calculado!=observado:raise RuntimeError(f'Hash canônico divergente: {campo}')
            return observado
        def gravar_json_atomico(caminho,objeto):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            temporario=caminho.with_name(caminho.name+'.tmp')
            temporario.write_text(json.dumps(objeto,indent=2,ensure_ascii=False),encoding='utf-8')
            os.replace(temporario,caminho)
        def gravar_csv_atomico(caminho,tabela):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            temporario=caminho.with_name(caminho.name+'.tmp');tabela.to_csv(temporario,index=False)
            os.replace(temporario,caminho)

        print(json.dumps({
            'status':'GPU_ASTA_APROVADA','gpu':torch.cuda.get_device_name(0),
            'vram_total_gb':round(total/1024**3,2),'vram_livre_gb':round(livre/1024**3,2),
            'torch':torch.__version__,'cuda':torch.version.cuda,'ultralytics':ultralytics.__version__
        },indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 1. Gate pré-evento — sem inferência

        Este gate valida o contrato publicado, o inventário, o checkpoint operacional e o Hough
        congelado. Os arquivos grandes não são copiados aqui: cada par será conferido por hash no
        momento em que for processado. Compartilhe o JSON antes de habilitar a seção 3.
        ''') ,
        code(r'''
        contrato_path=PROTOCOL_FINAL/'contrato_protocolo_asta_v3.json'
        ponteiro_path=PROTOCOL_ROOT/'LATEST_ASTA_PROTOCOL_V3.json'
        manifest_path=AUDIT_FINAL/'manifest_auditoria_asta_v3.csv'
        diagnostico_path=PROTOCOL_FINAL/'diagnostico_skeleton_asta_v3.csv'
        assert contrato_path.is_file() and ponteiro_path.is_file()

        contrato=json.loads(contrato_path.read_text(encoding='utf-8'))
        ponteiro=json.loads(ponteiro_path.read_text(encoding='utf-8'))
        assert verificar_hash_canonico(contrato,'sha256_contrato_protocolo_asta')==EXPECTED_PROTOCOL_SHA256
        assert ponteiro['sha256_contrato_protocolo_asta']==EXPECTED_PROTOCOL_SHA256
        assert contrato['status']=='APROVADO_SEM_INFERENCIA' and contrato['aprovado'] is True
        assert contrato['raw_somente_leitura'] is True
        assert contrato['yolo_executado'] is False and contrato['hough_executado'] is False
        assert contrato['inferencia_asta_executada'] is False
        cfg=contrato['configuracao']
        assert cfg['sha256_protocolo_candidato']=='33afc29c4479c0129bad8f4854f373215d588bc2566d54404e5770fc94bc02c1'
        assert cfg['tiles']=={'tamanho':512,'overlap':128,'stride':384,'grid_x':28,'grid_y':28,
            'tiles_por_frame':784,'tiles_total':139552,'padding_direita':320,'padding_abaixo':320,
            'grid_regular':True,'descartar_predicao_centro_fora_frame':True}
        assert cfg['modelo_yolo']['seed_operacional']==20260824
        assert cfg['modelo_yolo']['threshold_confianca']==0.45
        assert cfg['metricas']['primaria']=='f1_centerline_macro_frames_positivos'
        assert cfg['metricas']['tolerancia_centerline_px']==14.0
        assert cfg['restricoes']['ajuste_por_resultado_asta'] is False

        assert sha256_arquivo(manifest_path)==EXPECTED_AUDIT_MANIFEST_SHA256
        assert sha256_arquivo(diagnostico_path)==EXPECTED_SKELETON_DIAGNOSTIC_SHA256
        manifest=pd.read_csv(manifest_path).sort_values('id_asta').reset_index(drop=True)
        assert len(manifest)==178 and manifest.id_asta.nunique()==178
        assert int((manifest.mascara_pixels_positivos>0).sum())==177
        assert manifest.loc[manifest.mascara_pixels_positivos==0,'id_asta'].tolist()==['ML1_20200328_002539_red']
        for linha in manifest.itertuples():
            if not (ASTA_RAW/linha.arquivo_imagem).is_file():raise FileNotFoundError(linha.arquivo_imagem)
            if not (ASTA_RAW/linha.arquivo_mascara).is_file():raise FileNotFoundError(linha.arquivo_mascara)

        contrato_modelos=json.loads((FINAL_MODELS/'contrato_modelos_finais_v3.json').read_text(encoding='utf-8'))
        assert verificar_hash_canonico(contrato_modelos,'sha256_contrato_modelos_finais',ensure_ascii=True)==EXPECTED_FINAL_MODELS_CONTRACT_SHA256
        operacional=next(x for x in contrato_modelos['checkpoints'] if int(x['seed'])==20260824)
        peso_operacional=BACKUP_DIR/operacional['best_pt_relativo_drive']
        assert peso_operacional.is_file() and sha256_arquivo(peso_operacional)==EXPECTED_OPERATIONAL_WEIGHT_SHA256
        hough_config=json.loads(HOUGH_CONFIG_PATH.read_text(encoding='utf-8'))
        assert hough_config['sha256_config']==EXPECTED_HOUGH_CONFIG_SHA256

        CONFIG_EVENTO={
            'protocolo':'asta_external_protected_event_v3','versao':'3.0',
            'sha256_contrato_protocolo_asta':EXPECTED_PROTOCOL_SHA256,
            'sha256_manifest_auditoria':EXPECTED_AUDIT_MANIFEST_SHA256,
            'sha256_checkpoint_yolo':EXPECTED_OPERATIONAL_WEIGHT_SHA256,
            'sha256_hough_config':EXPECTED_HOUGH_CONFIG_SHA256,
            'frames':178,'positivos':177,'negativos':1,'tiles_total':139552,
            'batch_tiles_yolo':BATCH_TILES_YOLO,'dispositivo_yolo':'cuda:0','precisao_yolo':'fp32',
            'determinismo':{'cudnn_benchmark':False,'cudnn_deterministic':True,
                'hough_rng':'seed_sha256_por_frame_tile'},
            'checkpoint':'por_frame_e_metodo','retomada':True,
            'timing':{'preprocessamento':'decode_e_normalizacao_da_imagem_mais_extracao_dos_tiles',
                'inferencia':'chamada_do_detector_e_conversao_local_das_saidas_com_sincronizacao_cuda',
                'merging':'uniao_global_de_segmentos',
                'total_frame':'preprocessamento_mais_inferencia_mais_merging',
                'exclui':['copia_drive','decode_mascara','skeleton_ground_truth','avaliacao']},
            'resultado_primario':'media_macro_f1_centerline_nos_177_frames_positivos',
            'comparacao':'delta_pareado_yolo_menos_hough_por_frame',
            'bootstrap_replicacoes':2000,'bootstrap_seed':20260823,
            'ajuste_pos_resultado':False
        }
        CONFIG_EVENTO['sha256_config_evento']=sha256_canonico(CONFIG_EVENTO)
        gravar_json_atomico(LOCAL/'config_evento_asta_v3.json',CONFIG_EVENTO)
        evento_existente=json.loads(LOCK_PATH.read_text(encoding='utf-8')) if LOCK_PATH.is_file() else None
        gate={
            'status':'ASTA_PRONTO_PARA_EVENTO_EXTERNO_UNICO',
            'sha256_contrato_protocolo_asta':EXPECTED_PROTOCOL_SHA256,
            'sha256_config_evento':CONFIG_EVENTO['sha256_config_evento'],
            'pares':178,'tiles_total':139552,'checkpoint_operacional_seed':20260824,
            'sha256_checkpoint_operacional':EXPECTED_OPERATIONAL_WEIGHT_SHA256,
            'sha256_hough_config':EXPECTED_HOUGH_CONFIG_SHA256,
            'gpu':torch.cuda.get_device_name(0),'batch_tiles_yolo':BATCH_TILES_YOLO,
            'evento_existente':None if evento_existente is None else {
                'event_id':evento_existente.get('event_id'),'status':evento_existente.get('status'),
                'sha256_config_evento':evento_existente.get('sha256_config_evento')},
            'inferencia_executada_nesta_sessao':False
        }
        print(json.dumps(gate,indent=2,ensure_ascii=False))
        print('Gate pré-evento aprovado. Não habilite a inferência antes de compartilhar este relatório.')
        ''') ,
        md(r'''
        ## 2. Funções congeladas e testes determinísticos

        A máscara oficial continua semântica. O matching compara pixels das centerlines com
        tolerância simétrica de 14 px. Fragmentos de tiles só são unidos quando satisfazem
        simultaneamente ângulo, distância perpendicular e gap axial do contrato.
        ''') ,
        code(r'''
        TILE=cfg['tiles']['tamanho'];STRIDE=cfg['tiles']['stride']
        STARTS_X=[i*STRIDE for i in range(cfg['tiles']['grid_x'])]
        STARTS_Y=[i*STRIDE for i in range(cfg['tiles']['grid_y'])]

        def copiar_com_hash(origem,destino,esperado,tentativas=3,chunk=8*1024*1024):
            ultimo=None
            for tentativa in range(1,tentativas+1):
                try:
                    if destino.exists():destino.unlink()
                    h=hashlib.sha256()
                    with origem.open('rb') as entrada,destino.open('wb') as saida:
                        for bloco in iter(lambda:entrada.read(chunk),b''):
                            saida.write(bloco);h.update(bloco)
                    observado=h.hexdigest()
                    if observado!=esperado:raise IOError(f'hash divergente: {observado}')
                    return observado
                except (OSError,IOError) as exc:
                    ultimo=exc
                    if destino.exists():destino.unlink()
                    if tentativa<tentativas:time.sleep(2**tentativa)
            raise IOError(f'Falha após {tentativas} tentativas: {ultimo}')

        def percentil_histograma_uint8(imagem,percentil):
            hist=np.bincount(imagem.reshape(-1),minlength=256)
            alvo=(imagem.size-1)*(float(percentil)/100.0)
            return int(np.searchsorted(np.cumsum(hist),alvo,side='right'))

        def normalizar_frame_uint8(imagem):
            baixo=percentil_histograma_uint8(imagem,cfg['preprocessamento']['percentil_baixo'])
            alto=percentil_histograma_uint8(imagem,cfg['preprocessamento']['percentil_alto'])
            if alto<=baixo:alto=min(255,baixo+1)
            valores=np.arange(256,dtype=np.float32)
            lut=np.clip((valores-baixo)*255.0/(alto-baixo),0,255).round().astype(np.uint8)
            normalizada=lut[imagem]
            mediana=percentil_histograma_uint8(normalizada,50.0)
            return normalizada,{'percentil_baixo_valor':baixo,'percentil_alto_valor':alto,'mediana_normalizada':mediana}

        def extrair_tile(imagem,x,y,preenchimento):
            h,w=imagem.shape;tile=np.full((TILE,TILE),preenchimento,dtype=np.uint8)
            valido_h=max(0,min(TILE,h-y));valido_w=max(0,min(TILE,w-x))
            if valido_h and valido_w:tile[:valido_h,:valido_w]=imagem[y:y+valido_h,x:x+valido_w]
            return tile

        def caixa_para_eixo_maior(pontos):
            pts=np.asarray(pontos,dtype=float).reshape(4,2);centro=pts.mean(axis=0)
            cov=(pts-centro).T@(pts-centro)
            valores,vetores=np.linalg.eigh(cov);direcao=vetores[:,int(np.argmax(valores))]
            proj=(pts-centro)@direcao;menor,maior=float(proj.min()),float(proj.max())
            p1=centro+direcao*menor;p2=centro+direcao*maior
            return p1,p2

        def angulo_segmento(seg):
            dx=seg['x2']-seg['x1'];dy=seg['y2']-seg['y1']
            return math.degrees(math.atan2(dy,dx))%180.0

        def diferenca_angular(a,b):
            d=abs(a-b)%180.0
            return min(d,180.0-d)

        def segmentos_compativeis(a,b):
            limite=cfg['merging_global']
            aa,ab=angulo_segmento(a),angulo_segmento(b)
            if diferenca_angular(aa,ab)>limite['angulo_max_graus']:return False
            va=np.array([a['x2']-a['x1'],a['y2']-a['y1']],float)
            vb=np.array([b['x2']-b['x1'],b['y2']-b['y1']],float)
            va/=max(np.linalg.norm(va),1e-9);vb/=max(np.linalg.norm(vb),1e-9)
            if np.dot(va,vb)<0:vb=-vb
            eixo=va+vb;eixo/=max(np.linalg.norm(eixo),1e-9);normal=np.array([-eixo[1],eixo[0]])
            ma=np.array([(a['x1']+a['x2'])/2,(a['y1']+a['y2'])/2])
            mb=np.array([(b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2])
            if abs(float((mb-ma)@normal))>limite['distancia_perpendicular_max_px']:return False
            pa=sorted([float(np.array([a['x1'],a['y1']])@eixo),float(np.array([a['x2'],a['y2']])@eixo)])
            pb=sorted([float(np.array([b['x1'],b['y1']])@eixo),float(np.array([b['x2'],b['y2']])@eixo)])
            gap=max(0.0,max(pa[0],pb[0])-min(pa[1],pb[1]))
            return gap<=limite['gap_axial_max_px']

        def agregar_componente(segmentos,indices):
            pontos=[];pesos=[]
            for indice in indices:
                s=segmentos[indice];peso=max(float(s['peso']),1e-6)
                pontos.extend([[s['x1'],s['y1']],[s['x2'],s['y2']]]);pesos.extend([peso,peso])
            pts=np.asarray(pontos,float);w=np.asarray(pesos,float);centro=np.average(pts,axis=0,weights=w)
            desloc=pts-centro;cov=(desloc*w[:,None]).T@desloc/max(w.sum(),1e-9)
            valores,vetores=np.linalg.eigh(cov);eixo=vetores[:,int(np.argmax(valores))]
            proj=desloc@eixo;p1=centro+eixo*proj.min();p2=centro+eixo*proj.max()
            confiancas=[float(segmentos[i].get('confianca',0.0)) for i in indices]
            return {'x1':float(p1[0]),'y1':float(p1[1]),'x2':float(p2[0]),'y2':float(p2[1]),
                    'comprimento':float(np.linalg.norm(p2-p1)),'confianca_max':max(confiancas,default=0.0),
                    'fragmentos':len(indices)}

        def unir_segmentos(segmentos):
            """Mesmo grafo do predicado escalar, com pares avaliados em blocos NumPy."""
            n=len(segmentos)
            if n==0:return []
            pai=list(range(n))
            def raiz(i):
                while pai[i]!=i:pai[i]=pai[pai[i]];i=pai[i]
                return i
            def unir(i,j):
                ri,rj=raiz(i),raiz(j)
                if ri!=rj:pai[rj]=ri

            p1=np.asarray([[s['x1'],s['y1']] for s in segmentos],dtype=np.float64)
            p2=np.asarray([[s['x2'],s['y2']] for s in segmentos],dtype=np.float64)
            vetores=p2-p1
            normas=np.linalg.norm(vetores,axis=1)
            unidades=vetores/np.maximum(normas[:,None],1e-9)
            angulos=(np.degrees(np.arctan2(vetores[:,1],vetores[:,0]))%180.0)
            meios=(p1+p2)/2.0
            limite=cfg['merging_global']
            max_angulo=float(limite['angulo_max_graus'])
            max_distancia=float(limite['distancia_perpendicular_max_px'])
            max_gap=float(limite['gap_axial_max_px'])

            def avaliar_blocos(indices_a,indices_b,mesmo_bucket):
                ia_total=np.asarray(indices_a,dtype=np.int64)
                ib=np.asarray(indices_b,dtype=np.int64)
                if ia_total.size==0 or ib.size==0:return
                # Limita matrizes temporárias e mantém o consumo de RAM independente do número total de segmentos.
                bloco_a=128
                for inicio in range(0,len(ia_total),bloco_a):
                    ia=ia_total[inicio:inicio+bloco_a]
                    dif=np.abs(angulos[ia,None]-angulos[ib][None,:])%180.0
                    mascara=np.minimum(dif,180.0-dif)<=max_angulo
                    if mesmo_bucket:
                        mascara&=ib[None,:]>ia[:,None]
                    pos_a,pos_b=np.nonzero(mascara)
                    if pos_a.size==0:continue
                    ai=ia[pos_a];bi=ib[pos_b]
                    va=unidades[ai];vb=unidades[bi].copy()
                    inverter=np.einsum('ij,ij->i',va,vb)<0
                    vb[inverter]*=-1.0
                    eixos=va+vb
                    eixos/=np.maximum(np.linalg.norm(eixos,axis=1)[:,None],1e-9)
                    normais=np.column_stack((-eixos[:,1],eixos[:,0]))
                    distancia=np.abs(np.einsum('ij,ij->i',meios[bi]-meios[ai],normais))
                    manter=distancia<=max_distancia
                    if not np.any(manter):continue
                    ai=ai[manter];bi=bi[manter];eixos=eixos[manter]
                    a1=np.einsum('ij,ij->i',p1[ai],eixos);a2=np.einsum('ij,ij->i',p2[ai],eixos)
                    b1=np.einsum('ij,ij->i',p1[bi],eixos);b2=np.einsum('ij,ij->i',p2[bi],eixos)
                    amin=np.minimum(a1,a2);amax=np.maximum(a1,a2)
                    bmin=np.minimum(b1,b2);bmax=np.maximum(b1,b2)
                    gap=np.maximum(0.0,np.maximum(amin,bmin)-np.minimum(amax,bmax))
                    for i,j in zip(ai[gap<=max_gap].tolist(),bi[gap<=max_gap].tolist()):unir(i,j)

            buckets={}
            for i,s in enumerate(segmentos):buckets.setdefault((int(s['tile_x']),int(s['tile_y'])),[]).append(i)
            pares_processados=set()
            for tx,ty in buckets:
                chave_a=(tx,ty)
                for ny in range(ty-1,ty+2):
                    for nx in range(tx-1,tx+2):
                        chave_b=(nx,ny)
                        if chave_b not in buckets:continue
                        par=(chave_a,chave_b) if chave_a<=chave_b else (chave_b,chave_a)
                        if par in pares_processados:continue
                        pares_processados.add(par)
                        avaliar_blocos(buckets[par[0]],buckets[par[1]],par[0]==par[1])
            grupos={}
            for i in range(n):grupos.setdefault(raiz(i),[]).append(i)
            return [agregar_componente(segmentos,indices) for indices in grupos.values()]

        def rasterizar_segmentos(segmentos,shape):
            h,w=shape;canvas=np.zeros((h,w),np.uint8)
            for s in segmentos:
                p1=(int(round(s['x1'])),int(round(s['y1'])));p2=(int(round(s['x2'])),int(round(s['y2'])))
                ok,q1,q2=cv2.clipLine((0,0,w,h),p1,p2)
                if ok:cv2.line(canvas,q1,q2,1,1,cv2.LINE_8)
            return canvas

        def metricas_centerline(segmentos,skeleton_gt,tolerancia=14):
            pred=rasterizar_segmentos(segmentos,skeleton_gt.shape);gt=skeleton_gt.astype(np.uint8)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerancia+1,2*tolerancia+1))
            gt_dilatado=cv2.dilate(gt,kernel);pred_dilatado=cv2.dilate(pred,kernel)
            pred_pixels=int(pred.sum());gt_pixels=int(gt.sum())
            pred_hits=int(((pred>0)&(gt_dilatado>0)).sum())
            gt_hits=int(((gt>0)&(pred_dilatado>0)).sum())
            precisao=(pred_hits/pred_pixels) if pred_pixels else (1.0 if gt_pixels==0 else 0.0)
            recall=(gt_hits/gt_pixels) if gt_pixels else None
            f1=(2*precisao*recall/(precisao+recall)) if recall is not None and (precisao+recall)>0 else (0.0 if gt_pixels else None)
            return {'pred_pixels':pred_pixels,'gt_pixels':gt_pixels,'pred_hits':pred_hits,'gt_hits':gt_hits,
                    'precisao_centerline':float(precisao),'recall_centerline':None if recall is None else float(recall),
                    'f1_centerline':None if f1 is None else float(f1),'cobertura_gt':None if recall is None else float(recall)}

        # Regressões locais: grid, normalização, merging e tolerância centerline.
        assert len(STARTS_X)==len(STARTS_Y)==28 and STARTS_X[-1]+TILE==10880
        teste=np.arange(256,dtype=np.uint8).reshape(16,16);norm,meta=normalizar_frame_uint8(teste)
        assert norm.dtype==np.uint8 and norm.min()==0 and norm.max()==255 and meta['percentil_alto_valor']>meta['percentil_baixo_valor']
        base={'x1':10.,'y1':20.,'x2':300.,'y2':20.,'peso':1.,'confianca':0.9,'tile_x':0,'tile_y':0}
        duplicado={'x1':280.,'y1':21.,'x2':560.,'y2':21.,'peso':1.,'confianca':0.8,'tile_x':1,'tile_y':0}
        paralelo={'x1':10.,'y1':50.,'x2':300.,'y2':50.,'peso':1.,'confianca':0.7,'tile_x':0,'tile_y':0}
        assert len(unir_segmentos([base,duplicado]))==1 and len(unir_segmentos([base,paralelo]))==2

        def _unir_segmentos_escalar_referencia(segmentos):
            n=len(segmentos)
            if n==0:return []
            pai=list(range(n))
            def raiz(i):
                while pai[i]!=i:pai[i]=pai[pai[i]];i=pai[i]
                return i
            def unir(i,j):
                ri,rj=raiz(i),raiz(j)
                if ri!=rj:pai[rj]=ri
            buckets={}
            for i,s in enumerate(segmentos):buckets.setdefault((int(s['tile_x']),int(s['tile_y'])),[]).append(i)
            for (tx,ty),indices in buckets.items():
                for ny in range(ty-1,ty+2):
                    for nx in range(tx-1,tx+2):
                        for i in indices:
                            for j in buckets.get((nx,ny),[]):
                                if j>i and segmentos_compativeis(segmentos[i],segmentos[j]):unir(i,j)
            grupos={}
            for i in range(n):grupos.setdefault(raiz(i),[]).append(i)
            return [agregar_componente(segmentos,indices) for indices in grupos.values()]

        rng_teste=np.random.default_rng(20260830);segmentos_teste=[]
        for _ in range(180):
            tx=int(rng_teste.integers(0,4));ty=int(rng_teste.integers(0,4))
            x=float(tx*384+rng_teste.uniform(0,512));y=float(ty*384+rng_teste.uniform(0,512))
            ang=float(rng_teste.uniform(0,np.pi));comprimento=float(rng_teste.uniform(128,500))
            segmentos_teste.append({'x1':x,'y1':y,'x2':x+comprimento*np.cos(ang),'y2':y+comprimento*np.sin(ang),
                'peso':comprimento,'confianca':0.0,'comprimento':comprimento,'tile_x':tx,'tile_y':ty})
        def _canonico_segmentos(lista):
            return sorted((round(s['x1'],8),round(s['y1'],8),round(s['x2'],8),round(s['y2'],8),
                round(s['comprimento'],8),int(s['fragmentos'])) for s in lista)
        assert _canonico_segmentos(unir_segmentos(segmentos_teste))==_canonico_segmentos(_unir_segmentos_escalar_referencia(segmentos_teste))
        gt=np.zeros((128,128),bool);gt[64,10:110]=True
        identico=metricas_centerline([{'x1':10,'y1':64,'x2':109,'y2':64}],gt,14)
        distante=metricas_centerline([{'x1':10,'y1':90,'x2':109,'y2':90}],gt,14)
        assert identico['f1_centerline']==1.0 and distante['f1_centerline']==0.0
        print('Testes determinísticos do protocolo externo aprovados; nenhuma inferência executada.')
        ''') ,
        md(r'''
        ## 3. Smoke operacional sem dados ASTA

        Este smoke carrega o checkpoint operacional e executa dois tiles artificiais. Ele valida
        lote, GPU, OBB e Hough sem ler nenhuma imagem ou máscara ASTA e sem criar a trava do evento.
        ''') ,
        code(r'''
        print(f'[{agora()}] Smoke operacional sem dados ASTA: início')
        copiar_com_hash(peso_operacional,LOCAL_WEIGHT,EXPECTED_OPERATIONAL_WEIGHT_SHA256)
        from ultralytics import YOLO
        modelo_smoke=YOLO(str(LOCAL_WEIGHT))
        tile_vazio=np.zeros((512,512),np.uint8)
        tile_linha=np.zeros((512,512),np.uint8);cv2.line(tile_linha,(24,420),(488,70),220,3,cv2.LINE_AA)
        entradas_smoke=[cv2.cvtColor(x,cv2.COLOR_GRAY2BGR) for x in [tile_vazio,tile_linha]]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();inicio_smoke=time.perf_counter()
        resultados_smoke=modelo_smoke.predict(source=entradas_smoke,imgsz=512,conf=0.45,iou=0.7,
            max_det=100,device=0,verbose=False,batch=2,half=False)
        torch.cuda.synchronize();tempo_smoke=time.perf_counter()-inicio_smoke
        assert len(resultados_smoke)==2
        caixas_smoke=[0 if r.obb is None else int(len(r.obb)) for r in resultados_smoke]
        d_smoke=hough_config['detector'];bordas_smoke=cv2.Canny(tile_linha,int(d_smoke['canny_lo']),int(d_smoke['canny_hi']))
        cv2.setRNGSeed(20260823)
        linhas_smoke=cv2.HoughLinesP(bordas_smoke,1,np.pi/180,threshold=int(d_smoke['hough_threshold']),
            minLineLength=max(1,int(round(512*float(d_smoke['min_length_frac'])))),maxLineGap=int(d_smoke['max_gap']))
        assert linhas_smoke is not None and np.asarray(linhas_smoke).reshape(-1,4).shape[0]>0
        pico_vram=torch.cuda.max_memory_allocated()/1024**3
        relatorio_smoke={'status':'SMOKE_IMPLEMENTACAO_EVENTO_ASTA_APROVADO_SEM_DADOS_ASTA',
            'sha256_config_evento':CONFIG_EVENTO['sha256_config_evento'],'tiles_artificiais':2,
            'resultados_yolo':len(resultados_smoke),'obb_por_tile':caixas_smoke,
            'linhas_hough_tile_artificial':int(np.asarray(linhas_smoke).reshape(-1,4).shape[0]),
            'tempo_yolo_lote_segundos':round(tempo_smoke,4),'pico_vram_gb':round(pico_vram,3),
            'asta_lido':False,'trava_evento_criada':False}
        print(json.dumps(relatorio_smoke,indent=2,ensure_ascii=False))
        del modelo_smoke,resultados_smoke,entradas_smoke,tile_vazio,tile_linha,bordas_smoke,linhas_smoke
        gc.collect();torch.cuda.empty_cache()
        ''') ,
        md(r'''
        ## 4. Evento único, retomável por frame e por método

        Após a aprovação do gate pré-evento, altere apenas `EXECUTAR_EVENTO_ASTA=True`.
        Em uma sessão retomada, use também `RETOMAR_EVENTO_INCOMPLETO=True`. Nunca use a opção
        de retomada para um evento já concluído.

        O progresso possui uma barra externa de 178 frames e barras internas de 784 tiles. O
        checkpoint de cada método é publicado atomicamente logo após o frame ser concluído.
        ''') ,
        code(r'''
        EXECUTAR_EVENTO_ASTA=False
        RETOMAR_EVENTO_INCOMPLETO=False

        if 'relatorio_smoke' not in globals() or relatorio_smoke.get('status')!='SMOKE_IMPLEMENTACAO_EVENTO_ASTA_APROVADO_SEM_DADOS_ASTA':
            raise RuntimeError('Execute e aprove primeiro o smoke operacional sem dados ASTA.')

        def checkpoint_valido(caminho,metodo,id_asta):
            if not caminho.is_file():return None
            registro=json.loads(caminho.read_text(encoding='utf-8'))
            assert verificar_hash_canonico(registro,'sha256_checkpoint')==registro['sha256_checkpoint']
            assert registro['event_id']==EVENT_ID and registro['metodo']==metodo and registro['id_asta']==id_asta
            assert registro['sha256_config_evento']==CONFIG_EVENTO['sha256_config_evento']
            return registro

        def detectar_yolo_tiles(modelo,imagem_norm,preenchimento,id_asta):
            segmentos=[];metas=[];lote=[];tempo=0.0;tempo_tiles=0.0
            coordenadas=[(ix,iy,x,y) for iy,y in enumerate(STARTS_Y) for ix,x in enumerate(STARTS_X)]
            barra=tqdm(total=len(coordenadas),desc=f'{id_asta} YOLO tiles',unit='tile',leave=False)
            for inicio in range(0,len(coordenadas),BATCH_TILES_YOLO):
                t_tiles=time.perf_counter()
                bloco=coordenadas[inicio:inicio+BATCH_TILES_YOLO];lote=[];metas=[]
                for ix,iy,x,y in bloco:
                    tile=extrair_tile(imagem_norm,x,y,preenchimento)
                    lote.append(cv2.cvtColor(tile,cv2.COLOR_GRAY2BGR));metas.append((ix,iy,x,y))
                tempo_tiles+=time.perf_counter()-t_tiles
                torch.cuda.synchronize();t=time.perf_counter()
                resultados=modelo.predict(source=lote,imgsz=cfg['modelo_yolo']['imgsz'],
                    conf=cfg['modelo_yolo']['threshold_confianca'],iou=cfg['modelo_yolo']['iou_nms'],
                    max_det=cfg['modelo_yolo']['max_det'],device=0,verbose=False,batch=len(lote),half=False)
                assert len(resultados)==len(metas)
                for resultado,(ix,iy,x,y) in zip(resultados,metas):
                    if resultado.obb is None:continue
                    caixas=resultado.obb.xyxyxyxy.detach().cpu().numpy()
                    confs=resultado.obb.conf.detach().cpu().numpy()
                    for caixa,confianca in zip(caixas,confs):
                        p1,p2=caixa_para_eixo_maior(caixa);p1+=np.array([x,y]);p2+=np.array([x,y])
                        centro=(p1+p2)/2
                        if not (0<=centro[0]<10560 and 0<=centro[1]<10560):continue
                        comprimento=float(np.linalg.norm(p2-p1))
                        segmentos.append({'x1':float(p1[0]),'y1':float(p1[1]),'x2':float(p2[0]),'y2':float(p2[1]),
                            'peso':float(confianca),'confianca':float(confianca),'comprimento':comprimento,
                            'tile_x':ix,'tile_y':iy})
                torch.cuda.synchronize();tempo+=time.perf_counter()-t
                barra.update(len(bloco));del resultados,lote
            barra.close();return segmentos,tempo,tempo_tiles

        def detectar_hough_tiles(imagem_norm,preenchimento,id_asta):
            d=hough_config['detector'];segmentos=[];tempo=0.0;tempo_tiles=0.0
            coordenadas=[(ix,iy,x,y) for iy,y in enumerate(STARTS_Y) for ix,x in enumerate(STARTS_X)]
            for ix,iy,x,y in tqdm(coordenadas,desc=f'{id_asta} Hough tiles',unit='tile',leave=False):
                t_tiles=time.perf_counter();tile=extrair_tile(imagem_norm,x,y,preenchimento)
                tempo_tiles+=time.perf_counter()-t_tiles;t=time.perf_counter()
                base=tile
                if float(d.get('pre_blur_sigma',0.0))>0:
                    base=cv2.GaussianBlur(tile,(0,0),float(d['pre_blur_sigma']))
                bordas=cv2.Canny(base,int(d['canny_lo']),int(d['canny_hi']))
                seed_hough=int(hashlib.sha256(f'{id_asta}|{ix}|{iy}|hough_v3'.encode()).hexdigest()[:8],16)&0x7fffffff
                cv2.setRNGSeed(seed_hough)
                linhas=cv2.HoughLinesP(bordas,1,np.pi/180,threshold=int(d['hough_threshold']),
                    minLineLength=max(1,int(round(TILE*float(d['min_length_frac'])))),maxLineGap=int(d['max_gap']))
                if linhas is not None:
                    for x1,y1,x2,y2 in np.asarray(linhas).reshape(-1,4):
                        p1=np.array([float(x1+x),float(y1+y)]);p2=np.array([float(x2+x),float(y2+y)])
                        centro=(p1+p2)/2
                        if not (0<=centro[0]<10560 and 0<=centro[1]<10560):continue
                        comprimento=float(np.linalg.norm(p2-p1))
                        segmentos.append({'x1':float(p1[0]),'y1':float(p1[1]),'x2':float(p2[0]),'y2':float(p2[1]),
                            'peso':comprimento,'confianca':0.0,'comprimento':comprimento,'tile_x':ix,'tile_y':iy})
                tempo+=time.perf_counter()-t
            return segmentos,tempo,tempo_tiles

        def criar_registro_frame(metodo,linha,segmentos_brutos,segmentos_unidos,metricas,tempos,meta_norm):
            registro={'protocolo':'asta_external_frame_checkpoint_v3','event_id':EVENT_ID,
                'sha256_config_evento':CONFIG_EVENTO['sha256_config_evento'],'metodo':metodo,
                'id_asta':linha.id_asta,'sha256_imagem':linha.sha256_imagem,'sha256_mascara':linha.sha256_mascara,
                'gt_positivo':bool(int(linha.mascara_pixels_positivos)>0),'segmentos_brutos':len(segmentos_brutos),
                'segmentos_apos_merging':len(segmentos_unidos),'segmentos':segmentos_unidos,
                'metricas':metricas,'normalizacao':meta_norm,'tempos_segundos':tempos,'concluido':True}
            registro['sha256_checkpoint']=sha256_canonico(registro);return registro

        def carregar_registros(event_dir,metodo):
            registros=[]
            for caminho in sorted((event_dir/'checkpoints'/metodo).glob('*.json')):
                registro=json.loads(caminho.read_text(encoding='utf-8'))
                assert verificar_hash_canonico(registro,'sha256_checkpoint')==registro['sha256_checkpoint']
                assert registro['event_id']==EVENT_ID and registro['metodo']==metodo
                assert registro['sha256_config_evento']==CONFIG_EVENTO['sha256_config_evento']
                registros.append(registro)
            return registros

        def consolidar_metodo(registros,metodo):
            positivos=[r for r in registros if r['gt_positivo']];negativos=[r for r in registros if not r['gt_positivo']]
            pred=sum(r['metricas']['pred_pixels'] for r in positivos);pred_hits=sum(r['metricas']['pred_hits'] for r in positivos)
            gt=sum(r['metricas']['gt_pixels'] for r in positivos);gt_hits=sum(r['metricas']['gt_hits'] for r in positivos)
            p=pred_hits/pred if pred else 0.0;r=gt_hits/gt if gt else 0.0;f1=2*p*r/(p+r) if p+r else 0.0
            macro=float(np.mean([x['metricas']['f1_centerline'] for x in positivos]))
            area_mp=(10560*10560)/1e6
            fp_mp=float(np.mean([x['metricas']['pred_pixels']/area_mp for x in negativos])) if negativos else None
            taxa_fp=float(np.mean([x['metricas']['pred_pixels']>0 for x in negativos])) if negativos else None
            return {'metodo':metodo,'frames':len(registros),'frames_positivos':len(positivos),'frames_negativos':len(negativos),
                'f1_centerline_macro_frames_positivos':macro,'precisao_centerline_pooled':float(p),
                'recall_centerline_pooled':float(r),'f1_centerline_pooled':float(f1),
                'comprimento_fp_por_megapixel_negativo':fp_mp,'taxa_frames_negativos_com_fp':taxa_fp,
                'tempo_preprocessamento_medio_s':float(np.mean([x['tempos_segundos']['preprocessamento'] for x in registros])),
                'tempo_inferencia_medio_s':float(np.mean([x['tempos_segundos']['inferencia'] for x in registros])),
                'tempo_merging_medio_s':float(np.mean([x['tempos_segundos']['merging'] for x in registros])),
                'tempo_total_frame_medio_s':float(np.mean([x['tempos_segundos']['total_frame'] for x in registros]))}

        if not EXECUTAR_EVENTO_ASTA:
            print('Evento externo bloqueado. Compartilhe primeiro ASTA_PRONTO_PARA_EVENTO_EXTERNO_UNICO.')
        else:
            existente=json.loads(LOCK_PATH.read_text(encoding='utf-8')) if LOCK_PATH.is_file() else None
            if existente and existente.get('status')=='concluido':
                raise RuntimeError('Evento ASTA já concluído. Nova inferência é proibida.')
            if existente:
                if not RETOMAR_EVENTO_INCOMPLETO:raise RuntimeError('Evento incompleto existe. Use RETOMAR_EVENTO_INCOMPLETO=True.')
                assert existente['sha256_contrato_protocolo_asta']==EXPECTED_PROTOCOL_SHA256
                assert existente['sha256_config_evento']==CONFIG_EVENTO['sha256_config_evento']
                EVENT_ID=existente['event_id'];evento=existente
            else:
                if RETOMAR_EVENTO_INCOMPLETO:raise RuntimeError('Não existe evento para retomar.')
                EVENT_ID=str(uuid.uuid4())
                evento={'protocolo':'asta_external_protected_event_v3','status':'em_andamento','event_id':EVENT_ID,
                    'iniciado_em_utc':datetime.now(timezone.utc).isoformat(),
                    'sha256_contrato_protocolo_asta':EXPECTED_PROTOCOL_SHA256,
                    'sha256_config_evento':CONFIG_EVENTO['sha256_config_evento'],
                    'frames_previstos':178,'yolo_frames_concluidos':0,'hough_frames_concluidos':0}
                LOCK_PATH.parent.mkdir(parents=True,exist_ok=True)
                with LOCK_PATH.open('x',encoding='utf-8') as arq:json.dump(evento,arq,indent=2,ensure_ascii=False)

            EVENT_DIR=INFERENCE_ROOT/f'evento_{EVENT_ID}'
            for pasta in [EVENT_DIR/'checkpoints/yolo',EVENT_DIR/'checkpoints/hough',EVENT_DIR/'consolidado']:
                pasta.mkdir(parents=True,exist_ok=True)
            config_remota=EVENT_DIR/'config_evento_asta_v3.json'
            if config_remota.exists():
                config_existente=json.loads(config_remota.read_text(encoding='utf-8'))
                assert config_existente['sha256_config_evento']==CONFIG_EVENTO['sha256_config_evento']
            else:gravar_json_atomico(config_remota,CONFIG_EVENTO)

            def atualizar_lock(**campos):
                evento.update(campos);evento['atualizado_em_utc']=datetime.now(timezone.utc).isoformat()
                gravar_json_atomico(LOCK_PATH,evento)

            pendentes_yolo=[]
            for linha in manifest.itertuples():
                if not (EVENT_DIR/'checkpoints/yolo'/f'{linha.id_asta}.json').is_file():pendentes_yolo.append(linha.id_asta)
            modelo=None
            if pendentes_yolo:
                print(f'[{agora()}] Copiando e verificando checkpoint operacional YOLO...')
                copiar_com_hash(peso_operacional,LOCAL_WEIGHT,EXPECTED_OPERATIONAL_WEIGHT_SHA256)
                from ultralytics import YOLO
                modelo=YOLO(str(LOCAL_WEIGHT))

            try:
                for linha in tqdm(manifest.itertuples(),total=len(manifest),desc='Frames ASTA',unit='frame'):
                    caminho_yolo=EVENT_DIR/'checkpoints/yolo'/f'{linha.id_asta}.json'
                    caminho_hough=EVENT_DIR/'checkpoints/hough'/f'{linha.id_asta}.json'
                    reg_yolo=checkpoint_valido(caminho_yolo,'yolo',linha.id_asta) if caminho_yolo.is_file() else None
                    reg_hough=checkpoint_valido(caminho_hough,'hough',linha.id_asta) if caminho_hough.is_file() else None
                    if reg_yolo is not None and reg_hough is not None:continue

                    img_local=STAGING/linha.arquivo_imagem;mask_local=STAGING/linha.arquivo_mascara
                    t_copia=time.perf_counter()
                    copiar_com_hash(ASTA_RAW/linha.arquivo_imagem,img_local,linha.sha256_imagem)
                    copiar_com_hash(ASTA_RAW/linha.arquivo_mascara,mask_local,linha.sha256_mascara)
                    tempo_copia=time.perf_counter()-t_copia
                    t_pre=time.perf_counter()
                    imagem=cv2.imread(str(img_local),cv2.IMREAD_GRAYSCALE)
                    if imagem is None:raise RuntimeError(f'Decode da imagem falhou: {linha.id_asta}')
                    if imagem.shape!=(10560,10560):raise RuntimeError(f'Dimensão da imagem divergente: {linha.id_asta}')
                    imagem_norm,meta_norm=normalizar_frame_uint8(imagem)
                    tempo_pre=time.perf_counter()-t_pre
                    mascara=cv2.imread(str(mask_local),cv2.IMREAD_GRAYSCALE)
                    if mascara is None:raise RuntimeError(f'Decode da máscara falhou: {linha.id_asta}')
                    if mascara.shape!=(10560,10560):raise RuntimeError(f'Dimensão da máscara divergente: {linha.id_asta}')
                    skeleton_gt=skeletonize(mascara>0)
                    preenchimento=int(meta_norm['mediana_normalizada'])

                    if reg_yolo is None:
                        brutos,t_inf,t_tiles=detectar_yolo_tiles(modelo,imagem_norm,preenchimento,linha.id_asta)
                        t_merge=time.perf_counter();unidos=unir_segmentos(brutos);t_merge=time.perf_counter()-t_merge
                        t_metric=time.perf_counter();metricas=metricas_centerline(unidos,skeleton_gt,14);t_metric=time.perf_counter()-t_metric
                        tempos={'copia_drive':tempo_copia,'preprocessamento':tempo_pre+t_tiles,'inferencia':t_inf,
                            'merging':t_merge,'avaliacao':t_metric,'total_frame':tempo_pre+t_tiles+t_inf+t_merge}
                        reg_yolo=criar_registro_frame('yolo',linha,brutos,unidos,metricas,tempos,meta_norm)
                        gravar_json_atomico(caminho_yolo,reg_yolo)
                        del brutos,unidos

                    if reg_hough is None:
                        brutos,t_inf,t_tiles=detectar_hough_tiles(imagem_norm,preenchimento,linha.id_asta)
                        t_merge=time.perf_counter();unidos=unir_segmentos(brutos);t_merge=time.perf_counter()-t_merge
                        t_metric=time.perf_counter();metricas=metricas_centerline(unidos,skeleton_gt,14);t_metric=time.perf_counter()-t_metric
                        tempos={'copia_drive':tempo_copia,'preprocessamento':tempo_pre+t_tiles,'inferencia':t_inf,
                            'merging':t_merge,'avaliacao':t_metric,'total_frame':tempo_pre+t_tiles+t_inf+t_merge}
                        reg_hough=criar_registro_frame('hough',linha,brutos,unidos,metricas,tempos,meta_norm)
                        gravar_json_atomico(caminho_hough,reg_hough)
                        del brutos,unidos

                    atualizar_lock(yolo_frames_concluidos=len(list((EVENT_DIR/'checkpoints/yolo').glob('*.json'))),
                        hough_frames_concluidos=len(list((EVENT_DIR/'checkpoints/hough').glob('*.json'))),
                        ultimo_id_concluido=linha.id_asta)
                    for caminho in [img_local,mask_local]:
                        if caminho.exists():caminho.unlink()
                    del imagem,mascara,imagem_norm,skeleton_gt;gc.collect();torch.cuda.empty_cache()

                registros_yolo=carregar_registros(EVENT_DIR,'yolo');registros_hough=carregar_registros(EVENT_DIR,'hough')
                assert len(registros_yolo)==len(registros_hough)==178
                assert {r['id_asta'] for r in registros_yolo}==set(manifest.id_asta.astype(str))
                assert {r['id_asta'] for r in registros_hough}==set(manifest.id_asta.astype(str))

                linhas=[]
                for metodo,registros in [('yolo',registros_yolo),('hough',registros_hough)]:
                    for r in registros:
                        m=r['metricas'];t=r['tempos_segundos']
                        linhas.append({'id_asta':r['id_asta'],'metodo':metodo,'gt_positivo':r['gt_positivo'],
                            'precisao_centerline':m['precisao_centerline'],'recall_centerline':m['recall_centerline'],
                            'f1_centerline':m['f1_centerline'],'pred_pixels':m['pred_pixels'],'gt_pixels':m['gt_pixels'],
                            'pred_hits':m['pred_hits'],'gt_hits':m['gt_hits'],'segmentos_brutos':r['segmentos_brutos'],
                            'segmentos_apos_merging':r['segmentos_apos_merging'],'tempo_preprocessamento':t['preprocessamento'],
                            'tempo_inferencia':t['inferencia'],'tempo_merging':t['merging'],'tempo_total_frame':t['total_frame']})
                tabela=pd.DataFrame(linhas);tabela_path=EVENT_DIR/'consolidado/metricas_por_frame_metodo_asta_v3.csv'
                gravar_csv_atomico(tabela_path,tabela)
                resumo_yolo=consolidar_metodo(registros_yolo,'YOLO11n-OBB seed 20260824')
                resumo_hough=consolidar_metodo(registros_hough,'Hough congelado')

                positivos=tabela[tabela.gt_positivo].pivot(index='id_asta',columns='metodo',values='f1_centerline').dropna()
                assert len(positivos)==177 and {'yolo','hough'}<=set(positivos.columns)
                rng=np.random.default_rng(CONFIG_EVENTO['bootstrap_seed']);boot=[];n=len(positivos)
                valores_y=positivos.yolo.to_numpy(float);valores_h=positivos.hough.to_numpy(float)
                for _ in tqdm(range(CONFIG_EVENTO['bootstrap_replicacoes']),desc='Bootstrap ASTA por frame',unit='rep'):
                    idx=rng.integers(0,n,n);my=float(valores_y[idx].mean());mh=float(valores_h[idx].mean())
                    boot.append({'f1_yolo':my,'f1_hough':mh,'delta_f1_yolo_menos_hough':my-mh})
                boot_df=pd.DataFrame(boot);boot_path=EVENT_DIR/'consolidado/bootstrap_frames_asta_v3.csv';gravar_csv_atomico(boot_path,boot_df)
                intervalos={c:[float(x) for x in np.percentile(boot_df[c],[2.5,97.5])] for c in boot_df.columns}

                pares=tabela.pivot(index='id_asta',columns='metodo',values='f1_centerline').reset_index()
                pares['delta_yolo_menos_hough']=pares.yolo-pares.hough
                candidatos=pd.concat([pares.nsmallest(12,'yolo'),pares.nsmallest(12,'hough'),
                    pares.nsmallest(8,'delta_yolo_menos_hough'),pares.nlargest(8,'delta_yolo_menos_hough')])
                candidatos=candidatos.drop_duplicates('id_asta').sort_values('id_asta')
                candidatos_path=EVENT_DIR/'consolidado/casos_revisao_qualitativa_asta_v3.csv';gravar_csv_atomico(candidatos_path,candidatos)

                resumo={'protocolo':'asta_external_protected_event_v3','status':'EVENTO_EXTERNO_ASTA_CONCLUIDO',
                    'event_id':EVENT_ID,'sha256_contrato_protocolo_asta':EXPECTED_PROTOCOL_SHA256,
                    'sha256_config_evento':CONFIG_EVENTO['sha256_config_evento'],'frames':178,'positivos':177,'negativos':1,
                    'resultado_primario_yolo':resumo_yolo,'hough_congelado':resumo_hough,
                    'bootstrap_pareado_frames_positivos':{'replicacoes':2000,'seed':20260823,'n_frames':177,'ic95_percentil':intervalos},
                    'aviso_negativos':'ASTA contém somente um frame oficialmente negativo; métricas de falso positivo não têm IC estável.',
                    'selecionou_modelo_seed_ou_threshold_com_asta':False,'recalibrou_hough_com_asta':False}
                resumo['sha256_resultado_final']=sha256_canonico(resumo)
                resumo_path=EVENT_DIR/'consolidado/resumo_final_evento_asta_v3.json';gravar_json_atomico(resumo_path,resumo)
                artefatos={p.name:sha256_arquivo(p) for p in [tabela_path,boot_path,candidatos_path,resumo_path]}
                hashes_path=EVENT_DIR/'consolidado/hashes_resultados_asta_v3.json';gravar_json_atomico(hashes_path,artefatos)
                atualizar_lock(status='concluido',concluido_em_utc=datetime.now(timezone.utc).isoformat(),
                    yolo_frames_concluidos=178,hough_frames_concluidos=178,
                    sha256_resultado_final=resumo['sha256_resultado_final'],
                    resumo_relativo=str(resumo_path.relative_to(BACKUP_DIR)).replace('\\','/'))
                print(json.dumps(resumo,indent=2,ensure_ascii=False))
                print('EVENTO EXTERNO ASTA CONCLUÍDO E TRAVADO. Não reexecute nem recalibre.')
            except Exception as exc:
                atualizar_lock(status='incompleto',ultimo_erro=repr(exc))
                raise
        ''') ,
        md(r'''
        ## 5. Verificação pós-evento

        Execute após a conclusão. Esta célula não carrega modelos nem realiza inferência; apenas
        valida a trava, os 356 checkpoints e os artefatos consolidados no Drive.
        ''') ,
        code(r'''
        if not LOCK_PATH.is_file():
            print('Evento externo ainda não foi iniciado.')
        else:
            trava=json.loads(LOCK_PATH.read_text(encoding='utf-8'))
            if trava.get('status')!='concluido':
                print(json.dumps({'status':'EVENTO_ASTA_INCOMPLETO','event_id':trava.get('event_id'),
                    'yolo_frames_concluidos':trava.get('yolo_frames_concluidos',0),
                    'hough_frames_concluidos':trava.get('hough_frames_concluidos',0)},indent=2,ensure_ascii=False))
            else:
                event_dir=INFERENCE_ROOT/f"evento_{trava['event_id']}";consolidado=event_dir/'consolidado'
                resumo=json.loads((consolidado/'resumo_final_evento_asta_v3.json').read_text(encoding='utf-8'))
                assert verificar_hash_canonico(resumo,'sha256_resultado_final')==trava['sha256_resultado_final']
                hashes=json.loads((consolidado/'hashes_resultados_asta_v3.json').read_text(encoding='utf-8'))
                for nome,esperado in tqdm(sorted(hashes.items()),desc='Verificando resultados ASTA',unit='arq'):
                    assert sha256_arquivo(consolidado/nome)==esperado
                for metodo in ['yolo','hough']:
                    arquivos=sorted((event_dir/'checkpoints'/metodo).glob('*.json'));assert len(arquivos)==178
                    ids=[]
                    for caminho in tqdm(arquivos,desc=f'Verificando checkpoints {metodo}',unit='frame'):
                        r=json.loads(caminho.read_text(encoding='utf-8'))
                        assert verificar_hash_canonico(r,'sha256_checkpoint')==r['sha256_checkpoint']
                        assert r['event_id']==trava['event_id'] and r['metodo']==metodo
                        assert r['sha256_config_evento']==trava['sha256_config_evento'];ids.append(r['id_asta'])
                    assert len(set(ids))==178 and set(ids)==set(manifest.id_asta.astype(str))
                print(json.dumps({'status':'BACKUP_EVENTO_EXTERNO_ASTA_V3_APROVADO',
                    'event_id':trava['event_id'],'sha256_contrato_protocolo_asta':EXPECTED_PROTOCOL_SHA256,
                    'sha256_config_evento':trava['sha256_config_evento'],
                    'sha256_resultado_final':trava['sha256_resultado_final'],
                    'yolo_frames':178,'hough_frames':178,'reexecucao_proibida':True},indent=2,ensure_ascii=False))
        ''') ,
    ]
    return write_notebook('13_evento_externo_asta_yolo_hough_protegido_v3.ipynb', cells)


def build_14():
    cells = [
        md(r'''
        # 14 — Análise pós-teste, erros e figuras finais (v4)

        Este notebook é **somente leitura em relação aos experimentos**. Ele não carrega pesos,
        não executa YOLO ou Hough e não altera threshold, tiles, merging ou métricas. Seu objetivo é:

        1. verificar os resultados protegidos sintético, YOLO ASTA v3 e Hough ASTA corrigido v4.1;
        2. consolidar tabelas e figuras quantitativas;
        3. descrever erros sintéticos por subtipo/SNR sem retreino;
        4. materializar casos ASTA por regras determinísticas reaplicadas aos resultados corrigidos;
        5. produzir folhas para revisão qualitativa humana antes da publicação final da análise.

        Execute em **CPU**. Os resultados de teste são finais e não podem motivar retuning.
        ''') ,
        code(r'''
        %pip install -q pandas numpy matplotlib pillow tqdm opencv-python-headless

        from google.colab import drive
        from pathlib import Path
        from datetime import datetime, timezone
        from tqdm.auto import tqdm
        from PIL import Image, ImageDraw, ImageFont
        import gc, hashlib, json, math, os, shutil, time, zipfile
        import cv2
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Image.MAX_IMAGE_PIXELS=200_000_000
        drive.mount('/content/drive',force_remount=True)

        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        RUN_ID='synthetic_v3_c41d5091de56'
        SYNTH_EVENT_ID='29fcf556-5eb4-43fc-8d30-8e3d88edf6c9'
        YOLO_ASTA_EVENT_ID='4edef981-5590-4330-8dba-bd4d14dbcdda'
        HOUGH_ASTA_EVENT_ID='7d717aea-eed8-4f4b-ba63-d4b0ca51d29c'
        EXPECTED_SYNTH_RESULT='878f63dbedea0f72a4b7ed77c238d49cb2b17a89cfce4f51a27ea3b4d7fa0485'
        EXPECTED_ASTA_PROTOCOL='dc4c58381b1e3e7d2a24f7fdac8fc59276e53b9d900853c1191a497319e862bc'
        EXPECTED_ASTA_CONFIG='81ab0148b765fd3664b185812560826b21bf680081aa6c17d37ee2b94f773137'
        EXPECTED_YOLO_ASTA_RESULT='eef0f468754b195984a4888cc8c8439f12b5e9db3af94e875082272503542f7c'
        EXPECTED_HOUGH_ASTA_RESULT='303502ee091dfa26649df5bef4f61f0cf8d644925fd9487f96b2085772b444d8'
        EXPECTED_HOUGH_CORRECTION_CONFIG='ea65b9d12715fdfb486609605a5ae58deef43535e9f331652feadddeeed71cd7'
        EXPECTED_AUDIT_MANIFEST='91bfa07ebffceaa46271a254d3862348ce2b242a06ed44fc4c07efeeb58722ca'
        EXPECTED_SYNTH_MANIFEST='b635aff10e8b24d3729b4b399f50a8223e0d4373d155a7f16bf232c58d844b42'

        ASTA_RAW=BACKUP_DIR/'data/raw/asta_real'
        AUDIT_MANIFEST=BACKUP_DIR/'experiments/asta_external_v3/audit/final/manifest_auditoria_asta_v3.csv'
        ASTA_ROOT=BACKUP_DIR/'experiments/asta_external_v3/inference'
        YOLO_ASTA_LOCK=ASTA_ROOT/'TRAVA_EVENTO_EXTERNO_ASTA_V3.json'
        YOLO_ASTA_EVENT=ASTA_ROOT/f'evento_{YOLO_ASTA_EVENT_ID}'
        YOLO_ASTA_CONS=YOLO_ASTA_EVENT/'consolidado'
        HOUGH_CORR_ROOT=BACKUP_DIR/'experiments/hough_asta_correction_v4'
        HOUGH_ASTA_LOCK=HOUGH_CORR_ROOT/'TRAVA_EVENTO_HOUGH_ASTA_V4.json'
        HOUGH_ASTA_EVENT=HOUGH_CORR_ROOT/'events'/f'evento_{HOUGH_ASTA_EVENT_ID}'
        HOUGH_ASTA_CONS=HOUGH_ASTA_EVENT/'consolidado'
        SYNTH_RUN=BACKUP_DIR/'data/synthetic_runs'/RUN_ID
        SYNTH_EVENT=BACKUP_DIR/'experiments/yolo_protected_test_v3'/RUN_ID/f'evento_{SYNTH_EVENT_ID}'
        SYNTH_CONS=SYNTH_EVENT/'consolidado'
        VAL_DIR=BACKUP_DIR/'experiments/yolo_final_multiseed_v3'/RUN_ID/'avaliacao_val'
        ANALYSIS_ROOT=BACKUP_DIR/'experiments/post_test_analysis_v4'/RUN_ID
        DRAFT_DRIVE=ANALYSIS_ROOT/'draft'
        FINAL_DRIVE=ANALYSIS_ROOT/'final'
        LOCAL=Path('/content/tcc_post_test_analysis_v4')
        OUTPUT_LOCAL=LOCAL/'outputs';STAGING=LOCAL/'staging'
        for pasta in [OUTPUT_LOCAL,STAGING,DRAFT_DRIVE]:pasta.mkdir(parents=True,exist_ok=True)

        def agora():return datetime.now().strftime('%H:%M:%S')
        def sha256_arquivo(caminho,chunk=8*1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as arq:
                for bloco in iter(lambda:arq.read(chunk),b''):h.update(bloco)
            return h.hexdigest()
        def sha256_canonico(objeto,ensure_ascii=False):
            texto=json.dumps(objeto,sort_keys=True,separators=(',',':'),ensure_ascii=ensure_ascii)
            return hashlib.sha256(texto.encode('utf-8')).hexdigest()
        def verificar_canonico(registro,campo,ensure_ascii=False):
            copia=dict(registro);observado=copia.pop(campo)
            calculado=sha256_canonico(copia,ensure_ascii=ensure_ascii)
            if calculado!=observado:raise RuntimeError(f'Hash canônico divergente: {campo}')
            return observado
        def copiar_verificado(origem,destino,esperado):
            destino=Path(destino);destino.parent.mkdir(parents=True,exist_ok=True)
            if destino.is_file() and sha256_arquivo(destino)==esperado:return
            shutil.copy2(origem,destino)
            if sha256_arquivo(destino)!=esperado:raise RuntimeError(f'Hash divergente após cópia: {origem.name}')
        def gravar_json(caminho,objeto):
            Path(caminho).write_text(json.dumps(objeto,indent=2,ensure_ascii=False),encoding='utf-8')

        print('Notebook 14 preparado em CPU; nenhuma biblioteca de inferência foi carregada.')
        ''') ,
        md(r'''
        ## 1. Gate dos insumos finais protegidos

        Confirma hashes, eventos, manifests e artefatos consolidados antes de qualquer análise.
        Este gate não lê imagens ASTA e não executa detectores.
        ''') ,
        code(r'''
        assert YOLO_ASTA_LOCK.is_file() and HOUGH_ASTA_LOCK.is_file() and AUDIT_MANIFEST.is_file()
        trava_yolo=json.loads(YOLO_ASTA_LOCK.read_text(encoding='utf-8'))
        assert trava_yolo['status']=='concluido' and trava_yolo['event_id']==YOLO_ASTA_EVENT_ID
        assert trava_yolo['sha256_contrato_protocolo_asta']==EXPECTED_ASTA_PROTOCOL
        assert trava_yolo['sha256_config_evento']==EXPECTED_ASTA_CONFIG
        assert trava_yolo['sha256_resultado_final']==EXPECTED_YOLO_ASTA_RESULT
        trava_hough=json.loads(HOUGH_ASTA_LOCK.read_text(encoding='utf-8'))
        assert trava_hough['status']=='concluido' and trava_hough['event_id']==HOUGH_ASTA_EVENT_ID
        assert trava_hough['sha256_config_correcao']==EXPECTED_HOUGH_CORRECTION_CONFIG
        assert trava_hough['sha256_resultado_final']==EXPECTED_HOUGH_ASTA_RESULT

        hashes_asta=json.loads((YOLO_ASTA_CONS/'hashes_resultados_asta_v3.json').read_text(encoding='utf-8'))
        assert set(hashes_asta)=={'metricas_por_frame_metodo_asta_v3.csv','bootstrap_frames_asta_v3.csv',
            'casos_revisao_qualitativa_asta_v3.csv','resumo_final_evento_asta_v3.json'}
        for nome,esperado in tqdm(sorted(hashes_asta.items()),desc='Verificando consolidação YOLO ASTA v3',unit='arq'):
            assert sha256_arquivo(YOLO_ASTA_CONS/nome)==esperado
        resumo_asta_yolo=json.loads((YOLO_ASTA_CONS/'resumo_final_evento_asta_v3.json').read_text(encoding='utf-8'))
        assert verificar_canonico(resumo_asta_yolo,'sha256_resultado_final',ensure_ascii=False)==EXPECTED_YOLO_ASTA_RESULT
        resumo_asta_hough=json.loads((HOUGH_ASTA_CONS/'resumo_final_hough_asta_v4.json').read_text(encoding='utf-8'))
        assert verificar_canonico(resumo_asta_hough,'sha256_resultado_final',ensure_ascii=False)==EXPECTED_HOUGH_ASTA_RESULT
        assert resumo_asta_hough['event_id']==HOUGH_ASTA_EVENT_ID
        assert resumo_asta_hough['sha256_config_correcao']==EXPECTED_HOUGH_CORRECTION_CONFIG
        assert resumo_asta_hough['evento_v3_preservado']==YOLO_ASTA_EVENT_ID
        assert not resumo_asta_hough['yolo_executado'] and not resumo_asta_hough['busca_parametros'] and not resumo_asta_hough['recalibracao_asta']

        resumo_sint_path=SYNTH_CONS/'resumo_final_yolo_test_v3.json'
        assert resumo_sint_path.is_file()
        resumo_sint=json.loads(resumo_sint_path.read_text(encoding='utf-8'))
        assert resumo_sint['event_id']==SYNTH_EVENT_ID
        assert verificar_canonico(resumo_sint,'sha256_resultado_final',ensure_ascii=True)==EXPECTED_SYNTH_RESULT
        assert sha256_arquivo(SYNTH_RUN/'manifest_geracao_v3.csv')==EXPECTED_SYNTH_MANIFEST
        assert sha256_arquivo(AUDIT_MANIFEST)==EXPECTED_AUDIT_MANIFEST

        manifest_asta=pd.read_csv(AUDIT_MANIFEST)
        assert len(manifest_asta)==178 and manifest_asta.id_asta.nunique()==178
        metricas_v3=pd.read_csv(YOLO_ASTA_CONS/'metricas_por_frame_metodo_asta_v3.csv')
        metricas_yolo=metricas_v3[metricas_v3.metodo=='yolo'].copy();assert len(metricas_yolo)==178
        if 'cobertura_gt' not in metricas_yolo.columns:
            metricas_yolo['cobertura_gt']=metricas_yolo['recall_centerline']
        metricas_hough=pd.read_csv(HOUGH_ASTA_CONS/'metricas_hough_asta_v4.csv');assert len(metricas_hough)==178
        metricas_hough['metodo']='hough'
        if 'cobertura_gt' not in metricas_hough.columns:
            metricas_hough['cobertura_gt']=metricas_hough['recall_centerline']
        mapa_gt=(manifest_asta.set_index('id_asta').mascara_pixels_positivos.astype(int)>0).to_dict()
        metricas_hough['gt_positivo']=metricas_hough.id_asta.map(mapa_gt)
        assert metricas_hough.gt_positivo.notna().all()
        colunas_metricas=['id_asta','metodo','gt_positivo','pred_pixels','gt_pixels','pred_hits','gt_hits',
            'precisao_centerline','recall_centerline','f1_centerline','cobertura_gt']
        metricas_asta=pd.concat([metricas_yolo[colunas_metricas],metricas_hough[colunas_metricas]],ignore_index=True)
        assert len(metricas_asta)==356 and set(metricas_asta.metodo)=={'yolo','hough'}
        assert len(list((YOLO_ASTA_EVENT/'checkpoints/yolo').glob('*.json')))==178
        assert len(list((HOUGH_ASTA_EVENT/'checkpoints').glob('*.json')))==178
        for caminho in tqdm(sorted((HOUGH_ASTA_EVENT/'checkpoints').glob('*.json')),desc='Verificando checkpoints Hough v4.1',unit='frame'):
            registro=json.loads(caminho.read_text(encoding='utf-8'))
            assert verificar_canonico(registro,'sha256_checkpoint',ensure_ascii=False)==registro['sha256_checkpoint']
            assert registro['event_id']==HOUGH_ASTA_EVENT_ID and registro['sha256_config_correcao']==EXPECTED_HOUGH_CORRECTION_CONFIG

        pares_gate=metricas_asta[metricas_asta.gt_positivo.astype(bool)].pivot(index='id_asta',columns='metodo',values='f1_centerline').dropna()
        pares_gate['delta']=pares_gate.yolo-pares_gate.hough;categorias_gate={}
        def adicionar_gate(ids,rotulo):
            for id_asta in ids:categorias_gate.setdefault(str(id_asta),[]).append(rotulo)
        adicionar_gate(pares_gate.nsmallest(12,'yolo').index,'pior_yolo')
        adicionar_gate(pares_gate.nsmallest(12,'hough').index,'pior_hough_corrigido')
        adicionar_gate(pares_gate.nsmallest(8,'delta').index,'menor_delta_yolo_hough_corrigido')
        adicionar_gate(pares_gate.nlargest(8,'delta').index,'maior_delta_yolo_hough_corrigido')
        candidatos_evento=pd.DataFrame([{'id_asta':k,'categorias':'|'.join(v)} for k,v in sorted(categorias_gate.items())])

        relatorio_insumos={'status':'INSUMOS_ANALISE_POS_TESTE_V4_APROVADOS',
            'yolo_asta_event_id':YOLO_ASTA_EVENT_ID,'sha256_resultado_yolo_asta':EXPECTED_YOLO_ASTA_RESULT,
            'hough_asta_event_id':HOUGH_ASTA_EVENT_ID,'sha256_resultado_hough_asta':EXPECTED_HOUGH_ASTA_RESULT,
            'sintetico_event_id':SYNTH_EVENT_ID,
            'sha256_resultado_sintetico':EXPECTED_SYNTH_RESULT,'frames_asta':178,
            'checkpoints_asta':{'yolo_v3_reutilizado':178,'hough_v4_1_corrigido':178},'casos_evento':len(candidatos_evento),
            'inferencia_executada':False,'retuning_permitido':False}
        print(json.dumps(relatorio_insumos,indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 2. Consolidação quantitativa e análise descritiva de erros

        Produz tabelas e figuras a partir de resultados já persistidos. A comparação entre
        sintético e ASTA mantém os endpoints separados: detecção/OBB no sintético e centerline no
        ASTA. As análises por subtipo e SNR são descritivas e não autorizam novo treino.
        ''') ,
        code(r'''
        plt.style.use('seaborn-v0_8-whitegrid')
        CORES={'yolo':'#1f77b4','hough':'#d62728'}

        positivos_asta=metricas_asta[metricas_asta.gt_positivo.astype(bool)].copy()
        pares_asta=positivos_asta.pivot(index='id_asta',columns='metodo',values='f1_centerline').dropna()
        assert len(pares_asta)==177
        distribuicao=[]
        for metodo in ['yolo','hough']:
            valores=pares_asta[metodo].to_numpy(float)
            distribuicao.append({'metodo':metodo,'n':len(valores),'media':float(np.mean(valores)),
                'desvio_padrao':float(np.std(valores,ddof=1)),'q05':float(np.quantile(valores,.05)),
                'q25':float(np.quantile(valores,.25)),'mediana':float(np.median(valores)),
                'q75':float(np.quantile(valores,.75)),'q95':float(np.quantile(valores,.95)),
                'min':float(np.min(valores)),'max':float(np.max(valores))})
        tabela_distribuicao=pd.DataFrame(distribuicao)
        tabela_distribuicao.to_csv(OUTPUT_LOCAL/'distribuicao_f1_frames_asta_v4.csv',index=False)

        linhas_dominios=[]
        ys=resumo_sint['resultado_primario_yolo_media_seeds'];hs=resumo_sint['hough_congelado']
        ya=resumo_asta_yolo['resultado_primario_yolo'];ha=resumo_asta_hough['resultado_hough_corrigido']
        linhas_dominios.extend([
            {'dominio':'sintetico','endpoint':'matching OBB/geometria','metodo':'YOLO11n-OBB (média 3 seeds)',
             'precisao':ys['precisao'],'recall':ys['recall'],'f1':ys['f1'],'comparavel_diretamente_entre_dominios':False},
            {'dominio':'sintetico','endpoint':'matching OBB/geometria','metodo':'Hough congelado',
             'precisao':hs['precisao'],'recall':hs['recall'],'f1':hs['f1'],'comparavel_diretamente_entre_dominios':False},
            {'dominio':'ASTA real','endpoint':'centerline pooled','metodo':'YOLO11n-OBB seed 20260824',
             'precisao':ya['precisao_centerline_pooled'],'recall':ya['recall_centerline_pooled'],
             'f1':ya['f1_centerline_pooled'],'comparavel_diretamente_entre_dominios':False},
            {'dominio':'ASTA real','endpoint':'centerline pooled','metodo':'Hough corrigido v4.1',
             'precisao':ha['precisao_centerline_pooled'],'recall':ha['recall_centerline_pooled'],
             'f1':ha['f1_centerline_pooled'],'comparavel_diretamente_entre_dominios':False}])
        tabela_dominios=pd.DataFrame(linhas_dominios)
        tabela_dominios.to_csv(OUTPUT_LOCAL/'comparacao_resultados_finais_por_dominio_v4.csv',index=False)

        fig,ax=plt.subplots(figsize=(8,5))
        for metodo in ['yolo','hough']:
            valores=np.sort(pares_asta[metodo].to_numpy(float));ecdf=np.arange(1,len(valores)+1)/len(valores)
            ax.plot(valores,ecdf,label=metodo.upper(),color=CORES[metodo],linewidth=2)
        ax.set(xlabel='F1 centerline por frame positivo',ylabel='Fração acumulada de frames',xlim=(0,1),ylim=(0,1))
        ax.legend();fig.tight_layout();fig.savefig(OUTPUT_LOCAL/'ecdf_f1_centerline_asta_v4.png',dpi=180);plt.close(fig)

        fig,ax=plt.subplots(figsize=(6,6));ax.scatter(pares_asta.hough,pares_asta.yolo,s=24,alpha=.65,color='#4c78a8')
        ax.plot([0,1],[0,1],'--',color='black',linewidth=1);ax.set(xlim=(-.02,1),ylim=(-.02,1),
            xlabel='F1 centerline Hough',ylabel='F1 centerline YOLO')
        fig.tight_layout();fig.savefig(OUTPUT_LOCAL/'dispersao_f1_yolo_hough_asta_v4.png',dpi=180);plt.close(fig)

        tempos_hough=[]
        for caminho in sorted((HOUGH_ASTA_EVENT/'checkpoints').glob('*.json')):
            registro=json.loads(caminho.read_text(encoding='utf-8'));tempos_hough.append(registro['tempos_segundos'])
        assert len(tempos_hough)==178
        tempo_hough_pre=float(np.mean([x['preprocessamento'] for x in tempos_hough]))
        tempo_hough_inf=float(np.mean([x['inferencia_hough_nms_costura'] for x in tempos_hough]))
        tempo_hough_total=float(np.mean([x['total_frame'] for x in tempos_hough]))
        tempos=pd.DataFrame([
            {'metodo':'YOLO','preprocessamento':ya['tempo_preprocessamento_medio_s'],'inferencia':ya['tempo_inferencia_medio_s'],
             'merging':ya['tempo_merging_medio_s'],'total':ya['tempo_total_frame_medio_s']},
            {'metodo':'Hough corrigido v4.1','preprocessamento':tempo_hough_pre,'inferencia':tempo_hough_inf,
             'merging':0.0,'total':tempo_hough_total}])
        tempos.to_csv(OUTPUT_LOCAL/'tempos_medios_asta_v4.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(tempos.metodo,tempos.total,color=[CORES['yolo'],CORES['hough']])
        ax.set(ylabel='Tempo médio por frame (s)',title='Custo operacional no ASTA')
        for i,v in enumerate(tempos.total):ax.text(i,v,f'{v:.1f}s',ha='center',va='bottom')
        fig.tight_layout();fig.savefig(OUTPUT_LOCAL/'tempo_total_por_metodo_asta_v4.png',dpi=180);plt.close(fig)

        manifest_sint=pd.read_csv(SYNTH_RUN/'manifest_geracao_v3.csv')
        manifest_test=manifest_sint[manifest_sint.split=='test'].copy();assert len(manifest_test)==900
        resultados_test=[]
        for seed in [20260823,20260824,20260825]:
            pasta=SYNTH_EVENT/f'seed_{seed}';registro=json.loads((pasta/'conclusao_teste_seed.json').read_text(encoding='utf-8'))
            assert sha256_arquivo(pasta/'resultado_test.csv')==registro['sha256_resultado']
            tab=pd.read_csv(pasta/'resultado_test.csv');assert len(tab)==900;tab['seed']=seed;resultados_test.append(tab)
        teste_long=pd.concat(resultados_test,ignore_index=True)
        por_amostra=teste_long.groupby('id',as_index=False).agg(tipo=('tipo','first'),subtipo=('subtipo','first'),
            fundo_origem=('fundo_origem','first'),sha256_fundo=('sha256_fundo','first'),
            taxa_tp_seeds=('tp','mean'),fp_medio_seeds=('fp','mean'),fn_medio_seeds=('fn','mean'))
        colunas_meta=[c for c in ['id','snr_pico_alvo','comprimento_px','fonte_fundo','categoria_fundo'] if c in manifest_test.columns]
        por_amostra=por_amostra.merge(manifest_test[colunas_meta],on='id',how='left',validate='one_to_one')
        por_amostra.to_csv(OUTPUT_LOCAL/'erros_sinteticos_por_amostra_multiseed_v3.csv',index=False)

        pos=por_amostra[por_amostra.tipo=='positivo'].copy()
        subtipo=pos.groupby('subtipo',as_index=False).agg(n=('id','size'),taxa_acerto_media=('taxa_tp_seeds','mean'),
            falha_todas_seeds=('taxa_tp_seeds',lambda s:int((s==0).sum())))
        subtipo.to_csv(OUTPUT_LOCAL/'acerto_sintetico_por_subtipo_v3.csv',index=False)
        pos['faixa_snr']=pd.cut(pos.snr_pico_alvo,bins=[3,6,10,15,20,25],include_lowest=True,duplicates='drop')
        por_snr=pos.groupby('faixa_snr',observed=True).agg(n=('id','size'),taxa_acerto_media=('taxa_tp_seeds','mean'),
            falha_todas_seeds=('taxa_tp_seeds',lambda s:int((s==0).sum()))).reset_index()
        por_snr['faixa_snr']=por_snr.faixa_snr.astype(str);por_snr.to_csv(OUTPUT_LOCAL/'acerto_sintetico_por_snr_v3.csv',index=False)

        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].bar(subtipo.subtipo,subtipo.taxa_acerto_media,color='#4c78a8');axes[0].set(ylim=(0,1),ylabel='Taxa média de acerto (3 seeds)',title='Por perfil sintético')
        axes[1].bar(por_snr.faixa_snr,por_snr.taxa_acerto_media,color='#72b7b2');axes[1].set(ylim=(0,1),title='Por faixa de SNR')
        axes[1].tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(OUTPUT_LOCAL/'acerto_sintetico_subtipo_snr_v3.png',dpi=180);plt.close(fig)

        # Validação: descrição retrospectiva, sem alterar checkpoint ou threshold já congelados.
        resultados_val=[]
        for seed in [20260823,20260824,20260825]:
            caminho=VAL_DIR/f'resultado_val_seed_{seed}.csv'
            assert caminho.is_file();tab=pd.read_csv(caminho);assert len(tab)==900;tab['seed']=seed;resultados_val.append(tab)
        val_long=pd.concat(resultados_val,ignore_index=True)
        val_resumo=val_long.groupby(['seed','tipo','subtipo'],as_index=False).agg(n=('id','size'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'))
        val_resumo['recall']=val_resumo.tp/(val_resumo.tp+val_resumo.fn).replace(0,np.nan)
        val_resumo.to_csv(OUTPUT_LOCAL/'erros_validacao_descritivos_pos_congelamento_v3.csv',index=False)

        resumo_quant={'status':'CONSOLIDACAO_QUANTITATIVA_POS_TESTE_V4_CONCLUIDA',
            'event_id_yolo_asta':YOLO_ASTA_EVENT_ID,'sha256_resultado_yolo_asta':EXPECTED_YOLO_ASTA_RESULT,
            'event_id_hough_asta':HOUGH_ASTA_EVENT_ID,'sha256_resultado_hough_asta':EXPECTED_HOUGH_ASTA_RESULT,
            'event_id_sintetico':SYNTH_EVENT_ID,
            'sha256_resultado_sintetico':EXPECTED_SYNTH_RESULT,'f1_macro_asta_yolo':ya['f1_centerline_macro_frames_positivos'],
            'f1_macro_asta_hough':ha['f1_centerline_macro_frames_positivos'],
            'ic95_delta_asta':resumo_asta_hough['bootstrap_pareado_frames_positivos']['ic95_percentil']['delta_yolo_menos_hough'],
            'comparacao_direta_sintetico_asta_permitida':False,'retuning_realizado':False,
            'arquivos_quantitativos':len(list(OUTPUT_LOCAL.glob('*')))}
        gravar_json(OUTPUT_LOCAL/'resumo_quantitativo_pos_teste_v4.json',resumo_quant)
        print(json.dumps(resumo_quant,indent=2,ensure_ascii=False));print(tabela_dominios)
        ''') ,
        md(r'''
        ## 3. Folhas qualitativas dos casos pré-especificados

        A lista é reconstruída pelas regras já congeladas: 12 menores F1 YOLO, 12 menores
        F1 Hough corrigido, oito menores e oito maiores deltas. O único negativo oficial é incluído por
        representar toda a estratificação negativa, não por cherry-picking. Nenhum detector roda.

        Cada caso é mostrado em quatro quadros separados, tanto no frame inteiro quanto em um recorte
        determinístico: imagem normalizada sem sobreposição, referência oficial em verde, YOLO somente
        em ciano e Hough corrigido somente em vermelho. O GT não é repetido sobre os quadros dos
        detectores. Os títulos distinguem explicitamente `REFERÊNCIA` de `PREDIÇÃO`.

        As predições são primeiro rasterizadas como centerlines de **um pixel na resolução nativa**
        de 10.560 × 10.560. Só depois o mapa de ocupação é reduzido por `INTER_AREA` e mostrado com
        compressão de densidade. Isso evita que milhares de segmentos Hough sejam artificialmente
        engrossados ao serem desenhados diretamente no preview de 640 × 640.

        Execute primeiro com `False`. Após conferir o gate quantitativo, altere apenas para `True`.
        ''') ,
        code(r'''
        GERAR_FOLHAS_QUALITATIVAS=False

        def normalizar_frame_uint8(imagem):
            hist=np.bincount(imagem.ravel(),minlength=256);ac=np.cumsum(hist);total=int(ac[-1])
            def percentil(q):return int(np.searchsorted(ac,q/100*max(total-1,1),side='right'))
            lo,hi=percentil(1.0),percentil(99.8)
            if hi<=lo:hi=min(255,lo+1)
            saida=np.clip((imagem.astype(np.float32)-lo)*255.0/(hi-lo),0,255).astype(np.uint8)
            return saida

        def ler_checkpoint(metodo,id_asta):
            caminho=(YOLO_ASTA_EVENT/'checkpoints/yolo'/f'{id_asta}.json') if metodo=='yolo' else (HOUGH_ASTA_EVENT/'checkpoints'/f'{id_asta}.json')
            registro=json.loads(caminho.read_text(encoding='utf-8'))
            assert verificar_canonico(registro,'sha256_checkpoint',ensure_ascii=False)==registro['sha256_checkpoint']
            evento_esperado=YOLO_ASTA_EVENT_ID if metodo=='yolo' else HOUGH_ASTA_EVENT_ID
            assert registro['event_id']==evento_esperado and str(registro['id_asta'])==str(id_asta)
            if metodo=='yolo':assert registro['metodo']=='yolo'
            else:assert registro['sha256_config_correcao']==EXPECTED_HOUGH_CORRECTION_CONFIG
            return registro

        RENDERIZACAO_PREVIEW='painel_4x2_full_crop_predicoes_isoladas_v4'
        GANHO_DENSIDADE=1.8

        def mapa_densidade_preview(segmentos,shape_nativa=(10560,10560),preview=640):
            h,w=shape_nativa;canvas=np.zeros((h,w),np.uint8)
            for s in segmentos:
                p1=(int(round(s['x1'])),int(round(s['y1'])));p2=(int(round(s['x2'])),int(round(s['y2'])))
                ok,q1,q2=cv2.clipLine((0,0,w,h),p1,p2)
                if ok:cv2.line(canvas,q1,q2,255,1,cv2.LINE_8)
            pixels_nativos=int(np.count_nonzero(canvas))
            densidade=cv2.resize(canvas,(preview,preview),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
            del canvas
            return densidade,pixels_nativos

        def sobrepor_densidade(base_rgb,densidade,cor,ganho=GANHO_DENSIDADE,alpha_max=0.85):
            alpha=np.clip(np.sqrt(np.maximum(densidade,0.0))*ganho,0.0,alpha_max)[...,None]
            cor_arr=np.asarray(cor,dtype=np.float32).reshape(1,1,3)
            return np.clip(base_rgb.astype(np.float32)*(1.0-alpha)+cor_arr*alpha,0,255).astype(np.uint8)

        def quantidade_segmentos(registro,metodo):
            if metodo=='yolo':return int(registro['segmentos_apos_merging'])
            return int(registro['segmentos_pos_costura_proprietaria'])

        def calcular_recorte_deterministico(mascara,segmentos_yolo,tamanho=1000):
            h,w=mascara.shape
            linhas=np.flatnonzero(np.max(mascara,axis=1));colunas=np.flatnonzero(np.max(mascara,axis=0))
            if len(linhas) and len(colunas):
                cx=int(round((int(colunas[0])+int(colunas[-1]))/2));cy=int(round((int(linhas[0])+int(linhas[-1]))/2))
                regra='centro_bbox_referencia_oficial'
            elif segmentos_yolo:
                s=max(segmentos_yolo,key=lambda z:math.hypot(float(z['x2'])-float(z['x1']),float(z['y2'])-float(z['y1'])))
                cx=int(round((float(s['x1'])+float(s['x2']))/2));cy=int(round((float(s['y1'])+float(s['y2']))/2))
                regra='centro_maior_segmento_yolo_no_negativo'
            else:
                cx=w//2;cy=h//2;regra='centro_geometrico_sem_predicao_yolo'
            lado=min(int(tamanho),h,w);x0=max(0,min(w-lado,cx-lado//2));y0=max(0,min(h-lado,cy-lado//2))
            return (int(x0),int(y0),int(x0+lado),int(y0+lado)),regra

        def mapa_densidade_recorte(segmentos,recorte,preview=480):
            x0,y0,x1,y1=recorte;h=y1-y0;w=x1-x0;canvas=np.zeros((h,w),np.uint8);visiveis=0
            for s in segmentos:
                p1=(int(round(s['x1']))-x0,int(round(s['y1']))-y0);p2=(int(round(s['x2']))-x0,int(round(s['y2']))-y0)
                ok,q1,q2=cv2.clipLine((0,0,w,h),p1,p2)
                if ok:cv2.line(canvas,q1,q2,255,1,cv2.LINE_8);visiveis+=1
            pixels=int(np.count_nonzero(canvas))
            densidade=cv2.resize(canvas,(preview,preview),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
            return densidade,pixels,visiveis

        def sha256_segmentos(segmentos):
            texto=json.dumps(segmentos,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            return hashlib.sha256(texto.encode('utf-8')).hexdigest()

        # Regressão: a rasterização conserva a centerline de um pixel antes da redução.
        _dens_teste,_px_teste=mapa_densidade_preview(
            [{'x1':8,'y1':32,'x2':119,'y2':32}],shape_nativa=(128,128),preview=64)
        assert _px_teste==112 and _dens_teste.shape==(64,64) and 0<float(_dens_teste.max())<=1.0
        del _dens_teste
        _base_teste=np.full((8,8,3),100,np.uint8);_zero_teste=np.zeros((8,8),np.float32);_gt_teste=np.zeros((8,8),np.float32);_gt_teste[3:5,3:5]=1
        _painel_gt_teste=sobrepor_densidade(_base_teste,_gt_teste,(0,255,80))
        _painel_y_zero=sobrepor_densidade(_base_teste,_zero_teste,(0,220,255))
        _painel_h_zero=sobrepor_densidade(_base_teste,_zero_teste,(255,60,40))
        assert not np.array_equal(_painel_gt_teste,_base_teste)
        assert np.array_equal(_painel_y_zero,_base_teste) and np.array_equal(_painel_h_zero,_base_teste)
        _um=np.ones((1,1),np.float32);_preto=np.zeros((1,1,3),np.uint8)
        _ciano=sobrepor_densidade(_preto,_um,(0,220,255))[0,0];_verde=sobrepor_densidade(_preto,_um,(0,255,80))[0,0];_vermelho=sobrepor_densidade(_preto,_um,(255,60,40))[0,0]
        assert _ciano[2]>_ciano[0] and _verde[1]>_verde[2] and _vermelho[0]>_vermelho[1]
        del _base_teste,_zero_teste,_gt_teste,_painel_gt_teste,_painel_y_zero,_painel_h_zero,_um,_preto

        if not GERAR_FOLHAS_QUALITATIVAS:
            print('Folhas qualitativas bloqueadas. Confira CONSOLIDACAO_QUANTITATIVA_POS_TESTE_V4_CONCLUIDA e altere somente a chave para True.')
        else:
            pares=metricas_asta[metricas_asta.gt_positivo.astype(bool)].pivot(index='id_asta',columns='metodo',values='f1_centerline').reset_index()
            categorias={}
            def adicionar(ids,rotulo):
                for id_asta in ids:categorias.setdefault(str(id_asta),[]).append(rotulo)
            positivos=pares.dropna(subset=['yolo','hough']).copy();positivos['delta']=positivos.yolo-positivos.hough
            adicionar(positivos.nsmallest(12,'yolo').id_asta,'pior_yolo')
            adicionar(positivos.nsmallest(12,'hough').id_asta,'pior_hough_corrigido')
            adicionar(positivos.nsmallest(8,'delta').id_asta,'menor_delta_yolo_hough_corrigido')
            adicionar(positivos.nlargest(8,'delta').id_asta,'maior_delta_yolo_hough_corrigido')
            assert set(candidatos_evento.id_asta.astype(str))==set(categorias)
            negativos=manifest_asta[manifest_asta.mascara_pixels_positivos.astype(int)==0]
            assert len(negativos)==1;id_neg=str(negativos.iloc[0].id_asta);categorias[id_neg]=['unico_negativo_oficial']
            ids=sorted(categorias);PREVIEW=480
            paineis=OUTPUT_LOCAL/'paineis_qualitativos';folhas=OUTPUT_LOCAL/'folhas_qualitativas'
            paineis.mkdir(exist_ok=True);folhas.mkdir(exist_ok=True)
            try:
                fonte=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
                fonte_titulo=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
            except Exception:fonte=fonte_titulo=ImageFont.load_default()
            indice=[]
            mapa=manifest_asta.set_index('id_asta')
            for id_asta in tqdm(ids,desc='Gerando painéis qualitativos',unit='frame'):
                linha=mapa.loc[id_asta];img_local=STAGING/linha.arquivo_imagem;mask_local=STAGING/linha.arquivo_mascara
                copiar_verificado(ASTA_RAW/linha.arquivo_imagem,img_local,linha.sha256_imagem)
                copiar_verificado(ASTA_RAW/linha.arquivo_mascara,mask_local,linha.sha256_mascara)
                imagem=cv2.imread(str(img_local),cv2.IMREAD_GRAYSCALE);mascara=cv2.imread(str(mask_local),cv2.IMREAD_GRAYSCALE)
                assert imagem is not None and mascara is not None and imagem.shape==mascara.shape==(10560,10560)
                norm=normalizar_frame_uint8(imagem)
                pequena=cv2.resize(norm,(PREVIEW,PREVIEW),interpolation=cv2.INTER_AREA)
                mask_density=cv2.resize(mascara,(PREVIEW,PREVIEW),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
                base=cv2.cvtColor(pequena,cv2.COLOR_GRAY2RGB)
                gt=sobrepor_densidade(base,mask_density,(0,255,80),ganho=1.8,alpha_max=0.65)
                ry=ler_checkpoint('yolo',id_asta);rh=ler_checkpoint('hough',id_asta)
                dens_y,pixels_y_render=mapa_densidade_preview(ry['segmentos'],preview=PREVIEW)
                dens_h,pixels_h_render=mapa_densidade_preview(rh['segmentos'],preview=PREVIEW)
                assert pixels_y_render==int(ry['metricas']['pred_pixels'])
                assert pixels_h_render==int(rh['metricas']['pred_pixels'])
                # Predições são mostradas sobre a imagem limpa, nunca sobre o GT.
                painel_y=sobrepor_densidade(base,dens_y,(0,220,255))
                painel_h=sobrepor_densidade(base,dens_h,(255,60,40))
                recorte,regra_recorte=calcular_recorte_deterministico(mascara,ry['segmentos'],tamanho=1000)
                x0,y0,x1,y1=recorte
                base_crop=cv2.cvtColor(cv2.resize(norm[y0:y1,x0:x1],(PREVIEW,PREVIEW),interpolation=cv2.INTER_AREA),cv2.COLOR_GRAY2RGB)
                mask_crop=cv2.resize(mascara[y0:y1,x0:x1],(PREVIEW,PREVIEW),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
                gt_crop=sobrepor_densidade(base_crop,mask_crop,(0,255,80),ganho=1.8,alpha_max=0.65)
                dens_y_crop,pixels_y_crop,n_y_crop=mapa_densidade_recorte(ry['segmentos'],recorte,preview=PREVIEW)
                dens_h_crop,pixels_h_crop,n_h_crop=mapa_densidade_recorte(rh['segmentos'],recorte,preview=PREVIEW)
                painel_y_crop=sobrepor_densidade(base_crop,dens_y_crop,(0,220,255))
                painel_h_crop=sobrepor_densidade(base_crop,dens_h_crop,(255,60,40))
                CABECALHO=90;ALTURA_TITULO=42;ALTURA_PAINEL=CABECALHO+2*(ALTURA_TITULO+PREVIEW)
                canvas=Image.new('RGB',(PREVIEW*4,ALTURA_PAINEL),'white');draw=ImageDraw.Draw(canvas)
                fy=ry['metricas']['f1_centerline'];fh=rh['metricas']['f1_centerline']
                fy_txt='NA' if fy is None else f'{float(fy):.4f}';fh_txt='NA' if fh is None else f'{float(fh):.4f}'
                n_y=quantidade_segmentos(ry,'yolo');n_h=quantidade_segmentos(rh,'hough')
                texto=f"{id_asta} | {','.join(categorias[id_asta])}"
                texto_metricas=f"YOLO: F1={fy_txt}, n={n_y} | Hough corrigido v4.1: F1={fh_txt}, n={n_h} | recorte={recorte}"
                y_titulo_1=CABECALHO;y_imagem_1=CABECALHO+ALTURA_TITULO
                y_titulo_2=y_imagem_1+PREVIEW;y_imagem_2=y_titulo_2+ALTURA_TITULO
                draw.text((12,8),texto,fill='black',font=fonte)
                draw.text((12,38),texto_metricas,fill='black',font=fonte)
                titulos_full=['FRAME INTEIRO | imagem limpa','REFERÊNCIA OFICIAL (verde) | NÃO É DETECÇÃO',
                    'YOLO (ciano) | SOMENTE PREDIÇÃO, SEM MÁSCARA','HOUGH v4.1 (vermelho) | SOMENTE PREDIÇÃO, SEM MÁSCARA']
                titulos_crop=['RECORTE | imagem limpa','RECORTE | REFERÊNCIA OFICIAL, NÃO É DETECÇÃO',
                    f'RECORTE | YOLO, SÓ PREDIÇÃO (n visível={n_y_crop})',f'RECORTE | HOUGH v4.1, SÓ PREDIÇÃO (n visível={n_h_crop})']
                for j,titulo in enumerate(titulos_full):draw.text((j*PREVIEW+8,y_titulo_1+10),titulo,fill='black',font=fonte_titulo)
                for j,titulo in enumerate(titulos_crop):draw.text((j*PREVIEW+8,y_titulo_2+10),titulo,fill='black',font=fonte_titulo)
                for j,q in enumerate([base,gt,painel_y,painel_h]):canvas.paste(Image.fromarray(q),(j*PREVIEW,y_imagem_1))
                for j,q in enumerate([base_crop,gt_crop,painel_y_crop,painel_h_crop]):canvas.paste(Image.fromarray(q),(j*PREVIEW,y_imagem_2))
                if n_y==0:
                    assert np.array_equal(painel_y,base) and np.array_equal(painel_y_crop,base_crop)
                if n_h==0:
                    assert np.array_equal(painel_h,base) and np.array_equal(painel_h_crop,base_crop)
                painel_path=paineis/f'{id_asta}.jpg';canvas.save(painel_path,quality=90,optimize=True)
                indice.append({'id_asta':id_asta,'categorias':'|'.join(categorias[id_asta]),
                    'f1_yolo':fy,'f1_hough_corrigido':fh,'segmentos_yolo':n_y,
                    'segmentos_hough_corrigido':n_h,'pred_pixels_yolo':pixels_y_render,
                    'pred_pixels_hough':pixels_h_render,'sha256_checkpoint_yolo':ry['sha256_checkpoint'],
                    'sha256_checkpoint_hough':rh['sha256_checkpoint'],'sha256_segmentos_yolo':sha256_segmentos(ry['segmentos']),
                    'sha256_segmentos_hough':sha256_segmentos(rh['segmentos']),
                    'recorte_x0':x0,'recorte_y0':y0,'recorte_x1':x1,'recorte_y1':y1,'regra_recorte':regra_recorte,
                    'pixels_yolo_no_recorte':pixels_y_crop,'pixels_hough_no_recorte':pixels_h_crop,
                    'segmentos_yolo_visiveis_recorte':n_y_crop,'segmentos_hough_visiveis_recorte':n_h_crop,
                    'renderizacao_preview':RENDERIZACAO_PREVIEW,'painel':painel_path.name,
                    'observacao_humana':'','padrao_erro':'','aprovado_para_figura_tcc':''})
                for p in [img_local,mask_local]:
                    if p.exists():p.unlink()
                del imagem,mascara,norm,pequena,mask_density,base,gt,dens_y,dens_h,painel_y,painel_h,base_crop,mask_crop,gt_crop,dens_y_crop,dens_h_crop,painel_y_crop,painel_h_crop,ry,rh,canvas;gc.collect()

            indice_df=pd.DataFrame(indice)
            assert indice_df.sha256_checkpoint_yolo.nunique()==len(indice_df)
            assert indice_df.sha256_checkpoint_hough.nunique()==len(indice_df)
            assert indice_df.sha256_segmentos_hough.nunique()==len(indice_df), 'Predições Hough idênticas entre casos distintos.'
            indice_path=OUTPUT_LOCAL/'indice_revisao_qualitativa_asta_v4.csv';indice_df.to_csv(indice_path,index=False)
            por_folha=2;ALTURA_PAINEL=90+2*(42+PREVIEW)
            for inicio in range(0,len(indice_df),por_folha):
                lote=indice_df.iloc[inicio:inicio+por_folha];imgs=[Image.open(paineis/x.painel).convert('RGB') for x in lote.itertuples()]
                folha=Image.new('RGB',(PREVIEW*4,ALTURA_PAINEL*len(imgs)),'white')
                for i,img in enumerate(imgs):folha.paste(img,(0,i*ALTURA_PAINEL))
                folha.save(folhas/f'folha_{inicio//por_folha+1:02d}.jpg',quality=90,optimize=True)
                for img in imgs:img.close()

            zip_path=OUTPUT_LOCAL/'folhas_revisao_qualitativa_asta_v4.zip'
            with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(paineis.glob('*.jpg'))+sorted(folhas.glob('*.jpg')):z.write(p,p.relative_to(OUTPUT_LOCAL))
                z.write(indice_path,indice_path.name)
            arquivos=[p for p in OUTPUT_LOCAL.rglob('*') if p.is_file() and p.name not in {
                'hashes_artefatos_analise_v4.json','relatorio_rascunho_analise_v4.json'}]
            hashes={str(p.relative_to(OUTPUT_LOCAL)).replace('\\','/'):sha256_arquivo(p) for p in sorted(arquivos)}
            gravar_json(OUTPUT_LOCAL/'hashes_artefatos_analise_v4.json',hashes)
            for p in tqdm(sorted(OUTPUT_LOCAL.rglob('*')),desc='Publicando rascunho pós-teste',unit='arq'):
                if p.is_file():
                    alvo=DRAFT_DRIVE/p.relative_to(OUTPUT_LOCAL);alvo.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,alvo)
            for nome,esperado in hashes.items():assert sha256_arquivo(DRAFT_DRIVE/nome)==esperado
            relatorio={'status':'ANALISE_POS_TESTE_V4_PRONTA_PARA_REVISAO',
                'event_id_yolo_asta':YOLO_ASTA_EVENT_ID,'sha256_resultado_yolo_asta':EXPECTED_YOLO_ASTA_RESULT,
                'event_id_hough_asta':HOUGH_ASTA_EVENT_ID,'sha256_resultado_hough_asta':EXPECTED_HOUGH_ASTA_RESULT,
                'casos_evento':len(candidatos_evento),
                'negativos_incluidos':1,'casos_totais':len(indice_df),'folhas':len(list(folhas.glob('*.jpg'))),
                'sha256_zip_folhas':sha256_arquivo(zip_path),'sha256_indice':sha256_arquivo(indice_path),
                'renderizacao_preview':RENDERIZACAO_PREVIEW,'espessura_centerline_nativa_px':1,
                'composicao_painel':'4x2: frame inteiro e recorte deterministico; imagem limpa | referencia verde | YOLO ciano sem GT | Hough v4.1 vermelho sem GT',
                'regra_recorte_positivos':'centro_bbox_referencia_oficial','regra_recorte_negativo':'centro_maior_segmento_yolo; fallback centro_geometrico',
                'checkpoints_yolo_unicos':int(indice_df.sha256_checkpoint_yolo.nunique()),
                'checkpoints_hough_unicos':int(indice_df.sha256_checkpoint_hough.nunique()),
                'predicoes_hough_unicas':int(indice_df.sha256_segmentos_hough.nunique()),
                'predicoes_yolo_unicas':int(indice_df.sha256_segmentos_yolo.nunique()),
                'reducao_preview':'INTER_AREA','transformacao_alpha':'clip(sqrt(densidade)*1.8,0,0.85)',
                'artefatos_rascunho':len(hashes),'inferencia_executada':False,'retuning_realizado':False,
                'publicacao_final_aprovada':False}
            gravar_json(OUTPUT_LOCAL/'relatorio_rascunho_analise_v4.json',relatorio)
            shutil.copy2(OUTPUT_LOCAL/'relatorio_rascunho_analise_v4.json',DRAFT_DRIVE/'relatorio_rascunho_analise_v4.json')
            print(json.dumps(relatorio,indent=2,ensure_ascii=False))
            print('Envie folhas_revisao_qualitativa_asta_v4.zip e o relatório antes da publicação final.')
        ''') ,
        md(r'''
        ## 4. Publicação após revisão humana

        Só execute depois de compartilhar e revisar as folhas. A publicação não altera resultados;
        apenas registra a interpretação humana e congela tabelas/figuras pós-teste.
        ''') ,
        code(r'''
        APROVAR_PUBLICACAO_ANALISE=False
        NOTA_REVISAO_HUMANA='Unknown / to be confirmed'

        if not APROVAR_PUBLICACAO_ANALISE:
            print('Publicação final bloqueada. Revise primeiro as folhas qualitativas.')
        else:
            if NOTA_REVISAO_HUMANA.strip() in {'','Unknown / to be confirmed'}:
                raise RuntimeError('Registre uma síntese objetiva da revisão humana antes de publicar.')
            hashes_path=DRAFT_DRIVE/'hashes_artefatos_analise_v4.json'
            relatorio_path=DRAFT_DRIVE/'relatorio_rascunho_analise_v4.json'
            assert hashes_path.is_file() and relatorio_path.is_file()
            hashes=json.loads(hashes_path.read_text(encoding='utf-8'))
            relatorio=json.loads(relatorio_path.read_text(encoding='utf-8'))
            assert relatorio['sha256_resultado_yolo_asta']==EXPECTED_YOLO_ASTA_RESULT
            assert relatorio['sha256_resultado_hough_asta']==EXPECTED_HOUGH_ASTA_RESULT and not relatorio['inferencia_executada']
            # Versões anteriores podiam incluir o próprio relatório no manifesto antes de
            # sobrescrevê-lo com o conteúdo final, tornando esse único hash autocontraditório.
            # Arquivos de controle são validados semanticamente abaixo; os artefatos científicos
            # permanecem integralmente verificados pelo manifesto.
            controles={'hashes_artefatos_analise_v4.json','relatorio_rascunho_analise_v4.json'}
            hashes_artefatos={nome:esperado for nome,esperado in hashes.items() if Path(nome).name not in controles}
            divergencias=[]
            for nome,esperado in tqdm(sorted(hashes_artefatos.items()),desc='Verificando rascunho aprovado',unit='arq'):
                atual=sha256_arquivo(DRAFT_DRIVE/nome)
                if atual!=esperado:divergencias.append({'arquivo':nome,'esperado':esperado,'atual':atual})
            if divergencias:
                raise RuntimeError(f'Artefatos científicos divergentes no rascunho: {divergencias[:5]}')
            assert sha256_arquivo(DRAFT_DRIVE/'folhas_revisao_qualitativa_asta_v4.zip')==relatorio['sha256_zip_folhas']
            assert sha256_arquivo(DRAFT_DRIVE/'indice_revisao_qualitativa_asta_v4.csv')==relatorio['sha256_indice']
            if (FINAL_DRIVE/'resumo_final_analise_pos_teste_v4.json').exists():
                raise RuntimeError('Análise pós-teste já publicada; sobrescrita é proibida.')
            for p in tqdm(sorted(DRAFT_DRIVE.rglob('*')),desc='Publicando análise final',unit='arq'):
                if p.is_file():
                    alvo=FINAL_DRIVE/p.relative_to(DRAFT_DRIVE);alvo.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,alvo)
            resumo_final={'protocolo':'post_test_analysis_v4','status':'ANALISE_POS_TESTE_V4_PUBLICADA',
                'publicado_em_utc':datetime.now(timezone.utc).isoformat(),
                'event_id_yolo_asta':YOLO_ASTA_EVENT_ID,'sha256_resultado_yolo_asta':EXPECTED_YOLO_ASTA_RESULT,
                'event_id_hough_asta':HOUGH_ASTA_EVENT_ID,'sha256_resultado_hough_asta':EXPECTED_HOUGH_ASTA_RESULT,
                'event_id_sintetico':SYNTH_EVENT_ID,
                'sha256_resultado_sintetico':EXPECTED_SYNTH_RESULT,'revisao_humana_aprovada':True,
                'nota_revisao_humana':NOTA_REVISAO_HUMANA.strip(),'casos_revisados':relatorio['casos_totais'],
                'inferencia_executada':False,'retuning_realizado':False,'resultados_experimentais_alterados':False}
            resumo_final['sha256_analise_final']=sha256_canonico(resumo_final,ensure_ascii=False)
            gravar_json(FINAL_DRIVE/'resumo_final_analise_pos_teste_v4.json',resumo_final)
            print(json.dumps(resumo_final,indent=2,ensure_ascii=False))
            print('ANÁLISE PÓS-TESTE PUBLICADA. Resultados experimentais permanecem imutáveis.')
        ''') ,
    ]
    return write_notebook('14_analise_pos_teste_erros_figuras_finais_v3.ipynb', cells)


def build_15():
    cells = [
        md(r'''
        # 15 — Auditoria e correção Hough-only no ASTA (v4)

        Este notebook corrige uma divergência de implementação descoberta **depois** do evento
        ASTA v3. O padrão vermelho observado tem periodicidade igual ao stride de 384 px dos tiles.
        A auditoria do código encontrou duas diferenças em relação ao baseline sintético congelado:

        1. `minLineLength` usou `512 × 0,25 = 128 px`, em vez da diagonal congelada
           `hypot(512,512) × 0,25 = 181 px`;
        2. o NMS exato aplicado depois de `HoughLinesP` no sintético não foi reproduzido por tile.

        O evento v3 permanece imutável. Este notebook não carrega nem executa YOLO, não busca
        parâmetros e não altera Canny, limiar Hough, overlap, normalização, merging ou métricas.
        Primeiro executa um diagnóstico Hough-only em um frame já observado; os 178 frames só são
        liberados após revisão explícita e publicação de um contrato corretivo versionado.

        Execute em **CPU**.
        ''') ,
        code(r'''
        %pip install -q opencv-python-headless pillow pandas tqdm scikit-image

        from google.colab import drive
        from pathlib import Path
        from datetime import datetime, timezone
        from tqdm.auto import tqdm
        from PIL import Image, ImageDraw, ImageFont
        from skimage.morphology import skeletonize
        import gc, hashlib, json, math, os, shutil, time, uuid
        import cv2
        import numpy as np
        import pandas as pd

        Image.MAX_IMAGE_PIXELS=200_000_000
        drive.mount('/content/drive',force_remount=True)

        BACKUP_DIR=Path('/content/drive/MyDrive/tcc-satellite-streaks')
        ASTA_RAW=BACKUP_DIR/'data/raw/asta_real'
        AUDIT_MANIFEST=BACKUP_DIR/'experiments/asta_external_v3/audit/final/manifest_auditoria_asta_v3.csv'
        PROTOCOL_PATH=BACKUP_DIR/'experiments/asta_external_v3/protocol/final/contrato_protocolo_asta_v3.json'
        HOUGH_CONFIG_PATH=BACKUP_DIR/'data/synthetic_runs/synthetic_v3_c41d5091de56/hough_v3/config_congelada_v3.json'
        OLD_EVENT_ID='4edef981-5590-4330-8dba-bd4d14dbcdda'
        OLD_EVENT=BACKUP_DIR/'experiments/asta_external_v3/inference'/f'evento_{OLD_EVENT_ID}'
        OLD_CONS=OLD_EVENT/'consolidado'
        CORR_ROOT=BACKUP_DIR/'experiments/hough_asta_correction_v4'
        DIAG_DIR=CORR_ROOT/'diagnostic'
        CONTRACT_DIR=CORR_ROOT/'contract'
        EVENTS_DIR=CORR_ROOT/'events'
        LOCK_PATH=CORR_ROOT/'TRAVA_EVENTO_HOUGH_ASTA_V4.json'
        LOCAL=Path('/content/hough_asta_correction_v4');STAGING=LOCAL/'staging'
        for p in [CORR_ROOT,DIAG_DIR,CONTRACT_DIR,EVENTS_DIR,LOCAL,STAGING]:p.mkdir(parents=True,exist_ok=True)

        EXPECTED_PROTOCOL='dc4c58381b1e3e7d2a24f7fdac8fc59276e53b9d900853c1191a497319e862bc'
        EXPECTED_MANIFEST='91bfa07ebffceaa46271a254d3862348ce2b242a06ed44fc4c07efeeb58722ca'
        EXPECTED_HOUGH_CONFIG='14ef0ea56a2e86bf1581225c4b9a0733d482834e5f7d3fab18b3eaf76f03a2e5'
        EXPECTED_OLD_RESULT='eef0f468754b195984a4888cc8c8439f12b5e9db3af94e875082272503542f7c'
        DIAGNOSTIC_ID='ML1_20220525_195141_red'

        def sha256_arquivo(caminho,chunk=8*1024*1024):
            h=hashlib.sha256()
            with open(caminho,'rb') as f:
                for bloco in iter(lambda:f.read(chunk),b''):h.update(bloco)
            return h.hexdigest()
        def sha256_canonico(objeto):
            texto=json.dumps(objeto,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            return hashlib.sha256(texto.encode('utf-8')).hexdigest()
        def verificar_canonico(registro,campo):
            copia=dict(registro);observado=copia.pop(campo)
            if sha256_canonico(copia)!=observado:raise RuntimeError(f'Hash canônico divergente: {campo}')
            return observado
        def gravar_json_atomico(caminho,objeto):
            caminho=Path(caminho);caminho.parent.mkdir(parents=True,exist_ok=True)
            tmp=caminho.with_name(caminho.name+'.tmp')
            tmp.write_text(json.dumps(objeto,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,caminho)
        def copiar_com_hash(origem,destino,esperado,tentativas=3):
            ultimo=None
            for tentativa in range(1,tentativas+1):
                try:
                    if destino.exists():destino.unlink()
                    shutil.copy2(origem,destino)
                    if sha256_arquivo(destino)!=esperado:raise IOError('hash divergente')
                    return
                except (OSError,IOError) as exc:
                    ultimo=exc
                    if destino.exists():destino.unlink()
                    if tentativa<tentativas:time.sleep(2**tentativa)
            raise IOError(f'Falha de cópia após {tentativas} tentativas: {ultimo}')

        print('Notebook 15 preparado em CPU. YOLO não será carregado nem executado.')
        ''') ,
        md(r'''
        ## 1. Gate e contrato técnico candidato — sem inferência

        Confirma o evento v3, o protocolo, o manifest e a configuração Hough congelada. A correção
        é derivada diretamente do código do Notebook 06; nenhum valor é escolhido pelo ASTA.
        ''') ,
        code(r'''
        assert sha256_arquivo(AUDIT_MANIFEST)==EXPECTED_MANIFEST
        manifest=pd.read_csv(AUDIT_MANIFEST).sort_values('id_asta').reset_index(drop=True)
        assert len(manifest)==178 and manifest.id_asta.nunique()==178 and DIAGNOSTIC_ID in set(manifest.id_asta)
        protocolo=json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
        assert verificar_canonico(protocolo,'sha256_contrato_protocolo_asta')==EXPECTED_PROTOCOL
        hough_config=json.loads(HOUGH_CONFIG_PATH.read_text(encoding='utf-8'))
        assert hough_config['sha256_config']==EXPECTED_HOUGH_CONFIG
        old_resumo=json.loads((OLD_CONS/'resumo_final_evento_asta_v3.json').read_text(encoding='utf-8'))
        assert verificar_canonico(old_resumo,'sha256_resultado_final')==EXPECTED_OLD_RESULT
        assert len(list((OLD_EVENT/'checkpoints/yolo').glob('*.json')))==178
        assert len(list((OLD_EVENT/'checkpoints/hough').glob('*.json')))==178

        cfg_asta=protocolo['configuracao'];d=hough_config['detector']
        TILE=int(cfg_asta['tiles']['tamanho']);STRIDE=int(cfg_asta['tiles']['stride'])
        STARTS_X=[i*STRIDE for i in range(int(cfg_asta['tiles']['grid_x']))]
        STARTS_Y=[i*STRIDE for i in range(int(cfg_asta['tiles']['grid_y']))]
        MIN_LENGTH_SINTETICO=int(math.hypot(TILE,TILE)*float(d['min_length_frac']))
        MIN_LENGTH_V3=int(round(TILE*float(d['min_length_frac'])))
        assert (TILE,STRIDE,len(STARTS_X),len(STARTS_Y))==(512,384,28,28)
        assert MIN_LENGTH_V3==128 and MIN_LENGTH_SINTETICO==181

        CONFIG_CORRECAO={
            'protocolo':'hough_asta_equivalence_correction_v4_1','versao':'4.1',
            'base_evento_v3':OLD_EVENT_ID,'sha256_resultado_v3':EXPECTED_OLD_RESULT,
            'sha256_contrato_asta':EXPECTED_PROTOCOL,'sha256_manifest_asta':EXPECTED_MANIFEST,
            'sha256_hough_config':EXPECTED_HOUGH_CONFIG,
            'detector':d,
            'correcoes':{
                'min_line_length_formula':'int(hypot(tile_height,tile_width)*min_length_frac)',
                'min_line_length_px':MIN_LENGTH_SINTETICO,
                'nms_por_tile':{'angulo_max_graus':3.0,'dist_max_px':12.0,'cobertura_min':0.30,
                    'origem':'funcao nms_linhas congelada no Notebook 06'},
                'costura_tiles':{'metodo':'particao_proprietaria_por_meio_da_sobreposicao',
                    'regra':'cada pixel global pertence exatamente a um tile; segmentos locais sao recortados ao dominio proprietario antes da uniao binaria',
                    'origem':'geometria congelada tile=512, stride=384, overlap=128',
                    'merging_global_aplicado':False}},
            'inalterado':{'tile':512,'stride':384,'overlap':128,'grid':'28x28',
                'normalizacao':'percentis exatos uint8 1.0-99.8 por frame',
                'canny':[int(d['canny_lo']),int(d['canny_hi'])],
                'hough_threshold':int(d['hough_threshold']),'max_gap':int(d['max_gap']),
                'tolerancia_centerline_px':14.0},
            'substituido_por_defeito_de_equivalencia':{
                'merging_global_v3':cfg_asta['merging_global'],
                'motivo':'o Notebook 06 nao possui merging transitive tiled; a costura deve apenas reconciliar dominios sobrepostos'},
            'proibicoes':{'busca_parametros':True,'recalibracao_asta':True,'execucao_yolo':True,
                'sobrescrever_evento_v3':True},
            'diagnostico_unico_id':DIAGNOSTIC_ID,'bootstrap_replicacoes':2000,'bootstrap_seed':20260823}
        CONFIG_CORRECAO['sha256_config_correcao']=sha256_canonico(CONFIG_CORRECAO)
        print(json.dumps({'status':'GATE_CORRECAO_HOUGH_V4_1_APROVADO_SEM_INFERENCIA',
            'diagnostico_id':DIAGNOSTIC_ID,'min_length_v3_px':MIN_LENGTH_V3,
            'min_length_fiel_sintetico_px':MIN_LENGTH_SINTETICO,
            'nms_sintetico_reposto':True,'costura_proprietaria_pre_registrada':True,
            'sha256_config_correcao':CONFIG_CORRECAO['sha256_config_correcao'],
            'yolo_executado':False,'hough_executado':False},indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 2. Implementação fiel e testes de equivalência

        O NMS vetorizado abaixo preserva a ordem gulosa do código sintético. Ele é comparado com
        uma referência escalar em dados artificiais antes que qualquer pixel ASTA seja lido.
        ''') ,
        code(r'''
        def percentil_histograma_uint8(imagem,q):
            hist=np.bincount(imagem.ravel(),minlength=256);alvo=(imagem.size-1)*float(q)/100
            return int(np.searchsorted(np.cumsum(hist),alvo,side='right'))
        def normalizar_frame_uint8(imagem):
            lo=percentil_histograma_uint8(imagem,1.0);hi=percentil_histograma_uint8(imagem,99.8)
            if hi<=lo:hi=min(255,lo+1)
            lut=np.clip((np.arange(256,dtype=np.float32)-lo)*255/(hi-lo),0,255).round().astype(np.uint8)
            norm=lut[imagem];return norm,int(percentil_histograma_uint8(norm,50.0))
        def extrair_tile(imagem,x,y,preenchimento):
            h,w=imagem.shape;tile=np.full((TILE,TILE),preenchimento,np.uint8)
            vh=max(0,min(TILE,h-y));vw=max(0,min(TILE,w-x))
            if vh and vw:tile[:vh,:vw]=imagem[y:y+vh,x:x+vw]
            return tile
        def parametros_linha(p0,p1):
            p0=np.asarray(p0,float);p1=np.asarray(p1,float);v=p1-p0;l=float(np.linalg.norm(v))
            return (p0+p1)/2,v/max(l,1e-9),l
        def erro_angular(d1,d2):return float(np.degrees(np.arccos(np.clip(abs(np.dot(d1,d2)),-1,1))))
        def distancia_ponto_linha(p,c,d):
            v=np.asarray(p)-np.asarray(c);return float(np.linalg.norm(v-np.dot(v,d)*d))
        def cobertura_projetada(p0a,p1a,p0b,p1b):
            cb,db,lb=parametros_linha(p0b,p1b)
            if lb<=0:return 0.0
            t=sorted([np.dot(np.asarray(p0a)-cb,db),np.dot(np.asarray(p1a)-cb,db)])
            return float(max(0,min(t[1],lb/2)-max(t[0],-lb/2))/lb)
        def nms_escalar_referencia(linhas):
            ordenadas=sorted(linhas,key=lambda l:-math.dist(l[0],l[1]));mantidas=[]
            for atual in ordenadas:
                ca,da,_=parametros_linha(*atual);suprimir=False
                for m in mantidas:
                    cm,dm,_=parametros_linha(*m)
                    if erro_angular(da,dm)>3.0:continue
                    if max(distancia_ponto_linha(ca,cm,dm),distancia_ponto_linha(cm,ca,da))>12.0:continue
                    if max(cobertura_projetada(*atual,*m),cobertura_projetada(*m,*atual))>=0.30:
                        suprimir=True;break
                if not suprimir:mantidas.append(atual)
            return mantidas
        def nms_vetorizado_exato(linhas):
            if not linhas:return []
            p1=np.asarray([l[0] for l in linhas],float);p2=np.asarray([l[1] for l in linhas],float)
            vec=p2-p1;compr=np.linalg.norm(vec,axis=1);ordem=np.argsort(-compr,kind='stable')
            centros=(p1+p2)/2;dirs=vec/np.maximum(compr[:,None],1e-9);mantidos=[]
            for idx in ordem.tolist():
                if not mantidos:mantidos.append(idx);continue
                k=np.asarray(mantidos,int);dot=np.abs(dirs[k]@dirs[idx]);ang=np.degrees(np.arccos(np.clip(dot,-1,1)))
                possiveis=k[ang<=3.0]
                suprimir=False
                for j in possiveis.tolist():
                    if max(distancia_ponto_linha(centros[idx],centros[j],dirs[j]),
                           distancia_ponto_linha(centros[j],centros[idx],dirs[idx]))>12.0:continue
                    atual=(p1[idx],p2[idx]);m=(p1[j],p2[j])
                    if max(cobertura_projetada(*atual,*m),cobertura_projetada(*m,*atual))>=0.30:
                        suprimir=True;break
                if not suprimir:mantidos.append(idx)
            return [linhas[i] for i in mantidos]

        def angulo_segmento(s):return math.degrees(math.atan2(s['y2']-s['y1'],s['x2']-s['x1']))%180
        def agregar_componente(segmentos,indices):
            pts=[];pesos=[]
            for i in indices:
                s=segmentos[i];peso=max(float(s['peso']),1e-6)
                pts.extend([[s['x1'],s['y1']],[s['x2'],s['y2']]]);pesos.extend([peso,peso])
            pts=np.asarray(pts,float);w=np.asarray(pesos,float);centro=np.average(pts,axis=0,weights=w)
            d=pts-centro;cov=(d*w[:,None]).T@d/max(w.sum(),1e-9);_,v=np.linalg.eigh(cov);eixo=v[:,-1]
            proj=d@eixo;p1=centro+eixo*proj.min();p2=centro+eixo*proj.max()
            return {'x1':float(p1[0]),'y1':float(p1[1]),'x2':float(p2[0]),'y2':float(p2[1]),
                'comprimento':float(np.linalg.norm(p2-p1)),'fragmentos':len(indices),'confianca_max':0.0}
        def unir_segmentos(segmentos):
            n=len(segmentos)
            if n==0:return []
            pai=list(range(n))
            def raiz(i):
                while pai[i]!=i:pai[i]=pai[pai[i]];i=pai[i]
                return i
            def unir(i,j):
                a,b=raiz(i),raiz(j)
                if a!=b:pai[b]=a
            p1=np.asarray([[s['x1'],s['y1']] for s in segmentos],float);p2=np.asarray([[s['x2'],s['y2']] for s in segmentos],float)
            vec=p2-p1;norm=np.linalg.norm(vec,axis=1);u=vec/np.maximum(norm[:,None],1e-9)
            ang=np.degrees(np.arctan2(vec[:,1],vec[:,0]))%180;meios=(p1+p2)/2
            limite=cfg_asta['merging_global'];maxa=float(limite['angulo_max_graus']);maxd=float(limite['distancia_perpendicular_max_px']);maxg=float(limite['gap_axial_max_px'])
            def avaliar(ia,ib,mesmo):
                ia=np.asarray(ia,int);ib=np.asarray(ib,int)
                for ini in range(0,len(ia),128):
                    a=ia[ini:ini+128];dif=np.abs(ang[a,None]-ang[ib][None,:])%180;mask=np.minimum(dif,180-dif)<=maxa
                    if mesmo:mask&=ib[None,:]>a[:,None]
                    ra,rb=np.nonzero(mask)
                    if not len(ra):continue
                    ai=a[ra];bi=ib[rb];va=u[ai];vb=u[bi].copy();vb[np.einsum('ij,ij->i',va,vb)<0]*=-1
                    e=va+vb;e/=np.maximum(np.linalg.norm(e,axis=1)[:,None],1e-9);normal=np.column_stack((-e[:,1],e[:,0]))
                    keep=np.abs(np.einsum('ij,ij->i',meios[bi]-meios[ai],normal))<=maxd
                    ai=ai[keep];bi=bi[keep];e=e[keep]
                    if not len(ai):continue
                    a1=np.einsum('ij,ij->i',p1[ai],e);a2=np.einsum('ij,ij->i',p2[ai],e);b1=np.einsum('ij,ij->i',p1[bi],e);b2=np.einsum('ij,ij->i',p2[bi],e)
                    gap=np.maximum(0,np.maximum(np.minimum(a1,a2),np.minimum(b1,b2))-np.minimum(np.maximum(a1,a2),np.maximum(b1,b2)))
                    for i,j in zip(ai[gap<=maxg].tolist(),bi[gap<=maxg].tolist()):unir(i,j)
            buckets={}
            for i,s in enumerate(segmentos):buckets.setdefault((int(s['tile_x']),int(s['tile_y'])),[]).append(i)
            vistos=set()
            for chave in buckets:
                tx,ty=chave
                for ny in range(ty-1,ty+2):
                    for nx in range(tx-1,tx+2):
                        outra=(nx,ny)
                        if outra not in buckets:continue
                        par=(chave,outra) if chave<=outra else (outra,chave)
                        if par in vistos:continue
                        vistos.add(par);avaliar(buckets[par[0]],buckets[par[1]],par[0]==par[1])
            grupos={}
            for i in range(n):grupos.setdefault(raiz(i),[]).append(i)
            return [agregar_componente(segmentos,x) for x in grupos.values()]

        rng=np.random.default_rng(20260830);linhas_teste=[]
        for _ in range(80):
            p0=rng.uniform(0,512,2);a=rng.uniform(0,np.pi);l=rng.uniform(100,400);p1=p0+l*np.array([np.cos(a),np.sin(a)])
            linhas_teste.append((tuple(p0),tuple(p1)))
        ref=nms_escalar_referencia(linhas_teste);vet=nms_vetorizado_exato(linhas_teste)
        canon=lambda xs:[tuple(round(float(v),8) for p in x for v in p) for x in xs]
        assert canon(ref)==canon(vet)
        linhas_sobrepostas=[((0,0),(300,0)),((10,1),(290,1)),((0,30),(300,30)),((0,0),(0,300))]
        assert canon(nms_escalar_referencia(linhas_sobrepostas))==canon(nms_vetorizado_exato(linhas_sobrepostas))
        assert len(nms_vetorizado_exato(linhas_sobrepostas))==3
        print('NMS vetorizado equivalente ao baseline sintético em teste determinístico.')
        ''') ,
        md(r'''
        ## 3. Diagnóstico de um frame — Hough-only

        Altere somente `EXECUTAR_DIAGNOSTICO_HOUGH=True`. Esta etapa não executa YOLO e não
        publica um resultado final. Ela compara o checkpoint v3, a reposição fiel do detector ainda
        submetida ao merging tiled v3 e uma costura proprietária determinística. Nesta última, cada
        pixel do frame pertence a exatamente um tile, eliminando duplicação de sobreposição sem
        escolher parâmetros a partir do resultado ASTA.
        ''') ,
        code(r'''
        EXECUTAR_DIAGNOSTICO_HOUGH=False

        def limites_propriedade(starts,indice,n):
            inicio=0 if indice==0 else int(round((starts[indice-1]+TILE+starts[indice])/2))
            fim=n if indice==len(starts)-1 else int(round((starts[indice]+TILE+starts[indice+1])/2))
            return inicio,fim
        def validar_particao_proprietaria(starts,n):
            intervalos=[limites_propriedade(starts,i,n) for i in range(len(starts))]
            assert intervalos[0][0]==0 and intervalos[-1][1]==n
            assert all(a<b for a,b in intervalos)
            assert all(intervalos[i][1]==intervalos[i+1][0] for i in range(len(intervalos)-1))
            assert sum(b-a for a,b in intervalos)==n
            return intervalos
        PROPRIEDADE_X=validar_particao_proprietaria(STARTS_X,10560)
        PROPRIEDADE_Y=validar_particao_proprietaria(STARTS_Y,10560)
        def recortar_ao_dominio_proprietario(p1,p2,ix,iy):
            esquerda,direita=PROPRIEDADE_X[ix];topo,base=PROPRIEDADE_Y[iy]
            a=(int(round(p1[0])),int(round(p1[1])));b=(int(round(p2[0])),int(round(p2[1])))
            ok,q1,q2=cv2.clipLine((esquerda,topo,direita-esquerda,base-topo),a,b)
            if not ok:return None
            comp=float(math.dist(q1,q2))
            if comp<=0:return None
            return {'x1':float(q1[0]),'y1':float(q1[1]),'x2':float(q2[0]),'y2':float(q2[1]),
                'peso':comp,'comprimento':comp,'confianca':0.0,'tile_x':ix,'tile_y':iy}
        def detectar_hough_corrigido(imagem_norm,preenchimento,id_asta):
            segmentos=[];proprietarios=[];raw_total=0;nms_total=0;tempos={'tiles':0.0,'hough_nms':0.0}
            coords=[(ix,iy,x,y) for iy,y in enumerate(STARTS_Y) for ix,x in enumerate(STARTS_X)]
            for ix,iy,x,y in tqdm(coords,desc=f'{id_asta} Hough corrigido',unit='tile',leave=False):
                t=time.perf_counter();tile=extrair_tile(imagem_norm,x,y,preenchimento);tempos['tiles']+=time.perf_counter()-t
                base=cv2.GaussianBlur(tile,(0,0),float(d['pre_blur_sigma'])) if float(d.get('pre_blur_sigma',0))>0 else tile
                t=time.perf_counter();bordas=cv2.Canny(base,int(d['canny_lo']),int(d['canny_hi']))
                seed=int(hashlib.sha256(f'{id_asta}|{ix}|{iy}|hough_v4'.encode()).hexdigest()[:8],16)&0x7fffffff;cv2.setRNGSeed(seed)
                linhas=cv2.HoughLinesP(bordas,1,np.pi/180,threshold=int(d['hough_threshold']),minLineLength=MIN_LENGTH_SINTETICO,maxLineGap=int(d['max_gap']))
                locais=[] if linhas is None else [((int(a),int(b)),(int(c),int(e))) for a,b,c,e in np.asarray(linhas).reshape(-1,4)]
                raw_total+=len(locais);mantidas=nms_vetorizado_exato(locais);nms_total+=len(mantidas)
                for (x1,y1),(x2,y2) in mantidas:
                    p1=np.array([x1+x,y1+y],float);p2=np.array([x2+x,y2+y],float);centro=(p1+p2)/2
                    if not (0<=centro[0]<10560 and 0<=centro[1]<10560):continue
                    comp=float(np.linalg.norm(p2-p1));segmentos.append({'x1':float(p1[0]),'y1':float(p1[1]),'x2':float(p2[0]),'y2':float(p2[1]),'peso':comp,'comprimento':comp,'confianca':0.0,'tile_x':ix,'tile_y':iy})
                    recortado=recortar_ao_dominio_proprietario(p1,p2,ix,iy)
                    if recortado is not None:proprietarios.append(recortado)
                tempos['hough_nms']+=time.perf_counter()-t
            return segmentos,proprietarios,{'segmentos_houghlinesp':raw_total,'segmentos_pos_nms':nms_total,
                'segmentos_pos_costura_proprietaria':len(proprietarios),**tempos}
        def rasterizar(segmentos,shape):
            h,w=shape;canvas=np.zeros((h,w),np.uint8)
            for s in segmentos:
                p1=(int(round(s['x1'])),int(round(s['y1'])));p2=(int(round(s['x2'])),int(round(s['y2'])))
                ok,a,b=cv2.clipLine((0,0,w,h),p1,p2)
                if ok:cv2.line(canvas,a,b,1,1,cv2.LINE_8)
            return canvas
        def metricas_canvas(pred,gt,tol=14):
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tol+1,2*tol+1));gd=cv2.dilate(gt.astype(np.uint8),k);pd=cv2.dilate(pred,k)
            pp=int(pred.sum());gp=int(gt.sum());ph=int(((pred>0)&(gd>0)).sum());gh=int(((gt>0)&(pd>0)).sum())
            p=ph/pp if pp else (1.0 if gp==0 else 0.0);r=gh/gp if gp else None;f=2*p*r/(p+r) if r is not None and p+r else (0.0 if gp else None)
            return {'pred_pixels':pp,'gt_pixels':gp,'pred_hits':ph,'gt_hits':gh,'precisao_centerline':float(p),'recall_centerline':None if r is None else float(r),'f1_centerline':None if f is None else float(f)}
        def diagnostico_grade(segmentos,pred,band=3):
            pontos=np.asarray([[s[k] for k in ['x1','y1','x2','y2']] for s in segmentos],float).reshape(-1,2) if segmentos else np.empty((0,2))
            dist=lambda v:np.minimum(np.mod(v,STRIDE),STRIDE-np.mod(v,STRIDE))
            alinh=float(np.mean((dist(pontos[:,0])<=band)|(dist(pontos[:,1])<=band))) if len(pontos) else 0.0
            ang=np.asarray([angulo_segmento(s) for s in segmentos]);eixo=float(np.mean((np.minimum(ang,180-ang)<=3)|(np.abs(ang-90)<=3))) if len(ang) else 0.0
            rows=sorted({v for y in STARTS_Y for v in range(max(0,y-band),min(pred.shape[0],y+band+1))})
            cols=sorted({v for x in STARTS_X for v in range(max(0,x-band),min(pred.shape[1],x+band+1))})
            total=int(pred.sum());hits=int(pred[rows,:].sum()+pred[:,cols].sum()-pred[np.ix_(rows,cols)].sum()) if total else 0
            area=1-(1-len(rows)/pred.shape[0])*(1-len(cols)/pred.shape[1]);frac=hits/total if total else 0.0
            return {'segmentos':len(segmentos),'endpoints_proximos_grade_frac':alinh,'segmentos_quase_horiz_vert_frac':eixo,'pixels_em_faixa_grade_frac':frac,'fracao_area_faixa_grade':float(area),'enriquecimento_grade':float(frac/area) if area else None}
        def overlay(base,dens,cor):
            alpha=np.clip(np.sqrt(np.maximum(dens,0))*1.8,0,0.85)[...,None];c=np.asarray(cor,float).reshape(1,1,3)
            return np.clip(base*(1-alpha)+c*alpha,0,255).astype(np.uint8)

        if not EXECUTAR_DIAGNOSTICO_HOUGH:
            print('Diagnóstico bloqueado. Altere somente EXECUTAR_DIAGNOSTICO_HOUGH=True.')
        else:
            linha=manifest.set_index('id_asta').loc[DIAGNOSTIC_ID]
            imgp=STAGING/linha.arquivo_imagem;maskp=STAGING/linha.arquivo_mascara
            copiar_com_hash(ASTA_RAW/linha.arquivo_imagem,imgp,linha.sha256_imagem);copiar_com_hash(ASTA_RAW/linha.arquivo_mascara,maskp,linha.sha256_mascara)
            imagem=cv2.imread(str(imgp),cv2.IMREAD_GRAYSCALE);mascara=cv2.imread(str(maskp),cv2.IMREAD_GRAYSCALE)
            assert imagem.shape==mascara.shape==(10560,10560);norm,med=normalizar_frame_uint8(imagem);gt=skeletonize(mascara>0).astype(np.uint8)
            atual=json.loads((OLD_EVENT/'checkpoints/hough'/f'{DIAGNOSTIC_ID}.json').read_text(encoding='utf-8'))
            assert atual['event_id']==OLD_EVENT_ID and atual['id_asta']==DIAGNOSTIC_ID and verificar_canonico(atual,'sha256_checkpoint')==atual['sha256_checkpoint']
            pred_atual=rasterizar(atual['segmentos'],gt.shape);grid_atual=diagnostico_grade(atual['segmentos'],pred_atual)
            t=time.perf_counter();brutos,proprietarios,stats=detectar_hough_corrigido(norm,med,DIAGNOSTIC_ID)
            t_merge=time.perf_counter();unidos=unir_segmentos(brutos);tempo_merge=time.perf_counter()-t_merge
            pred_merge=rasterizar(unidos,gt.shape);met_merge=metricas_canvas(pred_merge,gt);grid_merge=diagnostico_grade(unidos,pred_merge)
            pred_owner=rasterizar(proprietarios,gt.shape);met_owner=metricas_canvas(pred_owner,gt);grid_owner=diagnostico_grade(proprietarios,pred_owner)
            preview=640;base=cv2.cvtColor(cv2.resize(norm,(preview,preview),interpolation=cv2.INTER_AREA),cv2.COLOR_GRAY2RGB).astype(float)
            dens_gt=cv2.resize(gt.astype(np.float32),(preview,preview),interpolation=cv2.INTER_AREA)
            dens_a=cv2.resize(pred_atual.astype(np.float32),(preview,preview),interpolation=cv2.INTER_AREA)
            dens_m=cv2.resize(pred_merge.astype(np.float32),(preview,preview),interpolation=cv2.INTER_AREA)
            dens_o=cv2.resize(pred_owner.astype(np.float32),(preview,preview),interpolation=cv2.INTER_AREA)
            quadros=[base.astype(np.uint8),overlay(base,dens_gt,(0,255,80)),overlay(base,dens_a,(255,60,40)),
                overlay(base,dens_m,(255,60,40)),overlay(base,dens_o,(255,60,40))]
            canvas=Image.new('RGB',(preview*3,preview*2+110),'white');draw=ImageDraw.Draw(canvas)
            try:fonte=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
            except Exception:fonte=ImageFont.load_default()
            titulos=['Imagem limpa','GT oficial — verde',f'Hough v3 — {len(atual["segmentos"])} segmentos',
                f'Fiel + merging v3 — {len(unidos)} segmentos',f'Fiel + costura proprietária — {len(proprietarios)} segmentos']
            for i,(q,tit) in enumerate(zip(quadros,titulos)):
                x=(i%3)*preview;y=70+(i//3)*preview;canvas.paste(Image.fromarray(q),(x,y));draw.text((x+10,45+(i//3)*preview),tit,fill='black',font=fonte)
            draw.text((10,10),f'{DIAGNOSTIC_ID} | auditoria de costura, sem YOLO e sem busca de parâmetros',fill='black',font=fonte)
            imagem_saida=DIAG_DIR/'comparacao_hough_v3_merge_costura_proprietaria_um_frame.png';canvas.save(imagem_saida)
            rel={'protocolo':'hough_asta_diagnostic_v4_1','status':'DIAGNOSTICO_COSTURA_HOUGH_UM_FRAME_CONCLUIDO','id_asta':DIAGNOSTIC_ID,
                'sha256_config_correcao':CONFIG_CORRECAO['sha256_config_correcao'],'hough_v3':{'metricas':atual['metricas'],'grade':grid_atual,'segmentos_apos_merging':len(atual['segmentos'])},
                'hough_fiel_com_merging_v3':{'metricas':met_merge,'grade':grid_merge,'segmentos_houghlinesp':stats['segmentos_houghlinesp'],
                    'segmentos_pos_nms':stats['segmentos_pos_nms'],'segmentos_apos_merging':len(unidos)},
                'hough_fiel_com_costura_proprietaria':{'metricas':met_owner,'grade':grid_owner,
                    'segmentos_houghlinesp':stats['segmentos_houghlinesp'],'segmentos_pos_nms':stats['segmentos_pos_nms'],
                    'segmentos_pos_costura_proprietaria':len(proprietarios),'cobertura_particao_exata':True,'merging_global_aplicado':False},
                'tempos_segundos':{'total_diagnostico':time.perf_counter()-t,'merging_v3_comparativo':tempo_merge},'yolo_executado':False,'busca_parametros':False,
                'sha256_imagem_diagnostico':sha256_arquivo(imagem_saida)}
            rel['sha256_relatorio_diagnostico']=sha256_canonico(rel);gravar_json_atomico(DIAG_DIR/'relatorio_diagnostico_hough_v4_1.json',rel)
            print(json.dumps(rel,indent=2,ensure_ascii=False));print('Envie o relatório e comparacao_hough_v3_merge_costura_proprietaria_um_frame.png antes da seção 4.')
            for p in [imgp,maskp]:
                if p.exists():p.unlink()
            del imagem,mascara,norm,gt,pred_atual,pred_merge,pred_owner,base,dens_gt,dens_a,dens_m,dens_o,canvas;gc.collect()
        ''') ,
        md(r'''
        ## 4. Publicação do contrato corretivo

        Só habilite após revisar o diagnóstico. A aprovação confirma correção de equivalência,
        não escolha do melhor resultado.
        ''') ,
        code(r'''
        APROVAR_CORRECAO_HOUGH=False
        NOTA_APROVACAO='Unknown / to be confirmed'

        CONTRACT_PATH=CONTRACT_DIR/'contrato_correcao_hough_asta_v4.json'
        if not APROVAR_CORRECAO_HOUGH:
            print('Contrato corretivo bloqueado. Revise primeiro o diagnóstico de um frame.')
        else:
            if NOTA_APROVACAO.strip() in {'','Unknown / to be confirmed'}:raise RuntimeError('Registre a justificativa de aprovação.')
            diag=json.loads((DIAG_DIR/'relatorio_diagnostico_hough_v4_1.json').read_text(encoding='utf-8'))
            assert verificar_canonico(diag,'sha256_relatorio_diagnostico')==diag['sha256_relatorio_diagnostico']
            assert diag['sha256_config_correcao']==CONFIG_CORRECAO['sha256_config_correcao'] and not diag['yolo_executado'] and not diag['busca_parametros']
            assert diag['status']=='DIAGNOSTICO_COSTURA_HOUGH_UM_FRAME_CONCLUIDO'
            assert diag['hough_fiel_com_costura_proprietaria']['cobertura_particao_exata']
            assert not diag['hough_fiel_com_costura_proprietaria']['merging_global_aplicado']
            contrato={'protocolo':'hough_asta_correction_contract_v4','status':'APROVADO_PARA_EVENTO_HOUGH_ONLY',
                'aprovado_em_utc':datetime.now(timezone.utc).isoformat(),'configuracao':CONFIG_CORRECAO,
                'sha256_relatorio_diagnostico':diag['sha256_relatorio_diagnostico'],'nota_aprovacao':NOTA_APROVACAO.strip(),
                'evento_v3_preservado':True,'yolo_reexecucao_proibida':True,'ajuste_por_resultado_asta':False}
            contrato['sha256_contrato_correcao']=sha256_canonico(contrato)
            if CONTRACT_PATH.exists():
                existente=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'));assert existente==contrato
            else:gravar_json_atomico(CONTRACT_PATH,contrato)
            print(json.dumps(contrato,indent=2,ensure_ascii=False))
        ''') ,
        md(r'''
        ## 5. Evento corretivo completo — 178 frames, somente Hough

        Execute somente após compartilhar o contrato aprovado. O evento é retomável por frame,
        grava em diretório novo e nunca modifica o evento v3 ou os checkpoints YOLO. O Hough é
        aplicado fielmente por tile e costurado pela partição proprietária pré-registrada; o merging
        global transitivo v3 não é utilizado.
        ''') ,
        code(r'''
        EXECUTAR_EVENTO_HOUGH_V4=False
        RETOMAR_EVENTO_HOUGH_V4=False

        def resumo_metodo(registros):
            pos=[r for r in registros if r['gt_positivo']];neg=[r for r in registros if not r['gt_positivo']]
            vals=[r['metricas']['f1_centerline'] for r in pos]
            pp=sum(r['metricas']['pred_pixels'] for r in registros);ph=sum(r['metricas']['pred_hits'] for r in registros)
            gp=sum(r['metricas']['gt_pixels'] for r in pos);gh=sum(r['metricas']['gt_hits'] for r in pos)
            p=ph/pp if pp else 0.0;r=gh/gp if gp else 0.0;f=2*p*r/(p+r) if p+r else 0.0
            mp=10560*10560/1e6;fp_neg=sum(r['metricas']['pred_pixels'] for r in neg)/max(len(neg)*mp,1e-9)
            return {'frames':len(registros),'frames_positivos':len(pos),'frames_negativos':len(neg),'f1_centerline_macro_frames_positivos':float(np.mean(vals)),
                'precisao_centerline_pooled':float(p),'recall_centerline_pooled':float(r),'f1_centerline_pooled':float(f),
                'comprimento_fp_por_megapixel_negativo':float(fp_neg),'taxa_frames_negativos_com_fp':float(np.mean([x['metricas']['pred_pixels']>0 for x in neg]))}
        def checkpoint_valido(path,event_id,id_asta):
            if not path.is_file():return None
            r=json.loads(path.read_text(encoding='utf-8'))
            if r.get('event_id')!=event_id or r.get('id_asta')!=id_asta or r.get('sha256_config_correcao')!=CONFIG_CORRECAO['sha256_config_correcao']:return None
            verificar_canonico(r,'sha256_checkpoint');return r

        if not EXECUTAR_EVENTO_HOUGH_V4:
            print('Evento completo bloqueado. Compartilhe primeiro o contrato corretivo.')
        else:
            if not CONTRACT_PATH.is_file():raise RuntimeError('Contrato corretivo ausente.')
            contrato=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'));verificar_canonico(contrato,'sha256_contrato_correcao')
            assert contrato['status']=='APROVADO_PARA_EVENTO_HOUGH_ONLY' and contrato['configuracao']['sha256_config_correcao']==CONFIG_CORRECAO['sha256_config_correcao']
            if LOCK_PATH.is_file():
                lock=json.loads(LOCK_PATH.read_text(encoding='utf-8'))
                if lock['status']=='concluido':raise RuntimeError('Evento Hough v4 já concluído; reexecução proibida.')
                if not RETOMAR_EVENTO_HOUGH_V4:raise RuntimeError('Evento incompleto existe. Use RETOMAR_EVENTO_HOUGH_V4=True.')
                assert lock['sha256_config_correcao']==CONFIG_CORRECAO['sha256_config_correcao'];event_id=lock['event_id']
            else:
                if RETOMAR_EVENTO_HOUGH_V4:raise RuntimeError('Não existe evento para retomar.')
                event_id=str(uuid.uuid4());lock={'protocolo':'hough_asta_correction_event_v4','event_id':event_id,'status':'em_andamento',
                    'iniciado_em_utc':datetime.now(timezone.utc).isoformat(),'sha256_config_correcao':CONFIG_CORRECAO['sha256_config_correcao'],'frames_concluidos':0}
                gravar_json_atomico(LOCK_PATH,lock)
            event_dir=EVENTS_DIR/f'evento_{event_id}';check_dir=event_dir/'checkpoints';cons=event_dir/'consolidado'
            check_dir.mkdir(parents=True,exist_ok=True);cons.mkdir(parents=True,exist_ok=True)
            for linha in tqdm(list(manifest.itertuples()),desc='Frames Hough ASTA v4',unit='frame'):
                cp=check_dir/f'{linha.id_asta}.json';existente=checkpoint_valido(cp,event_id,linha.id_asta)
                if existente is not None:continue
                imgp=STAGING/linha.arquivo_imagem;maskp=STAGING/linha.arquivo_mascara
                copiar_com_hash(ASTA_RAW/linha.arquivo_imagem,imgp,linha.sha256_imagem);copiar_com_hash(ASTA_RAW/linha.arquivo_mascara,maskp,linha.sha256_mascara)
                imagem=cv2.imread(str(imgp),cv2.IMREAD_GRAYSCALE);mascara=cv2.imread(str(maskp),cv2.IMREAD_GRAYSCALE);assert imagem.shape==mascara.shape==(10560,10560)
                t0=time.perf_counter();norm,med=normalizar_frame_uint8(imagem);tempo_pre=time.perf_counter()-t0
                brutos,proprietarios,stats=detectar_hough_corrigido(norm,med,linha.id_asta)
                gt=skeletonize(mascara>0).astype(np.uint8);pred=rasterizar(proprietarios,gt.shape);met=metricas_canvas(pred,gt);grade=diagnostico_grade(proprietarios,pred)
                reg={'protocolo':'hough_asta_corrected_frame_v4','event_id':event_id,'sha256_config_correcao':CONFIG_CORRECAO['sha256_config_correcao'],
                    'id_asta':linha.id_asta,'sha256_imagem':linha.sha256_imagem,'sha256_mascara':linha.sha256_mascara,'gt_positivo':bool(int(linha.mascara_pixels_positivos)>0),
                    'segmentos_houghlinesp':stats['segmentos_houghlinesp'],'segmentos_pos_nms':stats['segmentos_pos_nms'],
                    'segmentos_pos_costura_proprietaria':len(proprietarios),'segmentos':proprietarios,'metricas':met,'diagnostico_grade':grade,
                    'costura':'particao_proprietaria_sem_merging_global','tempos_segundos':{'preprocessamento':tempo_pre+stats['tiles'],
                    'inferencia_hough_nms_costura':stats['hough_nms'],'merging':0.0,
                    'total_frame':tempo_pre+stats['tiles']+stats['hough_nms']},'concluido':True}
                reg['sha256_checkpoint']=sha256_canonico(reg);gravar_json_atomico(cp,reg)
                lock['frames_concluidos']=len(list(check_dir.glob('*.json')));gravar_json_atomico(LOCK_PATH,lock)
                for p in [imgp,maskp]:
                    if p.exists():p.unlink()
                del imagem,mascara,norm,gt,pred,brutos,proprietarios;gc.collect()
            registros=[checkpoint_valido(check_dir/f'{i}.json',event_id,i) for i in manifest.id_asta.astype(str)]
            assert all(x is not None for x in registros) and len(registros)==178
            resumo_h=resumo_metodo(registros)
            yolo=[]
            for id_asta in manifest.id_asta.astype(str):
                r=json.loads((OLD_EVENT/'checkpoints/yolo'/f'{id_asta}.json').read_text(encoding='utf-8'));assert r['event_id']==OLD_EVENT_ID and r['id_asta']==id_asta;verificar_canonico(r,'sha256_checkpoint');yolo.append(r)
            positivos=[i for i,r in enumerate(registros) if r['gt_positivo']];rng=np.random.default_rng(20260823);boot=[]
            yh=np.asarray([yolo[i]['metricas']['f1_centerline'] for i in positivos],float);hh=np.asarray([registros[i]['metricas']['f1_centerline'] for i in positivos],float)
            for _ in tqdm(range(2000),desc='Bootstrap Hough corrigido',unit='rep'):
                idx=rng.integers(0,len(positivos),len(positivos));my=float(yh[idx].mean());mh=float(hh[idx].mean());boot.append((my,mh,my-mh))
            b=np.asarray(boot);ic={'f1_yolo':[float(x) for x in np.percentile(b[:,0],[2.5,97.5])],
                'f1_hough_corrigido':[float(x) for x in np.percentile(b[:,1],[2.5,97.5])],
                'delta_yolo_menos_hough':[float(x) for x in np.percentile(b[:,2],[2.5,97.5])]}
            resumo={'protocolo':'hough_asta_correction_event_v4','status':'EVENTO_HOUGH_ASTA_V4_CONCLUIDO','event_id':event_id,
                'sha256_config_correcao':CONFIG_CORRECAO['sha256_config_correcao'],'evento_v3_preservado':OLD_EVENT_ID,
                'resultado_hough_corrigido':resumo_h,'resultado_yolo_reutilizado_sem_inferencia':old_resumo['resultado_primario_yolo'],
                'bootstrap_pareado_frames_positivos':{'replicacoes':2000,'seed':20260823,'n_frames':177,'ic95_percentil':ic},
                'yolo_executado':False,'busca_parametros':False,'recalibracao_asta':False}
            resumo['sha256_resultado_final']=sha256_canonico(resumo);gravar_json_atomico(cons/'resumo_final_hough_asta_v4.json',resumo)
            tabela=pd.DataFrame([{'id_asta':r['id_asta'],**r['metricas'],'segmentos_houghlinesp':r['segmentos_houghlinesp'],
                'segmentos_pos_nms':r['segmentos_pos_nms'],'segmentos_pos_costura_proprietaria':r['segmentos_pos_costura_proprietaria'],
                **r['diagnostico_grade']} for r in registros])
            tabela.to_csv(cons/'metricas_hough_asta_v4.csv',index=False)
            lock.update({'status':'concluido','concluido_em_utc':datetime.now(timezone.utc).isoformat(),'frames_concluidos':178,'sha256_resultado_final':resumo['sha256_resultado_final']});gravar_json_atomico(LOCK_PATH,lock)
            print(json.dumps(resumo,indent=2,ensure_ascii=False));print('EVENTO Hough v4 concluído. YOLO não foi reexecutado; evento v3 preservado.')
        ''') ,
        md(r'''
        ## 6. Verificação pós-evento
        ''') ,
        code(r'''
        if not LOCK_PATH.is_file():
            print('Nenhum evento Hough v4 publicado ainda.')
        else:
            lock=json.loads(LOCK_PATH.read_text(encoding='utf-8'))
            if lock['status']!='concluido':
                print(json.dumps({'status':'EVENTO_HOUGH_V4_INCOMPLETO','event_id':lock['event_id'],'frames_concluidos':lock['frames_concluidos']},indent=2))
            else:
                event_dir=EVENTS_DIR/f"evento_{lock['event_id']}";arquivos=list((event_dir/'checkpoints').glob('*.json'));assert len(arquivos)==178
                for p in tqdm(arquivos,desc='Verificando checkpoints Hough v4',unit='frame'):
                    r=json.loads(p.read_text(encoding='utf-8'));verificar_canonico(r,'sha256_checkpoint');assert r['event_id']==lock['event_id']
                resumo=json.loads((event_dir/'consolidado/resumo_final_hough_asta_v4.json').read_text(encoding='utf-8'))
                assert verificar_canonico(resumo,'sha256_resultado_final')==lock['sha256_resultado_final']
                print(json.dumps({'status':'BACKUP_EVENTO_HOUGH_ASTA_V4_APROVADO','event_id':lock['event_id'],'frames':178,
                    'sha256_resultado_final':lock['sha256_resultado_final'],'evento_v3_preservado':True,'yolo_reexecutado':False},indent=2,ensure_ascii=False))
        ''') ,
    ]
    return write_notebook('15_auditoria_correcao_hough_asta_v4.ipynb', cells)


def write_readme():
    text = _src(r'''
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
    ''')
    path=OUT/'README_EXECUCAO.md';path.write_text(text,encoding='utf-8');return path


if __name__ == '__main__':
    paths=[build_01(),build_02(),build_03(),build_04(),build_05(),build_06(),build_07(),build_08(),build_09(),build_10(),build_11(),build_12(),build_13(),build_14(),build_15(),write_readme()]
    print('\n'.join(str(p) for p in paths))
