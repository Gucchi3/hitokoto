# SLM

## 環境構築

```powershell
uv venv --python 3.14
uv pip install --python .venv -r pyproject.toml
```

続いて、[PyTorch 公式](https://pytorch.org/get-started/locally/)で OS・CUDA に合うコマンドを選び、`pip3 install...` を `uv pip install...` に置き換えて実行します。

## 実行方法

### 初回のみ：データ・トークナイザの準備

使うデータを選び、ダウンロード → 分割 → トークナイザ作成 → トークン化を一括実行。  
指定したデータを全件取得・使用。未指定のデータは除外。

```powershell
python -m tool.data all --tinystories --fineweb_edu --finemath --tokenizer_dir .\model\weight\hotaru
```

> [!WARNING]
> 指定先の既存トークナイザは上書きされます。
### TinyGpt 用のバイト単位データ（初回のみ）

TinyGpt（`lab/tinylm_subbit` の ccca 構成）はバイト単位のモデルです（語彙 256、特殊トークンなし）。テキストファイルのバイトをそのままトークン ID にして、`data/tinystories_byte/` とトークナイザ `model/weight/tiny_gpt_byte/` を作ります。

```powershell
python -m tool.data.byte_text --train <TinyStoriesV2-GPT4-train の先頭 200 MB> --valid <TinyStoriesV2-GPT4-valid.txt>
```

### 学習

```powershell
python main.py --config .\config\hotaru.json
```

TinyGpt は `--config .\config\tiny_gpt.json` で学習します。学習のあと、`lab/tinylm_subbit` と同じ指標（valid の固定 1024 窓での次バイト正解率と bits/byte）を次のコマンドで測れます。

```powershell
python -m tool.evaluate.byte_window --config .\config\tiny_gpt.json --checkpoint .\log\<実行>\best.pth
```

## 使用可能なデータセット
>[!NOTE]
>`./tool/data/sources.py`を参照
- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)**
- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**
- **[Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/cosmopedia-v2)**
- **[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)**

## 実装されているモデル
>[!NOTE]
>`./model/`を参照
- **[Glint-2](https://huggingface.co/Glint-Research/Glint-2)**
- **[Supra Mini v6](https://huggingface.co/SupraLabs/Supra-Mini-v6-1M)**
- **Yokoyama**
- **Hotaru**
- **TinyGpt**（`lab/tinylm_subbit` の ccca 構成、バイト単位）

## 測定可能なベンチマーク
>[!NOTE]
>`./utils/validator/benchmark.py`を参照

`--mode benchmark` またはテストデータがない場合の学習後に実行。データは初回に自動ダウンロード。

### [Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard)

- **[BLiMP](https://huggingface.co/datasets/nyu-mll/blimp)** (`blimp`)
- **[ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc)** (`arc_easy_tinyml`)
- **[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)** (`wikitext2`)

### [Open SLM Leaderboard](https://axiomiclabs-open-slm-leaderboard.static.hf.space/index.html)

- **[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)** (`hellaswag`)
- **[ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc)** (`arc_easy`)
- **[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)** (`arc_challenge`)
- **[PIQA](https://huggingface.co/datasets/baber/piqa)** (`piqa`)
- **[ArithMark-3](https://huggingface.co/datasets/AxiomicLabs/Arithmark-3.0)** (`arithmark3`)

> [!WARNING]
> `arc_easy_tinyml` と `arc_easy` は測定方法が異なります。Int Index の ARC-Easy には `arc_easy` が使用されます。

- すべて `float32` で測定します。
- Open SLM は、選択肢が1つでも `data.max_length`（ArithMark-3 は `data.max_length - 1`）トークンを超える問題をスキップし、精度の分母から除外します。
### Int Index と Avg

$$
\mathrm{Int\ Index} =
\frac{
\widehat{\mathrm{HellaSwag}} + \widehat{\mathrm{ARC}} + \widehat{\mathrm{PIQA}} + 0.65 \times \widehat{\mathrm{ArithMark\text{-}3}}
}{
1 + 1 + 1 + 0.65
}
$$

$$
\widehat{s} = 100 \times \frac{s - c}{100 - c}
$$

- $s$: スコア（%）、$c$: チャンスレート（PIQA は 50、ほかは 25）。
- ARC = `arc_easy` と `arc_challenge` の平均（片方だけなら、その値）。
- 未実行の項目は分子・分母から除外。4項目中2項目以上で `int_index` と `open_slm_avg` を算出。
- Avg = 変換前のスコアの単純平均。

## テスト

```powershell
uv pip install --python .venv -r pyproject.toml --extra test
.venv\Scripts\python -m pytest
```
