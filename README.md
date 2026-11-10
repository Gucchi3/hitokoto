# SLM

## 環境構築

```powershell
uv venv --python 3.14
uv pip install --python .venv rich huggingface-hub tokenizers pyarrow
```

PyTorchは、[公式のインストールページ](https://pytorch.org/get-started/locally/)でOS、Python、CUDAの構成を選び、表示されたGPU用のインストールコマンドを使用してください。表示されたコマンドの `pip3 install...` を `uv pip install...` に置き換えて実行します。GPUを使う場合は、PyTorchを通常のPyPIから選ばず、公式ページが指定するCUDA用のインデックスをそのまま使用してください。

## 実装されているモデル

- **Glint-2**
- **Hotaru**
- **Supra Mini v6**
- **Yokoyama**: EmbBERTを基にした、縮小埋め込み、差分スキップ注意、因果畳み込みを組み合わせたデコーダーモデル

## 測定可能なベンチマーク

- **BLiMP**: 文法的な文と非文法的な文を比較する正解率
- **ARC-Easy**: 選択式問題の正解率
- **WikiText-2**: バイト単位およびトークン単位のパープレキシティ
