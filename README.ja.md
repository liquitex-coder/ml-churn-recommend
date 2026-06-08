# 顧客離脱予兆検知 兼 クロスセル推奨システム (ML-Churn-Recommend)

> 日本語 README（メイン）。English: [README.md](README.md)

サブスク型／リピート型ビジネスの既存顧客の **解約（チャーン）リスクを早期検知** し、同時に
LTV を高める **最適なクロスセル提案を自動生成** するデモプロダクトです。マーケティング／
CS／営業の予算ホルダー向けに、離脱防止による売上維持効果と、データ根拠に基づく効率的な
アプローチリスト自動生成の価値を示します。

`git clone` 後、外部ネットワーク接続なし・同梱の合成データだけで即実行できます。

## バリュープロポジション

- **離脱の早期検知**: 顧客ごとに 0〜100% の離脱確率を算出（LightGBM）。
- **理由の説明 (Explainable AI)**: SHAP で「なぜ離脱リスクが高いか」を寄与度グラフ化。
- **クロスセル自動提案**: 協調フィルタ + グラフ解析で上位3製品を推奨。購入済みは除外。
- **即戦力ダッシュボード**: Streamlit 一画面でアラート・要因・推奨を同時提示。

## アーキテクチャ

```
                データ生成 (合成・シード固定 / FR-20, NFR-2)
                ┌───────────────────────────────────────────┐
                │ customers.csv  purchases.csv  support_texts │
                └───────────────────────────────────────────┘
                        │                        │
        ┌───────────────┴──────────┐             │
        ▼                          ▼             ▼
  text_analytics              churn_model    recommend
  TF-IDF 類似度 (FR-3)  ──hint──▶ LightGBM     item-item CF
  解約匂わせスコア              二値分類 (FR-1) + NetworkX グラフ
        │                       不均衡対応(FR-21)  (FR-6, FR-22)
        └───────FR-4 補正特徴量──▶│                  │
                                  ▼                  │
                              explain (SHAP)         │
                              要因可視化 (FR-5)        │
                                  │                  │
                                  ▼                  ▼
                       ┌──────────────────────────────────┐
                       │  app.py  Streamlit ダッシュボード    │
                       │  ①アラートリスト(FR-8)              │
                       │  ②SHAP寄与度(FR-5)                 │
                       │  ③クロスセル推奨(FR-6/FR-7/FR-22)   │
                       └──────────────────────────────────┘
```

## クイックスタート

```bash
# 1. 依存パッケージ
pip install -r requirements.txt

# 2. 合成データ生成（同梱済み。再生成する場合のみ・シード固定で再現可能）
python scripts/generate_data.py

# 3. ダッシュボード起動
streamlit run app.py

# 4. テスト
pytest -q
```

### データソースの切替（サンプル / 自社データ）
サイドバー先頭の「データソース」で、**同梱サンプル**と**自社CSVのアップロード**を切り替えられます。
アップロードでは顧客CSV（必須・テンプレDL可、必須列を検証して不足を明示）に加え、購買・問い合わせCSV
（任意。未指定ならサンプルにフォールバック）を読み込めます。同梱サンプルは生成器の既定サイズ
（顧客400件）で、ホールドアウト **AUC ≈ 0.785** を再現します。

Python 3.10 以上が必要です（NFR-1）。

## 合成データについて (NFR-2, FR-20)

`scripts/generate_data.py` が乱数シード固定（`config.SEED`）で以下を生成します。再実行しても
同一の出力になります（FR-20）。

| ファイル | 内容 |
|----------|------|
| `data/customers.csv` | Telco 風の顧客属性（在籍月数・月額・ログイン頻度・機能利用・支払失敗・サポート件数・契約種別・解約ラベル）。解約は少数クラス（FR-21）。 |
| `data/purchases.csv` | 顧客×製品の購買履歴。共起バンドルを含み item-item CF / グラフに信号を与える。 |
| `data/support_texts.csv` | 100 件以上の合成サポート問い合わせ。「解約したい」「他社に乗り換え」等の解約匂わせ／不満／中立／満足を混在。各テキストは `customer_id` に紐付く。 |

埋め込みは既定で TF-IDF（文字 n-gram、JP/EN 混在対応・オフライン）。任意で
`sentence-transformers` を入れると `text_analytics.embed_texts_sentence_transformers`
経由でその埋め込みも使えます（未インストール時は try/except で安全に無効化）。

## 要件トレーサビリティ

機械可読アンカー（`FR-x` / `NFR-x`）は以下に定義されています。

- 要件定義: [`docs/requirements/A_requirements.md`](docs/requirements/A_requirements.md)
- 受け入れ条件: [`docs/requirements/B_acceptance.md`](docs/requirements/B_acceptance.md)

各テストファイル冒頭にマッピングを記載し、テスト内に `# covers: FR-x` マーカーを置いて
FR-1〜FR-9 / FR-20〜FR-22 / NFR-1〜NFR-3 を網羅しています。

| 要件 | 実装 |
|------|------|
| FR-1 | `churn_model.ChurnModel.predict_proba_percent`（0〜100%） |
| FR-2 | `data_gen.generate_customers` + `churn_model.build_feature_frame` |
| FR-3 | `text_analytics.text_similarity_scores`（TF-IDF 既定 / ST 任意） |
| FR-4 | `compute_churn_hint_scores` → モデル特徴量に結合 |
| FR-5 | `explain.ChurnExplainer.explain_customer`（SHAP） |
| FR-6 | `recommend.CrossSellRecommender`（item-item CF + NetworkX） |
| FR-7 | `recommend_for_active_customers`（低リスク顧客対象） |
| FR-8 | `app.py` アラートリスト（リスク％降順） |
| FR-9 | `app.py` Streamlit 一画面 3 ブロック |
| FR-20 | `scripts/generate_data.py`（シード固定） |
| FR-21 | LightGBM `is_unbalance=True` / `class_weight="balanced"` |
| FR-22 | `recommend.recommend`（購入済み除外） |
| NFR-1 | Python 3.10+ |
| NFR-2 | 同梱合成データのみ・オフライン |
| NFR-3 | ホールドアウト AUC / Accuracy |
