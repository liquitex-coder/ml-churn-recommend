# 監査レポート — Claim Auditor 適用結果（顧客離脱予兆検知 兼 クロスセル推奨）

> 生成手段: `claim-audit place`（要件ギャップ）＋ `source_coverage.measure`（実装被覆）。
> いずれも LLM-free・決定論。要件アンカーは [`A_requirements.md`](A_requirements.md) / [`B_acceptance.md`](B_acceptance.md)。

## 1. 要件ギャップ（A=要件定義 ↔ B=受け入れ条件 の双方向 meet）

```
claim-audit place --spec docs/requirements/A_requirements.md \
                  --outcome docs/requirements/B_acceptance.md
```

| 領域 | アンカー | 意味 |
|---|---|---|
| **gap（抜け）** | `FR-20, FR-21, FR-22` | 受け入れ条件(B)が要求するが、元の要件定義書(A)に明示が無い項目 |
| **core（一致）** | `FR-1…FR-9, NFR-1…NFR-3`（12件） | A と B の両端で挟まれ確定した出荷コア |
| **over（A過剰）** | なし | A に在るが B が gate しない項目 |

### 検出された抜け（gap）への対応
元の要件定義書には記載が無かったが、デモとして出荷するには必要な 3 項目を機械検出した。
**いずれも本実装で手当て済み**（だが「要件段階では抜けていた」ことを記録する）:

- **FR-20** 合成データの再現性（シード固定）と生成スクリプト同梱 → `scripts/generate_data.py`
- **FR-21** クラス不均衡（離脱少数）の取り扱い → LightGBM `is_unbalance` / class weight
- **FR-22** 既購入製品を推奨から除外 → `recommend.recommend`

## 2. 実装被覆（要件アンカーがテストで witness されているか）

`source_coverage.measure`（`# covers: FR-x` マーカーの厳格 witness）:

| 指標 | 値 |
|---|---|
| 宣言アンカー（A） | 12 |
| 被覆（strict） | 12（**100.0%**） |
| 未被覆（実装の抜け） | なし |

加えて gap 項目 `FR-20 / FR-21 / FR-22` もテストに `# covers:` マーカーを持つ。

## 3. 結論
- 要件レベルの抜けは `FR-20/21/22` のみで、すべて実装・テスト済み。
- A の全要件はテストで決定論的に traceable（strict 100%）。
- **未対応の抜けは無し**。
