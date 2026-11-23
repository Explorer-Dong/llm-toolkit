# WebShop Abstain

Source: [WebShop](https://github.com/princeton-nlp/WebShop). See `LICENSE.md`.

## Files

- `webshop_abstain.json`: 1,000 evaluation instructions.
- `goal_indices.json`: source IDs mapped to the 12,087 original goal indices.
- `removed_product_asins.json`: 201 ASINs excluded from the 251 missing-target tasks.
- `text_environment/`: WebShop text simulator and reward logic.

## Catalog Setup

Requires `uv sync --extra webshop_abstain` and Java 21. Place catalog assets under `catalogs/` (default `--catalogs-dir`, ignored by Git):

```text
catalogs/
├── product_attributes.json
├── human_instructions.json
├── full/
│   ├── products.json
│   ├── search_index/
│   └── catalog.sqlite3        # auto-generated on first run
└── missing_target/
    ├── products.json
    ├── search_index/
    └── catalog.sqlite3        # auto-generated on first run
```

1. Download the full dataset from [WebShop setup](https://github.com/princeton-nlp/WebShop#-setup).
2. Rename upstream files:
   - `items_shuffle.json` -> `full/products.json`
   - `items_ins_v2.json` -> `product_attributes.json`
   - `items_human_ins.json` -> `human_instructions.json`
3. Create `missing_target/products.json` by removing the ASINs listed in `removed_product_asins.json` from `full/products.json` (bootstrap goal index `0`).
4. Build `full/search_index/` and `missing_target/search_index/` with Pyserini 0.44.0 and Java 21.
5. On first run, `WebShopCatalog` builds `catalog.sqlite3` in each variant directory from `products.json`, `product_attributes.json`, and `human_instructions.json`, and reuses it in read-only mode on subsequent runs.

## Usage

```bash
uv run --extra webshop_abstain rollout webshop_abstain --dataset-indices 0 500 749
uv run scoring webshop_abstain outputs/webshop_abstain/<run>
```
