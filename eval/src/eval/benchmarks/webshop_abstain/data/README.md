# WebShop Abstain

- `webshop_abstain.json`: 1,000 evaluation instructions.
- `goal_indices.json`: source IDs mapped to the 12,087 original goal indices.
- `removed_product_asins.json`: 201 products removed for the 251 missing-target tasks.
- `text_environment/`: upstream text simulator and reward code, with local imports and quiet startup logs.

Source: [WebShop](https://github.com/princeton-nlp/WebShop). See `LICENSE.md`.

Install `uv sync --extra webshop_abstain` and Java 21. Supply `--catalogs-dir` with:

```text
catalogs/
├── product_attributes.json
├── human_instructions.json
├── full/
│   ├── products.json
│   └── search_index/
└── missing_target/
    ├── products.json
    └── search_index/
```

The default is this directory's `catalogs/`, excluded from Git.
Use the full resources from [WebShop setup](https://github.com/princeton-nlp/WebShop#-setup).
Rename `items_shuffle.json` to `full/products.json`, `items_ins_v2.json` to
`product_attributes.json`, and `items_human_ins.json` to `human_instructions.json`.
Build both indexes with Pyserini 0.44.0 and Java 21. The missing-target catalog removes
exactly `removed_product_asins.json`; its bootstrap goal index is 0.

```bash
uv run --extra webshop_abstain eval-inference webshop_abstain --dataset-indices 0 500 749
uv run eval-scoring webshop_abstain outputs/webshop_abstain/<run>
```
