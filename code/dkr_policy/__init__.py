"""Walker package: the duration-guided KG retrieval used by the current method.

Modules
  kg_walker.py     BFS walk over Neo4j SNOMED; score = cos + λ·bc − μ·hop
  bc_llm_direct.py temporal score from LLM-judged log-normal durations
                   (WALKER_BC_MODE: `overlap` = Bhattacharyya, our continuous method;
                    `interval_sample` = point/interval baseline)
  bc_ondemand.py   generates a duration via the LLM on a cache miss, then persists it
  tui_roles.py     TUI → role (disease / finding / organism / anatomy) for the bc gate

Imports are flat (the package dir is placed on sys.path at runtime), so this file
intentionally imports nothing; it only marks the directory as a package.

Removed (superseded, originals preserved in scripts/dkr_policy/):
  types / run / stage4_rank / stage5_package — the legacy DKR-Policy v5 5-stage framework,
    formerly re-exported here, which made every walker run import the whole cluster.
  retrieve_simple / kde_bins / coverage_lookup — the older retrieval path plus its
    KDE-histogram temporal math and tiered duration source. Both are superseded by
    LLM-generated durations scored with bc_llm_direct. They survived only because
    kg_walker carried an unused `from retrieve_simple import _node_emb` import.
"""
