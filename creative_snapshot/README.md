Creative GDN Snapshot
=====================

This directory vendors the minimal creative Qwen3.5 GDN files needed by the
precision tests, so the test repo can run without a separate creative checkout.

Source:

- repo path: /workspace/qwen3.5_omni_creative
- git commit: 76940c4e9caf90f41ab6b634d7685ed5d6a5a9d2
- copied paths:
  - mindspeed_mm/fsdp/models/qwen3_5/chunk_gated_delta_rule.py
  - mindspeed_mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py
  - mindspeed_mm/fsdp/models/qwen3_5/triton/*.py

The tests load these files by path through a synthetic package. They do not
import the top-level mindspeed_mm package.
