# Xunce 中期缩减规模双门槛运行手册

G1、G2、G3 是当前中期交付链，使用 `configs/xunce_mid_dual_*.json`、对应 `scripts/run_xunce_mid_dual_*.py` 与 `tests/test_xunce_mid_dual_*.py`。

- G1：覆盖与修复，入口 `scripts/run_xunce_mid_dual_g1_coverage.py`。
- G2：规划时间与独立真值输入，入口 `scripts/run_xunce_mid_dual_g2_planning_time.py`。
- G3：闭环合同，入口 `scripts/run_xunce_mid_dual_g3_closed_loop.py`。

运行时 artifact 写入 `D:/xunce/out/<stage_short>`，并使用 `config.json`、`summary.json`、`routing.json`、`manifest.json`、`phase-state.jsonl` 与 `report.md` 等短 canonical 名。先运行相应 pytest 合同测试；本手册不授权完整 G 实验、训练、checkpoint 发布、executor 连接或 canary。
