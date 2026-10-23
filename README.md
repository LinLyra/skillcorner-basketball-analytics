# SkillCorner Basketball Analytics Lab

独立的 SkillCorner 篮球追踪数据研究项目。它不依赖、不导入、也不修改 LYKON 的 IMU/UWB 数据管道。

## 当前范围

当前主线是 **Beyond the Snapshot**：检验投篮前动态防守信息，相比出手瞬间的静态防守距离，是否提供稳定且可信的跨比赛增量价值。

已完成的研究基础设施：

1. 读取 `.jsonl` / `.jsonl.gz` tracking；
2. 将嵌套球员和篮球对象展开成长表；
3. 将英尺统一转换为米，同时保留原始质量字段；
4. 读取 Dynamic Events bundle；
5. 检查空帧、检测覆盖率、预测误差和每帧实体数量；
6. 按事件帧提取 tracking episode；
7. 提供 separation、closing speed 等无业务假设的几何原语。

暂不包含：LYKON、IMU、UWB、复杂 GNN、EPV，以及未经验证的综合 Pressure Score。

## 首轮完整实验

固定版本数据已经覆盖全部 10 场比赛。首轮结果包含 1,459 次投篮，严格筛选后的主分析队列为 1,211 次非犯规且事件/坐标距离对齐的投篮。

```bash
PYTHONPATH=src python scripts/run_study.py
PYTHONPATH=src python scripts/run_robustness.py
PYTHONPATH=src python scripts/run_subgroup_analysis.py
PYTHONPATH=src python scripts/run_smoothing_sensitivity.py
PYTHONPATH=src python scripts/run_spatial_geometry.py
PYTHONPATH=src python scripts/run_official_benchmark.py
MPLCONFIGDIR=/tmp/sc-mpl PYTHONPATH=src python scripts/run_negative_control.py
PYTHONPATH=src python scripts/render_results.py
PYTHONPATH=src python scripts/render_robustness.py
PYTHONPATH=src python scripts/render_disagreement_cases.py
PYTHONPATH=src python scripts/render_final_summary.py
MPLCONFIGDIR=/tmp/sc-mpl PYTHONPATH=src python scripts/animate_disagreement.py
```

初步发现：1 秒动态窗口相对静态防守基线只有很小、跨比赛不稳定的提升，2 秒窗口平均表现更差。完整说明见 [reports/preliminary_results.md](reports/preliminary_results.md)，预先固定的评估规则见 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)。

鲁棒性结果与模型分歧案例见 [reports/robustness_results.md](reports/robustness_results.md)。
篮球情境分层与平滑敏感性见 [reports/basketball_context_results.md](reports/basketball_context_results.md)。

最终研究稿见 [reports/final_report.md](reports/final_report.md)，比赛提交摘要与一分钟讲稿见 [reports/analytics_cup_submission.md](reports/analytics_cup_submission.md)。

完整重现实验与发布校验：

```bash
bash scripts/reproduce_all.sh
```

## 数据来源

数据不提交到本仓库。官方源和研究参考版本固定在 `sources.lock.json`，首场数据文件固定在 `datasets.lock.json`。

- <https://github.com/SkillCorner/opendata-basketball>

获取并校验首场真实数据：

```bash
python scripts/fetch_official_game.py --game-id 114243
```

只验证已经下载的数据：

```bash
python scripts/fetch_official_game.py --game-id 114243 --verify-only
```

文件保存在：

```text
data/raw/<game_id>/
  <game_id>_game_data.json
  <game_id>_dynamic_events.json
  <game_id>_tracking_data.jsonl.gz
```

## 安装与验证

```bash
cd skillcorner-basketball-analytics
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

检查一场比赛：

```bash
sc-basketball inspect \
  data/raw/114243/114243_tracking_data.jsonl.gz \
  --game-id 114243 \
  --output outputs/114243_quality.json
```

展开并保存标准表：

```bash
sc-basketball normalize \
  data/raw/114243/114243_tracking_data.jsonl.gz \
  --game-id 114243 \
  --output-dir data/interim/114243
```

## 数据产物

`frames.csv`：一帧一行。

```text
game_id, frame_idx, wall_clock_ms, period, game_clock_s,
shot_clock_s, game_clock_stopped
```

`entities.csv`：每帧每个实体一行。

```text
game_id, frame_idx, wall_clock_ms, period, entity_type,
team_side, player_id, jersey, x_m, y_m, z_m, speed_mps,
is_detected, pred_error_m
```

## 研究里程碑

- M0：数据合同、解析、质量报告和测试；
- M1：球场回放与事件对齐；
- M2：投篮前窗口及防守距离复算；
- M3：1v1 separation、closing speed、defender recovery；
- M4：可解释 Shot Quality baseline；
- M5：反事实空间分析和交互式展示。

详细范围见 [docs/research_plan.md](docs/research_plan.md)。

## 外部参考

外部仓库只读存放在被 Git 忽略的 `external/`，不会进入核心包。选择依据、审阅重点和风险见 [docs/reference_review.md](docs/reference_review.md)。
