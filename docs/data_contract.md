# Canonical data contract

本项目的标准表只服务于 SkillCorner 篮球研究，不与任何外部硬件 schema 合并。

## Frames

一行代表比赛视频的一帧。`frame_idx` 是事件关联的首选键；`wall_clock_ms` 用于交叉检查。

必需字段：

- `game_id`
- `frame_idx`
- `wall_clock_ms`
- `period`
- `game_clock_s`
- `game_clock_stopped`
- `shot_clock_s`

## Entities

一行代表某一帧中的一个球员或篮球。

- 所有位置转为米；
- 所有速度转为米每秒；
- `is_detected=False` 表示位置是外推值，不代表缺失；
- `pred_error_m` 必须跟随位置进入后续分析；
- 死球帧允许没有任何实体；
- 不假设每个有效帧必定包含10名球员和1个篮球。

## Dynamic events

Dynamic Events JSON 是以事件族为键、事件数组为值的 bundle。读取后保持一族一表，不提前做宽表拼接。

事件与 tracking 优先通过以下字段关联：

1. `frame` / `startFrame` / `endFrame`；
2. `wallClock` / `startWallClock` / `endWallClock`；
3. `period + gameClock` 只用于检查，不作为唯一键。

## 单位

- 官方坐标：feet；标准表：metres；
- 官方速度：feet/second；标准表：metres/second；
- `wallClock`：milliseconds；
- `gameClock`、`shotClock`：seconds。

