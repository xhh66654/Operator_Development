# 指标计算服务 — HTTP 接口文档

本文档描述 `refactored.service`（Flask）对外暴露的 HTTP 接口，供 Java 或其它调用方集成。路径、回调基址等默认值见 `app_config.py`，可用环境变量覆盖。

---

## 1. 服务地址与约定

| 项 | 说明 |
|----|------|
| 默认监听 | `CALC_LISTEN_HOST`（默认 `0.0.0.0`）+ `CALC_LISTEN_PORT`（默认 `19080`） |
| 字符编码 | 请求/响应 JSON 使用 UTF-8 |
| CORS | 已启用 `flask_cors` |

**根路径自检**：`GET /` 返回服务状态及当前注册的 POST 路径列表（便于探活）。

---

## 2. 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 健康检查与端点列表 |
| `POST` | `/system/result` | **体系计算**：接收 DAG 请求体并计算 |
| `POST` | `/evaluation/result` | **结果修改计算**：请求体形态与体系侧一致时复用同一执行管线；与体系的差异主要体现在异步模式下 **结果回调 URL** 不同（见 §5） |

路径常量：`API_RECEIVE_PATH_SYSTEM_RESULT`、`API_RECEIVE_PATH_EVALUATION_RESULT`（默认即上表路径）。

---

## 3. 请求体契约（两 POST 共用）

### 3.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `taskId` | 任意 JSON | 建议填 | 透传至响应与回调 |
| `runId` | 任意 JSON | 可选 | 与 `run_id` 二选一等价；用于并发去重（见 §6） |
| `systemId` | 任意 JSON | 建议填 | 透传 |
| `execution_mode` | 字符串 | **是** | 必须为 **`"dag"`**（大小写不敏感，会规范为小写比较） |
| `steps` | 数组 | **是** | 非空；根级步骤树，嵌套结构见实现与 `SYSTEM_EVALUATION_API_IMPLEMENTATION_SPEC.md` |

### 3.2 协议约束（校验失败时 HTTP 200 内业务失败或入口错误）

- 请求体必须是 **JSON 对象**。
- **禁止** 同时携带非空的顶层 **`reasoningDataList`**；新协议仅使用顶层 **`steps`**。
- `execution_mode` 不为 `dag`、`steps` 非数组或为空、或误带 `reasoningDataList` 时，计算层返回 `success: false` 及 `error_code`（通常为 `1004` `CONFIG_INVALID`）等，形态见 §4。

内部会将顶层 `steps` 包装为单条 `reasoningDataList` 后送入既有 DAG 引擎；调用方无需再传 `reasoningDataList`。

### 3.3 `result` 字段与响应裁剪

- 若希望某节点在响应中出现 **`result`**，则请求中该节点上须带有 **`result` 键**（值可为占位对象等）。
- 请求中 **没有** `result` 键的节点：响应中 **不出现** `result`（也不会出现 `result: null`）。

**`result` 的数值从哪来（实现约定，勿用 `config` 有无推断）**

1. **优先**：该节点在执行上下文中是否有输出（按节点的 `id` / `node_id` / `indicatorId` 写入）。
2. **兜底**：上下文仍无值、但请求已声明 `result` 且该节点下已有可回传的子片段时，使用子树回包规则（例如末个子片段的 `result`）。无 `operator_key` 的容器在执行阶段也可能已被写入与最后一子一致的值，与「本层产出」语义一致。
3. **顺序算子链**：若请求在**非最后一个**子 step 上也带了 `result`，回包会保留整条链上各步，而不会折叠成仅最后一步，以便每个声明了 `result` 的节点都能在 JSON 中出现。

调用方应容忍 **`result` 类型随算子变化**（标量、数组、对象等）；父子节点可能出现**相同标量**（容器与末子传播），属预期行为。

---

## 4. 响应形态

### 4.1 请求体非法（非 JSON 或空体）

- **HTTP 400**
- 示例：

```json
{
  "success": false,
  "taskId": null,
  "runId": null,
  "systemId": null,
  "error_code": 1000,
  "message": "请求体不能为空或不是合法 JSON",
  "steps": []
}
```

`1000` 为 **HTTP 入口层**专用码，**不属于** `ErrorCode` 枚举；详见 `ERROR_CODES.md`。

### 4.2 同步模式（`CALCULATE_SYNC`）

环境变量 `CALCULATE_SYNC` 为 `1` / `true` / `yes` / `on`（不区分大小写）时：

- 本次 POST **在响应中直接返回完整计算结果**（成功或业务失败），**不会**再向 Java 的 8091 等地址发起结果回调。
- **HTTP 200**： body 为完整业务 JSON（见下）。

**成功**（业务层）示例结构：

```json
{
  "success": true,
  "taskId": "与请求一致",
  "runId": "与请求一致",
  "systemId": "与请求一致",
  "execution_mode": "dag",
  "steps": []
}
```

**失败**（校验或计算失败等）：`success: false`，含 `taskId` / `runId` / `systemId`、`error_code`、`message`，`steps` 多为 `[]`。错误码含义见 `refactored/ERROR_CODES.md` 与 `core/exceptions.py` 中 `ErrorCode`。

### 4.3 异步模式（默认，未开启 `CALCULATE_SYNC`）

入队成功后 **立即** 返回 **受理确认**（**HTTP 200**），**完整结果** 在后台计算完成后通过 **POST** 发往 Java 侧配置的结果回调 URL（见 §5）。

```json
{
  "taskId": "与请求一致",
  "runId": "与请求一致",
  "systemId": "与请求一致",
  "status": "success"
}
```

注意：此处的 `"status": "success"` 表示 **请求已被接受并入队**，**不**表示 DAG 计算已成功完成。

### 4.4 `runId` 重复执行（去重）

若请求携带有效 `runId`（或 `run_id`），且该 `runId` **已在执行中**（与异步队列/同步计算的生命周期配合）：

- **不再次执行计算**；
- **HTTP 200**，body 与异步受理形态相同（`taskId`、`runId`、`systemId`、`status: "success"`），无完整 `steps` 结果。

### 4.5 未捕获异常（服务内部）

- **HTTP 500**
- body：`success: false`，`error_code` 为 `5001`（`RUNTIME_ERROR`），`message` 为异常字符串，`steps: []`。

### 4.6 JSON 数值

响应序列化时，**NaN / Infinity** 浮点数会被清理为 **`null`**，以保证合法 JSON。

---

## 5. 异步：Python → Java 回调

默认配置下（见 `app_config.py`，可用环境变量覆盖 host/port/path 或完整 URL）：

| 业务 | 结果回调 URL（POST 完整计算 JSON） |
|------|-----------------------------------|
| 体系 | `JAVA_RESULT_CALLBACK_URL_SYSTEM`（默认 `http://127.0.0.1:8091/system/result`） |
| 结果修改 | `JAVA_RESULT_CALLBACK_URL_EVALUATION`（默认 `http://127.0.0.1:8091/evaluation/result`） |

**状态回调**（计算开始排队后，在后台任务内发送）：由结果 URL 派生——若路径以 `result` 结尾，则将末尾 `result` 替换为 `status`（例如 `…/system/result` → `…/system/status`）。若结果 URL 不以 `result` 结尾，则**不发送**状态回调。

状态回调 body 大致包含：`taskId`、`runId`、`systemId`、`status`（`success` / `failed`）；失败时可能带 `message`。

结果回调 body 与 **§4.2** 中同步模式返回的成功/失败 JSON **同构**（成功含 `execution_mode` 与裁剪后的 `steps`）。

回调重试次数：`CALLBACK_RETRIES`（默认 `3`）。

---

## 6. 可选行为与环境变量摘要

| 变量 | 作用 |
|------|------|
| `CALC_LISTEN_HOST` / `CALC_LISTEN_PORT` | Flask 监听地址（须与 `INTERFACE_URL_PYTHON_*` 中的主机、端口一致） |
| `CALCULATE_SYNC` | 置为真值则同步返回全量结果，不调 8091 结果回调 |
| `INTERFACE_URL_PYTHON_SYSTEM` / `INTERFACE_URL_PYTHON_EVALUATION` | Java → Python 两条入站完整 URL |
| `INTERFACE_URL_JAVA_SYSTEM_RESULT` / `INTERFACE_URL_JAVA_EVALUATION_RESULT` | Python → Java 两条结果回调完整 URL（与上两条一一对应）。兼容旧环境变量 `JAVA_RESULT_CALLBACK_URL_SYSTEM` / `JAVA_RESULT_CALLBACK_URL_EVALUATION` |
| （派生）状态回调 | 仍由各结果 URL 将路径末尾 `result` 换为 `status` |
| `CALC_ACK_SAVE_DIR` | 若设置目录，收到合法 JSON 后会写入 `received_{taskId}_{runId}.json` 确认文件 |
| `CALC_HEARTBEAT_SEC` | 长耗时计算期间 INFO 心跳间隔（秒），`0` 关闭 |
| `CALC_WORKER_THREADS` | 异步线程池大小（默认 `8`） |
| `CALC_FLOAT_ROUND_DECIMALS` | 回包浮点保留小数位等（见 `tree_calculation`） |

---

## 7. 相关文档与代码

| 文档/模块 | 内容 |
|-----------|------|
| `SYSTEM_EVALUATION_API_IMPLEMENTATION_SPEC.md` | 协议背景、路径约定、字段草案与演进说明 |
| `ERROR_CODES.md` | 错误码表 |
| `app_config.py` | 路径与 ES、回调基址 |
| `service.py` | HTTP 路由与同步/异步分支 |
| `pipeline/system_protocol.py` | 顶层 `steps` 校验与执行入口 |

---

## 8. 修订说明

本文档以仓库内当前实现为准；若实现变更，请同步更新本文档与 `SYSTEM_EVALUATION_API_IMPLEMENTATION_SPEC.md`。
