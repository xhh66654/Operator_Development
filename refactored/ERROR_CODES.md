# 错误码说明（refactored）

本文档汇总本包内使用的**数值错误码**及其常见含义。算子与管道抛出的 `OperatorException` 会带 `code` 字段（对应下表）；部分 HTTP 入口（如 `service.py`）也会使用额外数值。

---

## 一、`ErrorCode` 枚举（`refactored/core/exceptions.py`）

| 数值 | 枚举名 | taxonomy 键名（`error_contract.py`） | 含义与常见原因 |
|------|--------|--------------------------------------|----------------|
| **1001** | `DATA_NOT_FOUND` | `data_not_found` | **数据/引用取不到**：如 `${step}` 在上下文中不存在、字段缺失、`get_value` 严格模式下引用解析为 `None`、比较/统计时待取值为空等。 |
| **1002** | `CONFIG_MISSING` | `config_missing` | **配置缺项**：必填参数未提供（如缺少 `first_value`、ES 缺少 `es_hosts`/`index`、文件算子缺少 `file_path` 等）。 |
| **1003** | `CONFIG_TYPE_ERROR` | `config_type_error` | **JSON Schema 配置校验失败**：类型不符、`required` 缺失、`additionalProperties` 不允许的键等（见 `config_schema.validate_operator_config`；**未安装 `jsonschema` 时该校验会跳过**，生产请按 `refactored/requirements.txt` 安装）。 |
| **1004** | `CONFIG_INVALID` | `config_invalid` | **配置不合法**：DAG 协议校验失败、`execution_mode` 不对、`reasoningDataList` 结构非法、业务规则不允许的配置组合等。 |
| **1005** | `DEPENDENCY_ERROR` | `dependency_error` | **依赖不满足**：前置条件未满足（如应先连接 ES 再提取、缺少依赖算子输出等，具体见各算子报错文案）。 |
| **1006** | `VERSION_UNSUPPORTED` | `version_unsupported` | **版本不支持**：请求或功能版本不被当前实现支持（若业务启用）。 |
| **1007** | `DUPLICATE_RUN_ID` | `duplicate_run_id` | **重复 runId**：同一 `runId` 已在执行中，被生命周期层拒绝重复提交（见 `run_lifecycle`）。 |
| **1008** | `CONFIG_FORMAT_ERROR` | `config_format_error` | **非法旧输入键或部分算子不支持的配置组合**：如配置中出现 `field`/`source`/`vectors` 等已弃用的输入键（见 `base_operator._reject_legacy_input_keys`）；部分算术算子在执行阶段对参数组合的限制也使用此码（见各算子实现）。 |
| **2001** | `TYPE_ERROR` | `type_error` | **类型错误**：值不是期望类型（如应为数值却是非数字字符串、矩阵维度不符等）。 |
| **2002** | `FORMAT_ERROR` | `format_error` | **格式错误**：字符串无法按预期解析为数字/日期等格式。 |
| **2003** | `SCHEMA_MISMATCH` | `schema_mismatch` | **结构/长度不匹配**：如加权算子权重数量与样本数量不一致、向量长度不一致等。 |
| **3001** | `CALC_LOGIC_ERROR` | `calc_logic_error` | **计算逻辑错误**：数学上不允许（如除零、分母为 0、几何/调和平均含 0）、业务规则禁止的计算路径等。 |
| **3002** | `OUT_OF_RANGE` | `out_of_range` | **越界**：值超出允许区间（若算子使用）。 |
| **4001** | `OOM` | `oom` | **内存不足**：显式检测到的内存压力或过大对象（若启用相关检查）。 |
| **4002** | `TIMEOUT` | `timeout` | **超时**：操作超过时限（若算子或服务层设置超时）。 |
| **4003** | `RESOURCE_LIMIT_EXCEEDED` | `resource_limit_exceeded` | **资源上限**：如 ES `max_hits`、批量大小等超过配置上限。 |
| **5001** | `RUNTIME_ERROR` | `runtime_error` | **运行时错误**：未归类异常、依赖库缺失（如 `pip install pandas`）、未预期的 Python 异常经包装后等。 |
| **5002** | `EXTERNAL_SERVICE_ERROR` | `external_service_error` | **外部服务错误**：如 ES 连接/查询失败、HTTP 外部接口失败等。 |
| **5999** | `UNKNOWN` | `unknown_error` | **未知**：未指定更细错误码时的默认值。 |

### 说明

- **1003 与 1008**：`CONFIG_TYPE_ERROR`（1003）专指 **jsonschema 层**校验失败；`CONFIG_FORMAT_ERROR`（1008）专指 **非法旧键**及部分算子对配置组合的拒绝，二者数值不同，调用方可区分。
- **taxonomy**：`refactored/core/error_contract.py` 中的 `ERROR_CODE_TAXONOMY` 为上述码及常用码提供英文 slug。

---

## 二、枚举外的项目内用法

| 数值 | 出现位置 | 含义与常见原因 |
|------|----------|----------------|
| **1000** | `refactored/service.py`（如请求体为空） | **HTTP 入口层**自定义码：表示请求体缺失等，**不属于** `ErrorCode` 枚举成员；与 `error_payload(..., error_code=1000)` 配合使用。 |

---

## 三、与回调/响应的关系（简要）

- 算子与 `execute_tree_calculation` 失败时，内部仍使用上述 `ErrorCode`；具体是否出现在对 Java 的 **callback** JSON 里，取决于当前实现（例如异步失败回调可能只带 `message` 字符串，而不单独带 `error_code` 字段）。
- 查问题时：**优先看异常/日志中的 `message` 全文**；需要程序化分类时再使用数值码或 `ERROR_CODE_TAXONOMY` 中的键名。

---

## 四、维护说明

- 新增或调整错误码时，请同步修改：
  - `refactored/core/exceptions.py`（`ErrorCode`）
  - `refactored/core/error_contract.py`（`ERROR_CODE_TAXONOMY`，如有新码）
  - 本文档 `refactored/ERROR_CODES.md`
