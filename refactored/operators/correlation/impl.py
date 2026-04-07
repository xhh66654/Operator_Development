"""
相关性与误差指标算子：
  covariance           — 协方差
  pearson_correlation  — 皮尔逊相关系数
  spearman_correlation — 斯皮尔曼相关系数
  r_squared            — 决定系数 R²
  residual             — 残差序列
  mse                  — 均方误差
  rmse                 — 均方根误差
  mae                  — 平均绝对误差
  mape                 — 平均绝对百分比误差、
"""
import math
import statistics
from typing import Any, List, Optional, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
from ...core.exceptions import OperatorException, ErrorCode
from ...core import BaseOperator, ExecutionContext, OperatorRegistry
from .._common import get_value, to_number
from ..statistics.stats import _ensure_number_list


def _coerce_corr_sequence(raw: Any, *, role: str, operator: str, config: dict) -> List[float]:
    """将一侧取值规范为浮点序列，保持原有顺序（展平嵌套数值列表，禁止静默丢元素）。"""
    if raw is None:
        return []
    if isinstance(raw, tuple):
        raw = list(raw)
    if not isinstance(raw, list):
        return [float(to_number(raw))]
    return _ensure_number_list(raw)


def _to_lists(
    data,
    config,
    context,
    *,
    operator: str,
) -> Tuple[List[float], List[float]]:
    """
    从 config 的 x/y 或 first_value/second_value 取两组数值，**等长逐对**对应。
    不再使用 min 截断，避免静默错配；长度不一致时抛错。
    """
    x_ref = config.get("x") if config.get("x") not in (None, "") else config.get("first_value")
    y_ref = config.get("y") if config.get("y") not in (None, "") else config.get("second_value")
    if x_ref in (None, "") or y_ref in (None, ""):
        raise OperatorException(
            f"{operator} 需要配置 x 与 y（或 first_value 与 second_value）",
            code=ErrorCode.CONFIG_MISSING,
            operator=operator,
            config=config,
        )
    x_val = get_value(data, x_ref, context)
    y_val = get_value(data, y_ref, context)
    if x_val is not None and x_val is y_val:
        raise OperatorException(
            f"{operator} 两侧数据来源为同一引用，无法配对（请检查 x/y 或 first/second 是否指向同一字段）",
            code=ErrorCode.CONFIG_INVALID,
            operator=operator,
            config=config,
        )
    x_vals = _coerce_corr_sequence(x_val, role="x/first_value", operator=operator, config=config)
    y_vals = _coerce_corr_sequence(y_val, role="y/second_value", operator=operator, config=config)
    if len(x_vals) != len(y_vals):
        raise OperatorException(
            f"{operator} 要求两组数据长度一致以便按位配对，当前 len(x)={len(x_vals)}、len(y)={len(y_vals)}",
            code=ErrorCode.SCHEMA_MISMATCH,
            operator=operator,
            config=config,
        )
    return x_vals, y_vals


def _is_vector_row_bundle(x: Any) -> bool:
    """外层为向量组：每项为一条向量（list，可含嵌套数值，交由 _to_numeric_vector 展平）。"""
    return isinstance(x, list) and len(x) > 0 and all(isinstance(row, list) for row in x)


def _try_vector_bundle_pairs(
    data: dict,
    config: dict,
    context: ExecutionContext,
    *,
    operator: str,
) -> Any:
    """
    若 x/y（或 first/second）一侧或两侧为二维向量组，则返回 (a_raw, b_raw) 的列表供逐对计算；
    否则返回 None，调用方应走 _to_lists 一维逐元素配对。
    """
    x_ref = config.get("x") if config.get("x") not in (None, "") else config.get("first_value")
    y_ref = config.get("y") if config.get("y") not in (None, "") else config.get("second_value")
    if x_ref in (None, "") or y_ref in (None, ""):
        raise OperatorException(
            f"{operator} 需要配置 x 与 y（或 first_value 与 second_value）",
            code=ErrorCode.CONFIG_MISSING,
            operator=operator,
            config=config,
        )
    x_raw = get_value(data, x_ref, context)
    y_raw = get_value(data, y_ref, context)
    if x_raw is not None and x_raw is y_raw:
        raise OperatorException(
            f"{operator} 两侧数据来源为同一引用，无法配对",
            code=ErrorCode.CONFIG_INVALID,
            operator=operator,
            config=config,
        )
    vb_x = _is_vector_row_bundle(x_raw)
    vb_y = _is_vector_row_bundle(y_raw)
    if not vb_x and not vb_y:
        return None

    from ..arithmetic.impl import _to_numeric_vector

    if vb_x and vb_y:
        if len(x_raw) != len(y_raw):
            raise OperatorException(
                f"{operator} 两侧为向量组时外层长度须一致: {len(x_raw)} != {len(y_raw)}",
                code=ErrorCode.SCHEMA_MISMATCH,
                operator=operator,
                config=config,
            )
        return list(zip(x_raw, y_raw))
    if vb_x:
        base_y = _to_numeric_vector(y_raw)
        if base_y is None:
            raise OperatorException(
                f"{operator}：y/second_value 无法解析为与向量组配对的数值向量",
                code=ErrorCode.TYPE_ERROR,
                operator=operator,
                config=config,
            )
        return [(a, base_y) for a in x_raw]
    base_x = _to_numeric_vector(x_raw)
    if base_x is None:
        raise OperatorException(
            f"{operator}：x/first_value 无法解析为与向量组配对的数值向量",
            code=ErrorCode.TYPE_ERROR,
            operator=operator,
            config=config,
        )
    return [(base_x, b) for b in y_raw]


def _euclidean_pair_numeric(aa: List[float], bb: List[float], *, operator: str, config: dict) -> float:
    if len(aa) != len(bb):
        raise OperatorException(
            f"{operator} 各向量维度须一致，当前: {[len(aa), len(bb)]}",
            code=ErrorCode.CONFIG_INVALID,
            operator=operator,
            config=config,
        )
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(aa, bb)))


def _angle_rad_pair_numeric(
    aa: List[float],
    bb: List[float],
    *,
    operator: str,
    config: dict,
    zero_vector_result: Optional[float] = None,
) -> float:
    if len(aa) != len(bb):
        raise OperatorException(
            f"{operator} 各向量维度须一致，当前: {[len(aa), len(bb)]}",
            code=ErrorCode.CONFIG_INVALID,
            operator=operator,
            config=config,
        )
    dot = sum(x * y for x, y in zip(aa, bb))
    norm_x = math.sqrt(sum(x * x for x in aa))
    norm_y = math.sqrt(sum(y * y for y in bb))
    if norm_x == 0 or norm_y == 0:
        if zero_vector_result is not None:
            return float(zero_vector_result)
        raise OperatorException(
            f"{operator} 存在零向量，无法计算夹角",
            code=ErrorCode.CALC_LOGIC_ERROR,
            operator=operator,
            config=config,
        )
    cos_angle = dot / (norm_x * norm_y)
    cos_angle = max(-1, min(1, cos_angle))
    return math.acos(cos_angle)


def _obs_pred(data, config, context):
    """从 config 的 observed/predicted 取出两个数值列表，最短对齐。"""
    obs_ref = config.get("observed") if config.get("observed") not in (None, "") else config.get("first_value")
    pred_ref = config.get("predicted") if config.get("predicted") not in (None, "") else config.get("second_value")
    obs = get_value(data, obs_ref, context)
    pred = get_value(data, pred_ref, context)
    if not isinstance(obs, list):
        obs = [obs] if obs is not None else []
    if not isinstance(pred, list):
        pred = [pred] if pred is not None else []
    n = min(len(obs), len(pred))
    return [to_number(x) for x in obs[:n]], [to_number(x) for x in pred[:n]]


def _rank_data(values: List[float]) -> List[float]:
    """返回每个值在排序后的平均排名（处理并列）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks



@OperatorRegistry.register("covariance")
class CovarianceOperator(BaseOperator):
    """
    协方差：衡量两组变量共同变化的程度。
    Cov(X,Y) = Σ(xi - x̄)(yi - ȳ) / (n-1)
    正值→同向变化，负值→反向变化，0→无线性关系。
    """
    name = "covariance"
    config_schema = {"type": "object", "properties": {"x": {}, "y": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        x_vals, y_vals = _to_lists(data, config, context, operator=self.name)
        n = len(x_vals)
        if n < 2:
            return 0.0
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(y_vals)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / (n - 1)
        return cov



@OperatorRegistry.register("pearson_correlation")
class PearsonCorrelationOperator(BaseOperator):
    """
    皮尔逊相关系数：衡量两变量线性相关程度，值域 [-1, 1]。
    ±1 表示完全线性相关，0 表示无线性关系。
    """
    name = "pearson_correlation"
    config_schema = {"type": "object", "properties": {"x": {}, "y": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        x_vals, y_vals = _to_lists(data, config, context, operator=self.name)
        n = len(x_vals)
        if n < 2:
            return 0.0
        if _HAS_NUMPY:
            try:
                m = np.row_stack((np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)))
                corr = np.corrcoef(m)[0, 1]
                return 0.0 if math.isnan(float(corr)) else float(corr)
            except (ValueError, TypeError, IndexError):
                pass
        x_mean, y_mean = statistics.mean(x_vals), statistics.mean(y_vals)
        std_x, std_y = statistics.stdev(x_vals), statistics.stdev(y_vals)
        if std_x == 0 or std_y == 0:
            return 0.0
        corr = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / ((n - 1) * std_x * std_y)
        return corr


@OperatorRegistry.register("spearman_correlation")
class SpearmanCorrelationOperator(BaseOperator):
    """
    斯皮尔曼相关系数：基于秩次（排名）的相关系数，对异常值不敏感。
    值域 [-1, 1]，适用于非线性单调关系。
    """
    name = "spearman_correlation"
    config_schema = {"type": "object", "properties": {"x": {}, "y": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        x_vals, y_vals = _to_lists(data, config, context, operator=self.name)
        n = len(x_vals)
        if n < 2:
            return 0.0
        try:
            from scipy.stats import spearmanr

            r, _ = spearmanr(x_vals, y_vals)
            rf = float(r) if r is not None else float("nan")
            if math.isnan(rf):
                return 0.0
            return rf
        except ImportError:
            pass
        rx, ry = _rank_data(x_vals), _rank_data(y_vals)
        mean_rx, mean_ry = statistics.mean(rx), statistics.mean(ry)
        std_rx, std_ry = statistics.stdev(rx), statistics.stdev(ry)
        if std_rx == 0 or std_ry == 0:
            return 0.0
        corr = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry)) / ((n - 1) * std_rx * std_ry)
        return corr



@OperatorRegistry.register("r_squared")
class RSquaredOperator(BaseOperator):
    """
    决定系数 R²：模型解释的方差占总方差的比例。
    R² = 1 - SS_res/SS_tot，值域 [0,1]，越接近 1 说明模型拟合越好。
    """
    name = "r_squared"
    config_schema = {"type": "object", "properties": {"observed": {}, "predicted": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        obs, pred = _obs_pred(data, config, context)
        if not obs:
            return 0.0
        mean_obs = statistics.mean(obs)
        ss_tot = sum((o - mean_obs) ** 2 for o in obs)
        ss_res = sum((o - p) ** 2 for o, p in zip(obs, pred))
        return 0.0 if ss_tot == 0 else (1 - ss_res / ss_tot)



@OperatorRegistry.register("residual")
class ResidualOperator(BaseOperator):
    """残差序列：observed[i] - predicted[i]，返回残差列表。"""
    name = "residual"
    config_schema = {"type": "object", "properties": {"observed": {}, "predicted": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        obs, pred = _obs_pred(data, config, context)
        return [o - p for o, p in zip(obs, pred)]


@OperatorRegistry.register("mse")
class MseOperator(BaseOperator):
    """均方误差（MSE）：mean((obs - pred)²)，误差的平方均值。"""
    name = "mse"
    config_schema = {"type": "object", "properties": {"observed": {}, "predicted": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        obs, pred = _obs_pred(data, config, context)
        if not obs:
            return 0.0
        return sum((o - p) ** 2 for o, p in zip(obs, pred)) / len(obs)


@OperatorRegistry.register("rmse")
class RmseOperator(BaseOperator):
    """均方根误差（RMSE）：sqrt(MSE)，与原始数据同单位，比 MSE 更直观。"""
    name = "rmse"
    config_schema = {"type": "object", "properties": {"observed": {}, "predicted": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        obs, pred = _obs_pred(data, config, context)
        if not obs:
            return 0.0
        mse_val = sum((o - p) ** 2 for o, p in zip(obs, pred)) / len(obs)
        return math.sqrt(mse_val)


@OperatorRegistry.register("mae")
class MaeOperator(BaseOperator):
    """平均绝对误差（MAE）：mean(|obs - pred|)，对异常值不敏感。"""
    name = "mae"
    config_schema = {"type": "object", "properties": {"observed": {}, "predicted": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        obs, pred = _obs_pred(data, config, context)
        if not obs:
            return 0.0
        return sum(abs(o - p) for o, p in zip(obs, pred)) / len(obs)


@OperatorRegistry.register("mape")
class MapeOperator(BaseOperator):
    """平均绝对百分比误差（MAPE）：mean(|obs-pred|/|obs|)*100，百分比形式，obs≠0。"""
    name = "mape"
    config_schema = {"type": "object", "properties": {"observed": {}, "predicted": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        obs, pred = _obs_pred(data, config, context)
        valid = [(o, p) for o, p in zip(obs, pred) if o != 0]
        if not valid:
            return 0.0
        return sum(abs(o - p) / abs(o) for o, p in valid) / len(valid) * 100


@OperatorRegistry.register("euclidean_distance")
class EuclideanDistanceOperator(BaseOperator):
    """欧几里得距离：sqrt(Σ(xi - yi)²)。支持一维逐元素配对或二维向量组一一对应（同 cosine_similarity）。"""
    name = "euclidean_distance"
    config_schema = {"type": "object", "properties": {"x": {}, "y": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        pairs = _try_vector_bundle_pairs(data, config, context, operator=self.name)
        if pairs is not None:
            from ..arithmetic.impl import _to_numeric_vector

            out: List[float] = []
            for a, b in pairs:
                aa, bb = _to_numeric_vector(a), _to_numeric_vector(b)
                if aa is None or bb is None:
                    raise OperatorException(
                        f"{self.name} 向量数据无效或非数值",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(_euclidean_pair_numeric(aa, bb, operator=self.name, config=config))
            return out[0] if len(out) == 1 else out
        x_vals, y_vals = _to_lists(data, config, context, operator=self.name)
        if not x_vals:
            return 0.0
        return _euclidean_pair_numeric(x_vals, y_vals, operator=self.name, config=config)


@OperatorRegistry.register("angle_between")
class AngleBetweenOperator(BaseOperator):
    """实际夹角：点积与模长算夹角（弧度）。支持向量组一一对应，规则同 euclidean_distance。"""
    name = "angle_between"
    config_schema = {"type": "object", "properties": {"x": {}, "y": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        pairs = _try_vector_bundle_pairs(data, config, context, operator=self.name)
        if pairs is not None:
            from ..arithmetic.impl import _to_numeric_vector

            out: List[float] = []
            for a, b in pairs:
                aa, bb = _to_numeric_vector(a), _to_numeric_vector(b)
                if aa is None or bb is None:
                    raise OperatorException(
                        f"{self.name} 向量数据无效或非数值",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(_angle_rad_pair_numeric(aa, bb, operator=self.name, config=config))
            return out[0] if len(out) == 1 else out
        x_vals, y_vals = _to_lists(data, config, context, operator=self.name)
        if not x_vals:
            return 0.0
        return _angle_rad_pair_numeric(
            x_vals, y_vals, operator=self.name, config=config, zero_vector_result=0.0
        )


@OperatorRegistry.register("vector_angle")
class VectorAngleOperator(BaseOperator):
    """向量夹角（弧度），与 angle_between 计算一致；支持向量组一一对应。"""
    name = "vector_angle"
    config_schema = {"type": "object", "properties": {"x": {}, "y": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        pairs = _try_vector_bundle_pairs(data, config, context, operator=self.name)
        if pairs is not None:
            from ..arithmetic.impl import _to_numeric_vector

            out: List[float] = []
            for a, b in pairs:
                aa, bb = _to_numeric_vector(a), _to_numeric_vector(b)
                if aa is None or bb is None:
                    raise OperatorException(
                        f"{self.name} 向量数据无效或非数值",
                        code=ErrorCode.TYPE_ERROR,
                        operator=self.name,
                        config=config,
                    )
                out.append(_angle_rad_pair_numeric(aa, bb, operator=self.name, config=config))
            return out[0] if len(out) == 1 else out
        x_vals, y_vals = _to_lists(data, config, context, operator=self.name)
        if not x_vals:
            return 0.0
        return _angle_rad_pair_numeric(
            x_vals, y_vals, operator=self.name, config=config, zero_vector_result=0.0
        )


@OperatorRegistry.register("jaccard_distance")
class JaccardDistanceOperator(BaseOperator):
    """Jaccard 距离：1 - (|A∩B| / |A∪B|)"""
    name = "jaccard_distance"
    config_schema = {"type": "object", "properties": {"set_a": {}, "set_b": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        set_a_raw = get_value(data, config.get("set_a") if config.get("set_a") not in (None, "") else config.get("first_value"), context)
        set_b_raw = get_value(data, config.get("set_b") if config.get("set_b") not in (None, "") else config.get("second_value"), context)
        try:
            set_a = set(set_a_raw) if isinstance(set_a_raw, list) else ({set_a_raw} if set_a_raw is not None else set())
            set_b = set(set_b_raw) if isinstance(set_b_raw, list) else ({set_b_raw} if set_b_raw is not None else set())
        except TypeError:
            raise OperatorException(
                "jaccard_distance 的集合元素必须为可哈希类型",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        jaccard = intersection / union
        return 1 - jaccard


@OperatorRegistry.register("jaccard_coefficient")
class JaccardCoefficientOperator(BaseOperator):
    """Jaccard 系数：|A∩B| / |A∪B|，衡量集合相似度"""
    name = "jaccard_coefficient"
    config_schema = {"type": "object", "properties": {"set_a": {}, "set_b": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        set_a_raw = get_value(data, config.get("set_a") if config.get("set_a") not in (None, "") else config.get("first_value"), context)
        set_b_raw = get_value(data, config.get("set_b") if config.get("set_b") not in (None, "") else config.get("second_value"), context)
        try:
            set_a = set(set_a_raw) if isinstance(set_a_raw, list) else ({set_a_raw} if set_a_raw is not None else set())
            set_b = set(set_b_raw) if isinstance(set_b_raw, list) else ({set_b_raw} if set_b_raw is not None else set())
        except TypeError:
            raise OperatorException(
                "jaccard_coefficient 的集合元素必须为可哈希类型",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union


@OperatorRegistry.register("sorensen_dice")
class SorensenDiceOperator(BaseOperator):
    """Sørensen-Dice 系数：2 * |A∩B| / (|A| + |B|)，对重叠部分更敏感"""
    name = "sorensen_dice"
    config_schema = {"type": "object", "properties": {"set_a": {}, "set_b": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        set_a_raw = get_value(data, config.get("set_a") if config.get("set_a") not in (None, "") else config.get("first_value"), context)
        set_b_raw = get_value(data, config.get("set_b") if config.get("set_b") not in (None, "") else config.get("second_value"), context)
        try:
            set_a = set(set_a_raw) if isinstance(set_a_raw, list) else ({set_a_raw} if set_a_raw is not None else set())
            set_b = set(set_b_raw) if isinstance(set_b_raw, list) else ({set_b_raw} if set_b_raw is not None else set())
        except TypeError:
            raise OperatorException(
                "sorensen_dice 的集合元素必须为可哈希类型",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        intersection = len(set_a & set_b)
        total = len(set_a) + len(set_b)
        if total == 0:
            return 0.0
        return 2 * intersection / total


@OperatorRegistry.register("set_intersection")
class SetIntersectionOperator(BaseOperator):
    """交集：返回两个集合的交集元素列表"""
    name = "set_intersection"
    config_schema = {"type": "object", "properties": {"set_a": {}, "set_b": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        set_a_raw = get_value(data, config.get("set_a") if config.get("set_a") not in (None, "") else config.get("first_value"), context)
        set_b_raw = get_value(data, config.get("set_b") if config.get("set_b") not in (None, "") else config.get("second_value"), context)
        try:
            set_a = set(set_a_raw) if isinstance(set_a_raw, list) else ({set_a_raw} if set_a_raw is not None else set())
            set_b = set(set_b_raw) if isinstance(set_b_raw, list) else ({set_b_raw} if set_b_raw is not None else set())
        except TypeError:
            raise OperatorException(
                "set_intersection 的集合元素必须为可哈希类型",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        return list(set_a & set_b)


@OperatorRegistry.register("set_union")
class SetUnionOperator(BaseOperator):
    """并集：返回两个集合的并集元素列表"""
    name = "set_union"
    config_schema = {"type": "object", "properties": {"set_a": {}, "set_b": {}, "first_value": {}, "second_value": {}}}
    default_config = {}

    def execute(self, data, config, context: ExecutionContext):
        set_a_raw = get_value(data, config.get("set_a") if config.get("set_a") not in (None, "") else config.get("first_value"), context)
        set_b_raw = get_value(data, config.get("set_b") if config.get("set_b") not in (None, "") else config.get("second_value"), context)
        try:
            set_a = set(set_a_raw) if isinstance(set_a_raw, list) else ({set_a_raw} if set_a_raw is not None else set())
            set_b = set(set_b_raw) if isinstance(set_b_raw, list) else ({set_b_raw} if set_b_raw is not None else set())
        except TypeError:
            raise OperatorException(
                "set_union 的集合元素必须为可哈希类型",
                code=ErrorCode.TYPE_ERROR,
                operator=self.name,
                config=config,
            )
        return list(set_a | set_b)
