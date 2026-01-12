import pandas as pd
import numpy as np
import pymannkendall as mk
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Literal, List

from utils.sql_executor import MySQLExecutor
oa_executor = MySQLExecutor('OA')

# Type aliases
TrendType = Literal['上涨', '下降', '暴涨', '暴跌', '无趋势', '计算错误']
TrendMethod = Literal['mk', 'ma', 'mix']


@dataclass
class TrendAnalysisConfig:
    """趋势分析配置参数"""
    short_term_days: int = 5
    very_short_term_days: int = 3
    window_period = 14


class SalesTrendAnalyzer:
    def __init__(self,
                 trend_method='mix',
                 extreme_std=3,
                 extrem_thres=10,
                 lr_slope_threshold=3,
                 lr_r2_score=0.4,
                 config: Optional[TrendAnalysisConfig] = None):

        self.config = config if config else TrendAnalysisConfig()
        self.extreme_std = extreme_std
        self.extrem_thres = extrem_thres
        self.trend_method = trend_method
        self.lr_slop_threshold = lr_slope_threshold
        self.lr_r2_score = lr_r2_score

    def extreme_detect(self, sales_data):
        '''
            检测暴涨/暴跌
            暴跌筛选条件：
                1. 最后一天比大多数天数都低
                2. 暴跌前一天不是暴涨
                3. 如果销量为0的天数大于7天, 不判断暴跌
        '''
        # 第一次检测，暴涨则直接返回结果
        initial_result = self.last_day_extreme_sigma_rule(sales_data)
        if initial_result != "暴跌":
            return initial_result

        # 暴跌条件1. 最后一天比大多数天数都低
        if ((sales_data - sales_data.iloc[-1]) > 0).sum() <= 7:
            return None

        # 暴跌条件2. 是否有7天为0
        if ((sales_data == 0).sum() > 6):
            return None

        # 第二次检测（去掉最后一个值），确保前一天不是暴涨
        second_result = self.last_day_extreme_sigma_rule(sales_data[:-1])
        if second_result != "暴涨":
            return initial_result  # 如果不是暴涨，直接返回

        # 第三次检测（去掉倒数第二个值后）
        modified_series = sales_data.drop(sales_data.index[-2])
        final_result = self.last_day_extreme_sigma_rule(modified_series)

        return final_result

    def last_day_extreme_sigma_rule(self, sales_data: pd.Series):
        """
            input: 单条item_id的近15天销售数据(pd.Series),
            output: ['暴涨', '暴跌', None]

            筛除条件：
                1. 如果销量为0的天数大于7天, 不判断暴跌
        """
        sales_data = sales_data.astype(float)

        # 计算日环比差值
        diffs = sales_data.diff().dropna()

        if len(diffs) < 1:
            return None

        # 替换极值（除最后一天，最大最小值替换为剩余值的平均值）
        extreme_mask = (diffs[:-1] == diffs[:-1].max()
                        ) | (diffs[:-1] == diffs[:-1].min())
        normal_diffs = diffs[:-1][~extreme_mask]
        avg_value = normal_diffs.mean() if len(normal_diffs) > 0 else 0
        processed_diffs = diffs[:-1].mask(extreme_mask, avg_value)

        # 计算统计量
        mean_diff = processed_diffs.mean()
        std_diff = processed_diffs.std()

        # 判断最后一天是否异常
        last_diff = diffs.iloc[-1]
        upper_bound = mean_diff + self.extreme_std * std_diff
        lower_bound = mean_diff - self.extreme_std * std_diff

        if last_diff > upper_bound and abs(last_diff) > self.extrem_thres:
            return '暴涨'
        elif last_diff < lower_bound and abs(last_diff) > self.extrem_thres:
            return '暴跌'
        else:
            return None

    @staticmethod
    def _mk_test(data: pd.Series, is_short: bool = False) -> Optional[str]:
        try:
            trend = mk.original_test(data)
            if trend.h:
                prefix = '短期' if is_short else ''
                return f"{prefix}上涨" if trend.z > 0 else f"{prefix}下降"
        except Exception:
            return None
        return None

    @staticmethod
    def _outliers_by_iqr(series: pd.Series):
        """
        判断非负Pandas Series是否存在异常值（IQR法）。

        参数:
            series: pd.Series，非负数据（含0）

        返回:
            bool，True表示存在异常值，False表示没有异常值
        """
        data = series.values.astype(float)

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1 * IQR
        upper_bound = Q3 + 1 * IQR

        outlier_mask = (data < lower_bound) | (data > upper_bound)

        return outlier_mask

    @staticmethod
    def _linear_trend(
        sales_data: pd.Series,
        period: str = '',
        slope_threshold=1.5,
        min_samples: int = 3,  # 最小有效数据量,
        r2_score=0.4   # 线性拟合要求
    ) -> str:
        if len(sales_data.dropna()) < min_samples:
            return None

        zero_ratio = (sales_data == 0).mean()
        if zero_ratio > 0.5:
            return None

        # 线性回归拟合
        X = np.arange(len(sales_data)).reshape(-1, 1)  # 时间轴
        y = sales_data.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0][0]

        r2 = model.score(X, y)
        if r2 < r2_score:
            return None

        # 判断趋势
        if abs(slope) < slope_threshold:  # 忽略微小斜率
            return None
        prefix = f'{period}天' if period else ''
        return f"{prefix}上涨" if slope > 0 else f"{prefix}下降"

    def analyze_trend(self, series: pd.Series) -> Dict[str, List[TrendType]]:
        """
            执行Mann-Kendall趋势分析:
                step1 长期趋势
                step2 检查暴涨暴跌
                step3 短期趋势

            返回： {'trend': ['无趋势']} or {'trend': ['上涨', '暴跌']}
        """

        result = {'trend': []}

        try:
            clean_series = pd.to_numeric(series, errors='coerce').fillna(0)
            # 超过10天销量低于2，不判断趋势
            low_sales_days_exceeded = ((clean_series < 2).sum() > 10)
            if low_sales_days_exceeded:
                result['trend'].append('无趋势')
                return result

            # 先看长期趋势
            long_term_trend = self._mk_test(clean_series)
            if long_term_trend:
                result['trend'].append(f'整体{long_term_trend}')

            # 检查各时间窗口的趋势
            for days, is_short in [
                (len(clean_series), False),  # 暴涨暴跌
                (self.config.short_term_days, True),  # 短期
                (self.config.very_short_term_days, True)  # 超短期
            ]:
                # 如果有暴涨暴跌，则跳过短期趋势
                if not is_short:
                    outlier = self.extreme_detect(clean_series)
                    if outlier:
                        result['trend'].append(outlier)
                        return result

                elif is_short and self.trend_method == 'mk':
                    trend_result = self._mk_test(
                        clean_series[-days:], is_short)
                    if trend_result:
                        result['trend'].append(trend_result)
                        return result

                elif is_short and self.trend_method == 'mix':
                    if self._outliers_by_iqr(clean_series)[-days]:
                        continue
                    if clean_series.iloc[-days] > clean_series[:-(days + 1)].sum():
                        continue
                    if days == self.config.short_term_days:
                        trend_result = self._linear_trend(
                            clean_series[-days:],
                            period=str(days),
                            slope_threshold=self.lr_slop_threshold,
                            r2_score=self.lr_r2_score)
                    else:
                        # 3天趋势 每天都增或者每天都降
                        diff = clean_series[-days:].diff()[-2:]
                        if not ((diff.min() > 0) or (diff.max() < 0)):
                            continue
                        trend_result = self._linear_trend(
                            clean_series[-days:],
                            period=str(days),
                            slope_threshold=self.lr_slop_threshold,
                            r2_score=0.75)
                    if trend_result:
                        result['trend'].append(trend_result)
                        return result

            if len(result['trend']) != 0:
                return result

        except Exception as e:
            result['trend'] = '计算错误'
            print(
                f"Trend analysis error (item_id: {series.name}): {type(e).__name__}: {e}")

        result['trend'].append('无趋势')
        return result

    @staticmethod
    def complete_missing_dates(df: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """补全缺失日期数据"""
        def complete_group(group):
            item_id = group.name
            try:
                result = group.set_index('date').reindex(
                    date_range, fill_value=0).reset_index()
                result['item_id'] = item_id
                return result
            except Exception as e:
                print(f"日期补全错误 (item_id: {item_id}): {str(e)}")
                return pd.DataFrame()

        return df.groupby('item_id', group_keys=False).apply(complete_group).reset_index(drop=True)

    def generate_trend_report(self, sales_df, end_date, metrics) -> pd.DataFrame:

        # 准备日期范围
        date_range = pd.date_range(
            start=end_date - pd.Timedelta(days=self.config.window_period),
            end=end_date - pd.Timedelta(days=1),
            freq='D',
            name='date'
        )

        # 补全日期
        complete_df = self.complete_missing_dates(
            sales_df, date_range).sort_values('date')

        detailed_results = complete_df.groupby('item_id').agg(
            **{
                f"{metric}": pd.NamedAgg(
                    column=col,
                    aggfunc=lambda x: self.analyze_trend(x)['trend'])
                for metric, col in metrics.items()
            }
        )
        exploded_results = detailed_results.explode('trend')

        # 合并数据
        merged_data = complete_df.merge(exploded_results, on='item_id')

        return merged_data
