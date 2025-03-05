# -*- coding: utf-8 -*-
import abc
import os
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Tuple, NoReturn, Union, List
from datetime import datetime

import feather
import numpy as np
import pandas as pd
import lightgbm as lgb 
from lightgbm import LGBMClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from collections import defaultdict
import gc

# 定义时间常量（单位：秒）
ONE_MINUTE = 60  # 一分钟的秒数
ONE_HOUR = 3600  # 一小时的秒数（60秒 * 60分钟）
ONE_DAY = 86400  # 一天的秒数（60秒 * 60分钟 * 24小时）


@dataclass
class Config(object):
    """
    配置类, 用于存储和管理程序的配置信息
    包括时间窗口大小、路径设置、日期范围、特征提取间隔等
    """

    # 重要! 如果使用 csv 文件则设置 DATA_SUFFIX 为 csv, 如果使用 feather 文件则设置为 feather
    DATA_SUFFIX: str = field(default="feather", init=False)

    # 时间窗口大小映射表, 键为时间长度(秒), 值为对应的字符串表示
    TIME_WINDOW_SIZE_MAP: dict = field(
        default_factory=lambda: {
            15 * ONE_MINUTE: "15m",
            1 * ONE_HOUR: "1h"
        },
        init=False, # 表示在创建 Config 类实例时，TIME_WINDOW_SIZE_MAP 字段不会作为一个参数传递给 __init__ 方法
    )

    # 与时间相关的列表, 存储常用的时间间隔(秒)
    TIME_RELATED_LIST: List[int] = field(
        default_factory=lambda: [15 * ONE_MINUTE, ONE_HOUR],
        init=False,
    )

    # 缺失值填充的默认值
    IMPUTE_VALUE: int = field(default=-1, init=False)

    # 是否使用多进程
    USE_MULTI_PROCESS: bool = field(default=True, init=False)

    # 如果使用多进程, 并行时 worker 的数量
    WORKER_NUM: int = field(default=32, init=False)

    # 数据路径配置, 分别是原始数据集路径、生成的特征路径、处理后训练集特征路径、处理后测试集特征路径、维修单路径
    data_path: str = "To be filled"
    feature_path: str = "To be filled"
    train_data_path: str = "To be filled"
    test_data_path: str = "To be filled"
    ticket_path: str = "To be filled"

    # 日期范围配置
    train_date_range: tuple = ("2024-01-01", "2024-06-01")
    test_data_range: tuple = ("2024-06-01", "2024-08-01")

    # 特征提取的时间间隔(秒), 为了更高的性能, 可以修改为 15 * ONE_MINUTE 或 30 * ONE_MINUTE
    feature_interval: int = ONE_HOUR


class FeatureFactory(object):
    """
    特征工厂类, 用于生成特征
    """

    # 考虑 DDR4 内存, 其 DQ_COUNT 和 BURST_COUNT 分别为 4 和 8
    DQ_COUNT = 4
    BURST_COUNT = 8

    def __init__(self, config: Config):
        """
        初始化特征工厂类

        :param config: 配置类实例, 包含路径等信息
        """

        self.config = config
        os.makedirs(self.config.feature_path, exist_ok=True)
        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.test_data_path, exist_ok=True)

    def _unique_num_filtered(self, input_array: np.ndarray) -> int:
        """
        对输入的列表进行去重, 再去除值为 IMPUTE_VALUE 的元素后, 统计元素个数

        :param input_array: 输入的列表
        :return: 返回经过过滤后的列表元素个数
        """

        unique_array = np.unique(input_array) # 获取 input_array 中的唯一元素
        return len(unique_array) - int(self.config.IMPUTE_VALUE in unique_array) # 检查 unique_array 中是否包含缺失值（IMPUTE_VALUE）

    @staticmethod     # 表示该方法是一个静态方法，不需要访问或修改类的状态
    def _calculate_ce_storm_count(
        log_times: np.ndarray,
        ce_storm_interval_seconds: int = 60,
        ce_storm_count_threshold: int = 10,
    ) -> int:
        """
        计算 CE 风暴的数量

        CE 风暴定义:
        - 首先定义相邻 CE 日志: 若两个 CE 日志 LogTime 时间间隔 < 60s, 则为相邻日志;
        - 如果相邻日志的个数 >10, 则为发生 1 次 CE 风暴(注意: 如果相邻日志数量持续增长, 超过了 10, 则也只是记作 1 次 CE 风暴)

        :param log_times: 日志 LogTime 列表
        :param ce_storm_interval_seconds: CE 风暴的时间间隔阈值
        :param ce_storm_count_threshold: CE 风暴的数量阈值
        :return: CE风暴的数量
        """

        log_times = sorted(log_times)
        ce_storm_count = 0
        consecutive_count = 0

        for i in range(1, len(log_times)):
            if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count > ce_storm_count_threshold:
                ce_storm_count += 1
                consecutive_count = 0

        return ce_storm_count

    def _get_temporal_features(
        self, window_df: pd.DataFrame, time_window_size: int
    ) -> Dict[str, int]:
        """
        获取时间特征, 包括 CE 数量、日志数量、CE 风暴数量、日志发生频率等

        :param window_df: 时间窗口内的数据
        :param time_window_size: 时间窗口大小
        :return: 时间特征

        - read_ce_log_num, read_ce_count: 时间窗口内, 读 CE 的 count 总数, 日志总数
        - scrub_ce_log_num, scrub_ce_count: 时间窗口内, 巡检 CE 的 count 总数, 日志总数
        - all_ce_log_num, all_ce_count: 时间窗口内, 所有 CE 的 count 总数, 日志总数
        - log_happen_frequency: 日志发生频率
        - ce_storm_count: CE 风暴数量
        """

        error_type_is_READ_CE = window_df["error_type_is_READ_CE"].values # 布尔数组
        error_type_is_SCRUB_CE = window_df["error_type_is_SCRUB_CE"].values
        ce_count = window_df["Count"].values # 包含每个日志错误次数的数组

        temporal_features = dict()
        temporal_features["read_ce_log_num"] = error_type_is_READ_CE.sum()
        temporal_features["scrub_ce_log_num"] = error_type_is_SCRUB_CE.sum()
        temporal_features["all_ce_log_num"] = len(window_df)

        temporal_features["read_ce_count"] = (error_type_is_READ_CE * ce_count).sum()
        temporal_features["scrub_ce_count"] = (error_type_is_SCRUB_CE * ce_count).sum()
        temporal_features["all_ce_count"] = ce_count.sum() # 计算所有 CE 的 Count 总数

        # 计算日志发生频率，日志发生频率定义为时间窗口大小除以日志总数
        temporal_features["log_happen_frequency"] = (
            time_window_size / len(window_df) if not window_df.empty else 0
        )
        temporal_features["ce_storm_count"] = self._calculate_ce_storm_count(
            window_df["LogTime"].values
        )

        # 筛选出 READ_CE 错误的日志记录
        read_ce_logs = window_df[window_df["error_type_is_READ_CE"] == 1]
        read_ce_times = read_ce_logs["LogTime"].sort_values().values
        if len(read_ce_times) > 1:
            read_ce_intervals = np.diff(read_ce_times)
            min_time_interval_read_ce = read_ce_intervals.min()
        else:
            min_time_interval_read_ce = -1
        temporal_features["min_time_interval_read_ce"] = min_time_interval_read_ce

        # 筛选出 SCRUB_CE 错误的日志记录
        scrub_ce_logs = window_df[window_df["error_type_is_SCRUB_CE"] == 1]
        scrub_ce_times = scrub_ce_logs["LogTime"].sort_values().values
        if len(scrub_ce_times) > 1:
            scrub_ce_intervals = np.diff(scrub_ce_times)
            min_time_interval_scrub_ce = scrub_ce_intervals.min()
        else:
            min_time_interval_scrub_ce = -1
        temporal_features["min_time_interval_scrub_ce"] = min_time_interval_scrub_ce
        
        return temporal_features

    def _get_spatio_features(self, window_df: pd.DataFrame) -> Dict[str, int]:
        """
        获取空间特征, 包括故障模式, 同时发生行列故障的数量

        :param window_df: 时间窗口内的数据
        :return: 空间特征

        - fault_mode_others: 其他故障, 即多个 device 发生故障
        - fault_mode_device: device 故障, 相同 id 的 device 发生多个 bank 故障
        - fault_mode_bank: bank 故障, 相同 id 的 bank 发生多个 row故障
        - fault_mode_row: row 故障, 相同 id 的 row 发生多个不同 column 的 cell 故障
        - fault_mode_column: column 故障, 相同 id 的 column 发生多个不同 row 的 cell 故障
        - fault_mode_cell: cell 故障, 发生多个相同 id 的 cell 故障
        - fault_row_num: 同时发生 row 故障的行个数
        - fault_column_num: 同时发生 column 故障的列个数
        """

        spatio_features = {
            "fault_mode_others": 0,
            "fault_mode_device": 0,
            "fault_mode_bank": 0,
            "fault_mode_row": 0,
            "fault_mode_column": 0,
            "fault_mode_cell": 0,
            "fault_row_num": 0,
            "fault_column_num": 0,
        }

        # 根据故障设备、Bank、行、列和单元的数量判断故障模式
        if self._unique_num_filtered(window_df["deviceID"].values) > 1:
            spatio_features["fault_mode_others"] = 1
        elif self._unique_num_filtered(window_df["BankId"].values) > 1:
            spatio_features["fault_mode_device"] = 1
        elif (
            self._unique_num_filtered(window_df["ColumnId"].values) > 1
            and self._unique_num_filtered(window_df["RowId"].values) > 1
        ):
            spatio_features["fault_mode_bank"] = 1
        elif self._unique_num_filtered(window_df["ColumnId"].values) > 1:
            spatio_features["fault_mode_row"] = 1
        elif self._unique_num_filtered(window_df["RowId"].values) > 1:
            spatio_features["fault_mode_column"] = 1
        elif self._unique_num_filtered(window_df["CellId"].values) == 1:
            spatio_features["fault_mode_cell"] = 1

        # 记录相同行对应的列地址信息
        row_pos_dict = {}
        # 记录相同列对应的行地址信息
        col_pos_dict = {}

        for device_id, bank_id, row_id, column_id in zip(
            window_df["deviceID"].values,
            window_df["BankId"].values,
            window_df["RowId"].values,
            window_df["ColumnId"].values,
        ):
            current_row = "_".join([str(pos) for pos in [device_id, bank_id, row_id]])
            current_col = "_".join(
                [str(pos) for pos in [device_id, bank_id, column_id]]
            )
            row_pos_dict.setdefault(current_row, [])
            col_pos_dict.setdefault(current_col, [])
            row_pos_dict[current_row].append(column_id)
            col_pos_dict[current_col].append(row_id)

        for row in row_pos_dict:
            # 遍历 row_pos_dict，如果某个 row 对应的 column 列表中唯一值的数量大于 1，则表示该 row 同时发生了多个 column 故障，fault_row_num 加 1
            if self._unique_num_filtered(np.array(row_pos_dict[row])) > 1:
                spatio_features["fault_row_num"] += 1
        for col in col_pos_dict:
            # 遍历 col_pos_dict，如果某个 column 对应的 row 列表中唯一值的数量大于 1，则表示该 column 同时发生了多个 row 故障，fault_column_num 加 1
            if self._unique_num_filtered(np.array(col_pos_dict[col])) > 1:
                spatio_features["fault_column_num"] += 1

        return spatio_features

    @staticmethod
    def _get_err_parity_features(window_df: pd.DataFrame) -> Dict[str, int]:
        """
        获取奇偶校验特征

        :param window_df: 时间窗口内的数据
        :return: 奇偶校验特征

        - error_bit_count: 时间窗口内, 总错误 bit 数
        - error_dq_count: 时间窗口内, 总 dq 错误数
        - error_burst_count: 时间窗口内, 总 burst 错误数
        - max_dq_interval: 时间窗口内, 每个 parity 最大错误 dq 距离的最大值
        - max_burst_interval: 时间窗口内, 每个 parity 最大错误 burst 距离的最大值
        - dq_count=n: dq 错误数等于 n 的总数量, n 取值范围为 [1, 2, 3, 4], 默认值为 0
        - burst_count=n: burst 错误数等于 n 的总数量, n 取值范围为 [1, 2, 3, 4, 5, 6, 7, 8], 默认值为 0
        """

        """
        err_parity_features = dict()

        # 计算总错误 bit 数、DQ 错误数和 Burst 错误数
        err_parity_features["error_bit_count"] = window_df["bit_count"].values.sum()
        err_parity_features["error_dq_count"] = window_df["dq_count"].values.sum()
        err_parity_features["error_burst_count"] = window_df["burst_count"].values.sum()

        # 计算最大 DQ 间隔和最大 Burst 间隔
        err_parity_features["max_dq_interval"] = window_df[
            "max_dq_interval"
        ].values.max()
        err_parity_features["max_burst_interval"] = window_df[
            "max_burst_interval"
        ].values.max()

        # 统计 DQ 错误数和 Burst 错误数的分布
        dq_counts = dict()
        burst_counts = dict()
        for dq, burst in zip(
            window_df["dq_count"].values, window_df["burst_count"].values
        ):
            dq_counts[dq] = dq_counts.get(dq, 0) + 1
            burst_counts[burst] = burst_counts.get(burst, 0) + 1

        # 计算 'dq错误数=n' 的总数量, DDR4 内存的 DQ_COUNT 为 4, 因此 n 取值 [1,2,3,4]
        for dq in range(1, FeatureFactory.DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)

        # 计算 'burst错误数=n' 的总数量, DDR4 内存的 BURST_COUNT 为 8, 因此 n 取值 [1,2,3,4,5,6,7,8]
        for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)

        return err_parity_features """

        err_parity_features = dict()
        spatial_features = [
            'bit_count', 'dq_count', 'burst_count',
            'adjacent_dq_count', 'min_dq_interval', 'avg_dq_interval',
            'adjacent_burst_count', 'min_burst_interval', 'avg_burst_interval',
            'max_dq_interval', 'max_burst_interval'
        ]

        # 计算统计量：sum/max/min/avg/std
        for feat in spatial_features:
            if feat in window_df.columns:
                values = window_df[feat]
                err_parity_features[f"{feat}_sum"] = values.sum()
                err_parity_features[f"{feat}_max"] = values.max() if len(values) > 0 else 0
                err_parity_features[f"{feat}_min"] = values.min() if len(values) > 0 else 0
                err_parity_features[f"{feat}_avg"] = values.mean() if len(values) > 0 else 0.0
                err_parity_features[f"{feat}_std"] = values.std() if len(values) > 0 else 0.0

        # 原始 DQ/Burst 分布统计
        dq_counts = defaultdict(int)
        burst_counts = defaultdict(int)
        for dq, burst in zip(window_df["dq_count"], window_df["burst_count"]):
            dq_counts[dq] += 1
            burst_counts[burst] += 1

        for dq in range(1, FeatureFactory.DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)
        for burst in range(1, FeatureFactory.BURST_COUNT + 1):
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)

        # 计算 CE DQ Cnt(θDQ)
        theta_values = [2, 3]  # 可配置的θ阈值
        for theta in theta_values:
            cnt = 0
            for _, row in window_df.iterrows():
                max_dq = row.get("max_dq", -1)
                min_dq = row.get("min_dq", -1)
                if max_dq != -1 and min_dq != -1 and (max_dq - min_dq + 1) <= theta:
                    cnt += 1
            err_parity_features[f"ce_dq_cnt_theta_{theta}"] = cnt

        return err_parity_features

    @staticmethod
    def _get_bit_dq_burst_info(parity: np.int64) -> Tuple[int, int, int, int, int]:
        """
        获取特定 parity 的奇偶校验信息

        :param parity: 奇偶校验值
        :return: parity 的奇偶校验信息

        - bit_count: parity 错误 bit 数量
        - dq_count: parity 错误 dq 数量
        - burst_count: parity 错误 burst 数量
        - max_dq_interval: parity 错误 dq 的最大间隔
        - max_burst_interval: parity 错误 burst 的最大间隔
        """

        # 将 Parity 转换为 32 位二进制字符串，通过切片操作去掉了二进制字符串的前缀 '0b'，并在字符串的左侧填充零直到字符串的长度达到32位
        bin_parity = bin(parity)[2:].zfill(32)

        # 计算错误 bit 数量
        bit_count = bin_parity.count("1")

        # 计算 DQ 相关特征
        # 将 bin_parity 字符串分成 4 组，每组包含 8 个连续的字符，用于分别计算每个 dq（数据位）上的错误数量
        binary_column_array = [bin_parity[i::4].count("1") for i in range(4)] 
        binary_column_array_indices = [idx for idx, value in enumerate(binary_column_array) if value > 0] # 记录错误的 dq 索引
        dq_count = len(binary_column_array_indices) # 错误的 dq 数量
        max_dq_interval = -1
        adjacent_dq_count = -1
        min_dq_interval = -1
        avg_dq_interval = -1
        max_dq = -1
        min_dq = -1

        # 计算相邻 DQ 数量
        if binary_column_array_indices:
            sorted_dq = sorted(binary_column_array_indices) # 排序后的错误 dq 索引
            max_dq_interval = sorted_dq[-1] - sorted_dq[0] # 最大间隔
            if max_dq_interval > 0:
                # 计算相邻 DQ 对数和间隔
                intervals = []
                for i in range(1, len(sorted_dq)):
                    interval = sorted_dq[i] - sorted_dq[i-1]
                    intervals.append(interval)
                    if interval == 1:
                        adjacent_dq_count += 1
                min_dq_interval = min(intervals) if intervals else 0
                avg_dq_interval = sum(intervals)/len(intervals) if intervals else 0.0
                max_dq = sorted_dq[-1]
                min_dq = sorted_dq[0]

        # 计算 Burst 相关特征
        binary_row_array = [bin_parity[i:i+4].count("1") for i in range(0, 32, 4)]
        binary_row_array_indices = [idx for idx, value in enumerate(binary_row_array) if value > 0]
        burst_count = len(binary_row_array_indices)
        max_burst_interval = -1
        adjacent_burst_count = -1
        min_burst_interval = -1
        avg_burst_interval = -1
        max_burst = -1
        min_burst = -1

        if binary_row_array_indices:
            sorted_burst = sorted(binary_row_array_indices)
            max_burst_interval = sorted_burst[-1] - sorted_burst[0]
            if max_burst_interval > 0:
                # 计算相邻 Burst 对数和间隔
                intervals = []
                for i in range(1, len(sorted_burst)):
                    interval = sorted_burst[i] - sorted_burst[i-1]
                    intervals.append(interval)
                    if interval == 1:
                        adjacent_burst_count += 1
                min_burst_interval = min(intervals) if intervals else 0
                avg_burst_interval = sum(intervals)/len(intervals) if intervals else 0.0
                max_burst = sorted_burst[-1]
                min_burst = sorted_burst[0]

        """
        # 计算 burst 相关特征
        binary_row_array = [bin_parity[i : i + 4].count("1") for i in range(0, 32, 4)] # 将 bin_parity 字符串分成 8 个 4 位的子字符串，并计算每个子字符串中 '1' 的数量
        binary_row_array_indices = [
            idx for idx, value in enumerate(binary_row_array) if value > 0
        ]
        burst_count = len(binary_row_array_indices)
        max_burst_interval = (
            binary_row_array_indices[-1] - binary_row_array_indices[0]
            if binary_row_array_indices
            else 0
        )

        # 计算 dq 相关特征
        # 通过步长为 4 的切片操作，将 bin_parity 字符串分成 4 组，每组包含 8 个连续的字符，用于分别计算每个 dq（数据位）上的错误数量
        binary_column_array = [bin_parity[i::4].count("1") for i in range(4)] # i::4 表示从索引 i 开始，每隔 4 个字符取一个字符，直到字符串结束
        binary_column_array_indices = [
            idx for idx, value in enumerate(binary_column_array) if value > 0
        ]
        dq_count = len(binary_column_array_indices)
        max_dq_interval = (
            binary_column_array_indices[-1] - binary_column_array_indices[0]
            if binary_column_array_indices
            else 0
        )

        return bit_count, dq_count, burst_count, max_dq_interval, max_burst_interval """
        return (
            bit_count, dq_count, burst_count,
            max_dq_interval, max_burst_interval,
            adjacent_dq_count, min_dq_interval, avg_dq_interval,
            max_dq, min_dq, adjacent_burst_count,
            min_burst_interval, avg_burst_interval, max_burst, min_burst
        )
    
    def _get_static_config_features(self, raw_df: pd.DataFrame) -> dict:
        """
        从原始raw_df中提取系统静态配置信息并进行独热编码

        :param raw_df: 原始数据DataFrame
        :return: 包含独热编码特征的字典
        """
        static_features = ['Manufacturer', 'Model', 'region', 'CpuId', 'ChannelId', 'DimmId', 'Capacity', 'FrequencyMHz']
        encoded_dict = {}

        for feature in static_features:
            # 区分数值型和类别型特征处理缺失值
            if feature in ['CpuId', 'ChannelId', 'DimmId', 'Capacity', 'FrequencyMHz']:
                # 数值型特征用-1填充缺失值，并转换为整数/浮点
                fill_value = -1
                raw_df[feature] = raw_df[feature].fillna(fill_value)
                if feature == 'FrequencyMHz':
                    raw_df[feature] = raw_df[feature].astype(float)
                else:
                    raw_df[feature] = raw_df[feature].astype(int)
            else:
                # 类别型特征用'NaN'填充缺失值
                raw_df[feature] = raw_df[feature].fillna('NaN').astype(str)

            # 定义每个特征的所有可能类别
            if feature == 'Manufacturer':
                categories = ['D', 'B', 'A', 'C', 'NaN']
            elif feature == 'Model':
                categories = ['L', 'D', 'B', 'K', 'M', 'H', 'I', 'A', 'J', 'P', 'N', 'F', 'E', 'NaN']
            elif feature == 'region':
                categories = ['L', 'D', 'B', 'K', 'M', 'O', 'G', 'I', 'H', 'A', 'C', 'N', 'P', 'E', 'F', 'Q', 'NaN']
            elif feature == 'CpuId':
                categories = [0, 1, 2, 3, -1]
            elif feature == 'ChannelId':
                categories = [0, 1, 2, 3, 4, 5, 6, 7, -1]
            elif feature == 'DimmId':
                categories = [0, 1, -1]
            elif feature == 'Capacity':
                categories = [32, 16, 64, 128, -1]
            elif feature == 'FrequencyMHz':
                categories = [2400.0, 3200.0, 2600.0, 2700.0, 2800.0, 2100.0, 2300.0, 3000.0, 2200.0, 3100.0, -1.0]

            # 转换为分类类型以确保生成所有可能的虚拟列
            cat_series = pd.Categorical(raw_df[feature], categories=categories)
            one_hot = pd.get_dummies(cat_series, prefix=feature, prefix_sep='_', sparse=True)

            # 确保数据类型为数值型
            one_hot = one_hot.astype(int)

            # 将列加入字典
            for col in one_hot.columns:
                encoded_dict[col] = one_hot[col].values[0]

        return encoded_dict

    def _calculate_error_counts(self, new_df, end_logtime):
        time_ranges = {
            "1min": 1 * ONE_MINUTE,
            "5min": 5 * ONE_MINUTE,
            "15min": 15 * ONE_MINUTE,
            "30min": 30 * ONE_MINUTE,
            "1h": ONE_HOUR,
            "3h": 3 * ONE_HOUR,
            "6h": 6 * ONE_HOUR,
            "12h": 12 * ONE_HOUR,
            "1d": ONE_DAY,
            "2d": 2 * ONE_DAY,
            "3d": 3 * ONE_DAY,
            "5d": 5 * ONE_DAY
        }
        error_counts = {}
        error_numbers = [2, 3, 5, 7]
        for time_range, seconds in time_ranges.items():
            start_time = end_logtime - seconds
            mask = (new_df['LogTime'] >= start_time) & (new_df['LogTime'] <= end_logtime)
            recent_data = new_df.loc[mask]
            ce_read_count = recent_data['error_type_is_READ_CE'].sum()
            ce_scrub_count = recent_data['error_type_is_SCRUB_CE'].sum()
            error_counts[f'error_count_CE.READ_{time_range}'] = ce_read_count
            error_counts[f'error_count_CE.SCRUB_{time_range}'] = ce_scrub_count

            # 处理 READ_CE 错误的最小时间间隔
            read_ce_logs = recent_data[recent_data['error_type_is_READ_CE'] == 1]
            read_ce_times = read_ce_logs['LogTime'].sort_values().values
            for num in error_numbers:
                if len(read_ce_times) >= num:
                    min_interval = np.inf
                    for i in range(len(read_ce_times) - num + 1):
                        interval = read_ce_times[i + num - 1] - read_ce_times[i]
                        if interval < min_interval:
                            min_interval = interval
                    error_counts[f'min_time_interval_READ_CE_{num}_{time_range}'] = min_interval
                else:
                    error_counts[f'min_time_interval_READ_CE_{num}_{time_range}'] = -1

            # 处理 SCRUB_CE 错误的最小时间间隔
            scrub_ce_logs = recent_data[recent_data['error_type_is_SCRUB_CE'] == 1]
            scrub_ce_times = scrub_ce_logs['LogTime'].sort_values().values
            for num in error_numbers:
                if len(scrub_ce_times) >= num:
                    min_interval = np.inf
                    for i in range(len(scrub_ce_times) - num + 1):
                        interval = scrub_ce_times[i + num - 1] - scrub_ce_times[i]
                        if interval < min_interval:
                            min_interval = interval
                    error_counts[f'min_time_interval_SCRUB_CE_{num}_{time_range}'] = min_interval
                else:
                    error_counts[f'min_time_interval_SCRUB_CE_{num}_{time_range}'] = -1
        return error_counts

    def _get_processed_df(self, sn_file: str) -> pd.DataFrame:
        """
        获取处理后的 DataFrame

        处理步骤包括：
        - 对 raw_df 按 LogTime 排序
        - 将 error_type 转换为独热编码
        - 填充缺失值
        - 添加奇偶校验特征

        :param sn_file: SN 文件名
        :return: 处理后的 DataFrame
        """

        parity_dict = dict()

        # 读取原始数据并按 LogTime 排序
        if self.config.DATA_SUFFIX == "csv":
            raw_df = pd.read_csv(os.path.join(self.config.data_path, sn_file), dtype={
                        'CpuId': 'int32',
                        'ChannelId': 'int32',
                        'DimmId': 'int32',
                        'RankId': 'int8',
                        'deviceID': 'int8',
                        'BankgroupId': 'int8',
                        'BankId': 'int8',
                        'RowId': 'int32',
                        'ColumnId': 'int32',
                        'RetryRdErrLogParity': 'int64',
                        'RetryRdErrLog': 'int64',
                        'burst_info': 'int64',
                        'Capacity': 'int32',
                        'FrequencyMHz': 'float32',
                        'MaxSpeedMHz': 'float32'
            })
        else:
            raw_df = feather.read_dataframe(os.path.join(self.config.data_path, sn_file))
            raw_df = raw_df.astype({
                        'CpuId': 'int32',
                        'ChannelId': 'int32',
                        'DimmId': 'int32',
                        'RankId': 'int8',
                        'deviceID': 'int8',
                        'BankgroupId': 'int8',
                        'BankId': 'int8',
                        'RowId': 'int32',
                        'ColumnId': 'int32',
                        'RetryRdErrLogParity': 'int64',
                        'RetryRdErrLog': 'int64',
                        'burst_info': 'int64',
                        'Capacity': 'int32',
                        'FrequencyMHz': 'float32',
                        'MaxSpeedMHz': 'float32'
                    })

        raw_df = raw_df.sort_values(by="LogTime").reset_index(drop=True)

        # 提取需要的列并初始化 processed_df
        processed_df = raw_df[
            [
                "LogTime",
                "deviceID",
                "BankId",
                "RowId",
                "ColumnId",
                "MciAddr",
                "RetryRdErrLogParity",
            ]
        ].copy()

        # deviceID 可能存在缺失值, 填充缺失值
        processed_df["deviceID"] = (
            processed_df["deviceID"].fillna(self.config.IMPUTE_VALUE).astype(int)
        )

        # 将 error_type 转换为独热编码
        processed_df["error_type_is_READ_CE"] = (
            raw_df["error_type_full_name"] == "CE.READ"
        ).astype(int)
        processed_df["error_type_is_SCRUB_CE"] = (
            raw_df["error_type_full_name"] == "CE.SCRUB"
        ).astype(int)

        processed_df["CellId"] = (
            processed_df["RowId"].astype(str)
            + "_"
            + processed_df["ColumnId"].astype(str)
        )
        processed_df["position_and_parity"] = (
            processed_df["deviceID"].astype(str)
            + "_"
            + processed_df["BankId"].astype(str)
            + "_"
            + processed_df["RowId"].astype(str)
            + "_"
            + processed_df["ColumnId"].astype(str)
            + "_"
            + processed_df["RetryRdErrLogParity"].astype(str)
        )

        err_log_parity_array = (
            processed_df["RetryRdErrLogParity"]
            .fillna(0)  # 对于缺失值（NaN），将其替换为整数 0。这一步是为了确保后续的操作不会因为存在缺失值而中断。
            .replace("", 0) # 如果存在空字符串（""），将其替换为整数 0。这一步是为了处理那些可能被错误地记录为空字符串的奇偶校验值。
            .astype(np.int64)  # 转换为 np.int64, 此处如果为 int 会溢出
            .values
        )

        # 计算每个 parity 的 bit_count、dq_count、burst_count、max_dq_interval 和 max_burst_interval
        bit_dq_burst_count = list()
        for idx, err_log_parity in enumerate(err_log_parity_array):
            if err_log_parity not in parity_dict:
                parity_dict[err_log_parity] = self._get_bit_dq_burst_info(
                    err_log_parity
                )
            bit_dq_burst_count.append(parity_dict[err_log_parity])

        processed_df = processed_df.join(
            pd.DataFrame(
                bit_dq_burst_count,
                columns=[
                    "bit_count", "dq_count", "burst_count",
                    "max_dq_interval", "max_burst_interval",
                    "adjacent_dq_count", "min_dq_interval", "avg_dq_interval",
                    "max_dq", "min_dq", "adjacent_burst_count",
                    "min_burst_interval", "avg_burst_interval", "max_burst", "min_burst"
                ],
            )
        )
        del raw_df
        gc.collect()
        return processed_df

    def process_single_sn(self, sn_file: str) -> NoReturn:
        """
        处理单个 sn 文件, 获取不同尺度的时间窗口特征

        :param sn_file: sn 文件名
        """

        # 读取原始数据获取静态配置信息
        if self.config.DATA_SUFFIX == "csv":
            raw_df = pd.read_csv(os.path.join(self.config.data_path, sn_file))
        else:
            raw_df = feather.read_dataframe(os.path.join(self.config.data_path, sn_file))
            raw_df = raw_df.astype({
                        'CpuId': 'int32',
                        'ChannelId': 'int32',
                        'DimmId': 'int32',
                        'RankId': 'int8',
                        'deviceID': 'int8',
                        'BankgroupId': 'int8',
                        'BankId': 'int8',
                        'RowId': 'int32',
                        'ColumnId': 'int32',
                        'RetryRdErrLogParity': 'int64',
                        'RetryRdErrLog': 'int64',
                        'burst_info': 'int64',
                        'Capacity': 'int32',
                        'FrequencyMHz': 'float32',
                        'MaxSpeedMHz': 'float32'
                    })

        static_config_features = self._get_static_config_features(raw_df)
        del raw_df
        gc.collect()

        # 获取处理后的 DataFrame
        new_df = self._get_processed_df(sn_file)

        # 根据生成特征的间隔, 计算时间索引
        new_df["time_index"] = new_df["LogTime"] // self.config.feature_interval
        log_times = new_df["LogTime"].values

        # 计算每个时间窗口的结束时间和开始时间, 每次生成特征最多用 max_window_size 的历史数据
        max_window_size = max(self.config.TIME_RELATED_LIST)
        window_end_times = new_df.groupby("time_index")["LogTime"].max().values
        window_start_times = window_end_times - max_window_size

        # 根据时间窗口的起始和结束时间, 找到对应的数据索引
        start_indices = np.searchsorted(log_times, window_start_times, side="left")
        end_indices = np.searchsorted(log_times, window_end_times, side="right")

        combined_dict_list = []
        for start_idx, end_idx, end_time in zip(
            start_indices, end_indices, window_end_times
        ):
            combined_dict = {}
            window_df = new_df.iloc[start_idx:end_idx]
            combined_dict["LogTime"] = window_df["LogTime"].values.max()

            # 计算每个 position_and_parity 的出现次数, 并去重
            window_df = window_df.assign(
                Count=window_df.groupby("position_and_parity")[
                    "position_and_parity"
                ].transform("count")
            )
            window_df = window_df.drop_duplicates(
                subset="position_and_parity", keep="first"
            )
            log_times = window_df["LogTime"].values
            end_logtime_of_filtered_window_df = window_df["LogTime"].values.max()

            # 统计不同时间范围的错误次数
            error_counts = self._calculate_error_counts(new_df, end_logtime_of_filtered_window_df)
            combined_dict.update(error_counts)

            # 遍历不同时间窗口大小, 提取时间窗特征(和前面 max_window_size 对应, 时间窗不超过 max_window_size)
            for time_window_size in self.config.TIME_RELATED_LIST:
                index = np.searchsorted(
                    log_times,
                    end_logtime_of_filtered_window_df - time_window_size,
                    side="left",
                )
                window_df_copy = window_df.iloc[index:]

                # 提取时间特征、空间特征和奇偶校验特征
                temporal_features = self._get_temporal_features(
                    window_df_copy, time_window_size
                )
                spatio_features = self._get_spatio_features(window_df_copy)
                err_parity_features = self._get_err_parity_features(window_df_copy)

                # 将特征合并到 combined_dict 中, 并添加时间窗口大小的后缀
                combined_dict.update(
                    {
                        f"{key}_{self.config.TIME_WINDOW_SIZE_MAP[time_window_size]}": value
                        for d in [
                            temporal_features,
                            spatio_features,
                            err_parity_features,
                        ]
                        for key, value in d.items()
                    }
                )

                # 处理完成后释放
                del window_df_copy
                gc.collect()

            # 添加静态配置信息
            combined_dict.update(static_config_features)
            
            combined_dict_list.append(combined_dict)

        # 将特征列表转换为 DataFrame 并保存为 feather 文件
        combined_df = pd.DataFrame(combined_dict_list)
        feather.write_dataframe(
            combined_df,
            os.path.join(self.config.feature_path, sn_file.replace("csv", "feather")),
        )

    def process_all_sn(self) -> NoReturn:
        """
        处理所有 sn 文件, 并保存特征, 支持多进程处理以提高效率
        """

        sn_files = os.listdir(self.config.data_path) # 获取原始数据集路径下所有文件的列表
        exist_sn_file_list = os.listdir(self.config.feature_path) # 获取生成特征路径下已存在的特征文件列表
        sn_files = [
            x for x in sn_files if x not in exist_sn_file_list and x.endswith(self.config.DATA_SUFFIX)
        ] # 筛选出那些原始数据集中存在但特征集中不存在的文件，并且文件扩展名符合配置中的 DATA_SUFFIX
        sn_files.sort()

        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                list(
                    tqdm(
                        pool.imap(self.process_single_sn, sn_files),
                        total=len(sn_files),
                        desc="Generating features",
                    )
                )
        else:
            for sn_file in tqdm(sn_files, desc="Generating features"):
                self.process_single_sn(sn_file)


class DataGenerator(metaclass=abc.ABCMeta):
    """
    数据生成器基类, 用于生成训练和测试数据
    """

    # 数据分块大小, 用于分批处理数据
    CHUNK_SIZE = 500 # 200

    def __init__(self, config: Config):
        """
        初始化数据生成器

        :param config: 配置类实例, 包含路径、日期范围等信息
        """

        self.config = config
        self.feature_path = self.config.feature_path
        self.train_data_path = self.config.train_data_path
        self.test_data_path = self.config.test_data_path
        self.ticket_path = self.config.ticket_path

        # 将日期范围转换为时间戳
        self.train_start_date = self._datetime_to_timestamp(
            self.config.train_date_range[0]
        )
        self.train_end_date = self._datetime_to_timestamp(
            self.config.train_date_range[1]
        )
        self.test_start_date = self._datetime_to_timestamp(
            self.config.test_data_range[0]
        )
        self.test_end_date = self._datetime_to_timestamp(
            self.config.test_data_range[1]
        )

        ticket = pd.read_csv(self.ticket_path)
        ticket = ticket[ticket["alarm_time"] <= self.train_end_date]
        self.ticket = ticket
        self.ticket_sn_map = {
            sn: sn_t
            for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
        }

        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.test_data_path, exist_ok=True)

    @staticmethod
    def concat_in_chunks(chunks: List) -> Union[pd.DataFrame, None]:
        """
        将 chunks 中的 DataFrame 进行拼接

        :param chunks: DataFrame 列表
        :return: 拼接后的 DataFrame, 如果 chunks 为空则返回 None
        """

        chunks = [chunk for chunk in chunks if chunk is not None]
        if chunks:
            return pd.concat(chunks)
        return None

    def parallel_concat(
        self, results: List, chunk_size: int = CHUNK_SIZE
    ) -> Union[pd.DataFrame, None]:
        """
        并行化的拼接操作, 可以视为 concat_in_chunks 的并行化版本

        :param results: 需要拼接的结果列表
        :param chunk_size: 每个 chunk 的大小
        :return: 拼接后的 DataFrame
        """

        chunks = [
            results[i : i + chunk_size] for i in range(0, len(results), chunk_size)
        ]

        # 使用多进程并行拼接
        worker_num = self.config.WORKER_NUM
        with Pool(worker_num) as pool:
            concatenated_chunks = pool.map(self.concat_in_chunks, chunks)

        return self.concat_in_chunks(concatenated_chunks)

    @staticmethod
    def _datetime_to_timestamp(date: str) -> int:
        """
        将 %Y-%m-%d 格式的日期转换为时间戳

        :param date: 日期字符串
        :return: 时间戳
        """

        return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    def _get_data(self) -> pd.DataFrame:
        """
        获取 feature_path 下的所有数据, 并进行处理

        :return: 处理后的数据
        """

        file_list = os.listdir(self.feature_path)
        file_list = [x for x in file_list if x.endswith(".feather")]
        file_list.sort()

        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._process_file, file_list),
                        total=len(file_list),
                        desc="Processing files",
                    )
                )
            data_all = self.parallel_concat(results)
        else:
            data_all = []
            data_chunk = []
            for i in tqdm(range(len(file_list)), desc="Processing files"):
                data = self._process_file(file_list[i])
                if data is not None:
                    data_chunk.append(data)
                if len(data_chunk) >= self.CHUNK_SIZE:
                    data_all.append(self.concat_in_chunks(data_chunk))
                    data_chunk = []
            if data_chunk:
                data_all.append(pd.concat(data_chunk))
            data_all = pd.concat(data_all)

        return data_all

    @abc.abstractmethod
    def _process_file(self, sn_file):
        """
        处理单个文件, 子类需要实现该方法

        :param sn_file: 文件名
        """

        raise NotImplementedError("Subclasses should implement this method")

    @abc.abstractmethod
    def generate_and_save_data(self):
        """
        生成并保存数据, 子类需要实现该方法
        """

        raise NotImplementedError("Subclasses should implement this method")


class PositiveDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取正样本数据

        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if self.ticket_sn_map.get(sn_name):
            # 设正样本的时间范围为维修单时间的前 30 天
            end_time = self.ticket_sn_map.get(sn_name)
            start_time = end_time - 30 * ONE_DAY

            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
            # 创建类型转换字典
            dtype_mapping = {}
            for col in data.columns:
                if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                    dtype_mapping[col] = "float32"
                else:
                    dtype_mapping[col] = "int32"
            # 应用类型转换
            data = data.astype(dtype_mapping)
            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            # data["label"] = 1
            if data.empty:
                return None

            # 创建标签列
            label_column = pd.DataFrame({'label': [1] * len(data)}, index=data.index)

            # 使用 pd.concat 一次性添加标签列
            data = pd.concat([data, label_column], axis=1)

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data

        # 如果 SN 名称不在维修单中, 则返回 None
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存正样本数据
        """

        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "positive_train.feather")
        )


class NegativeDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取负样本数据

        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if not self.ticket_sn_map.get(sn_name):
            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
            # 创建类型转换字典
            dtype_mapping = {}
            for col in data.columns:
                if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                    dtype_mapping[col] = "float32"
                else:
                    dtype_mapping[col] = "int32"
            # 应用类型转换
            data = data.astype(dtype_mapping)

            # 设负样本的时间范围为某段连续的 30 天
            end_time = self.train_end_date - 30 * ONE_DAY
            start_time = self.train_end_date - 60 * ONE_DAY

            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            if data.empty:
                return None
            # data["label"] = 0

            # 创建标签列
            label_column = pd.DataFrame({'label': [0] * len(data)}, index=data.index)

            # 使用 pd.concat 一次性添加标签列
            data = pd.concat([data, label_column], axis=1)

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data

        # 如果 SN 名称在维修单中, 则返回 None
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存负样本数据
        """

        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "negative_train.feather")
        )


class TestDataGenerator(DataGenerator):
    @staticmethod
    def _split_dataframe(df: pd.DataFrame, chunk_size: int = 2000000):
        """
        将 DataFrame 按照 chunk_size 进行切分

        :param df: 拆分前的 DataFrame
        :param chunk_size: chunk 大小
        :return: 切分后的 DataFrame, 每次返回一个 chunk
        """

        for start in range(0, len(df), chunk_size):
            yield df[start : start + chunk_size]

    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取测试数据

        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]

        # 读取特征文件, 并过滤出测试时间范围内的数据
        data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
        # 创建类型转换字典
        dtype_mapping = {}
        for col in data.columns:
            if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                dtype_mapping[col] = "float32"
            else:
                dtype_mapping[col] = "int32"
        # 应用类型转换
        data = data.astype(dtype_mapping)
        data = data[data["LogTime"] >= self.test_start_date]
        data = data[data["LogTime"] <= self.test_end_date]
        if data.empty:
            return None

        index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存测试数据
        """

        data_all = self._get_data()
        for index, chunk in enumerate(self._split_dataframe(data_all)):
            feather.write_dataframe(
                chunk, os.path.join(self.test_data_path, f"res_{index}.feather")
            )


class MFPmodel(object):
    """
    Memory Failure Prediction 模型类
    """

    def __init__(self, config: Config):
        """
        初始化模型类

        :param config: 配置类实例, 包含训练和测试数据的路径等信息
        """

        self.train_data_path = config.train_data_path
        self.test_data_path = config.test_data_path
        self.model_params = {
            "learning_rate": 0.02,
            "n_estimators": 1000,
            "max_depth": 8,
            "num_leaves": 20,
            "min_child_samples": 20,
            "verbose": 1,
            "n_jobs": -1,
        }

    def load_train_data(self) -> NoReturn:
        """
        加载训练数据
        """

        self.train_pos = feather.read_dataframe(
            os.path.join(self.train_data_path, "positive_train.feather")
        )
        # 创建类型转换字典
        dtype_mapping = {}
        for col in self.train_pos.columns:
            if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                dtype_mapping[col] = "float32"
            else:
                dtype_mapping[col] = "int32"
        # 应用类型转换
        self.train_pos = self.train_pos.astype(dtype_mapping)

        self.train_neg = feather.read_dataframe(
            os.path.join(self.train_data_path, "negative_train.feather")
        )
        # 创建类型转换字典
        dtype_mapping = {}
        for col in self.train_neg.columns:
            if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                dtype_mapping[col] = "float32"
            else:
                dtype_mapping[col] = "int32"
        # 应用类型转换
        self.train_neg = self.train_neg.astype(dtype_mapping)

    def load_test_data(self) -> NoReturn:
        """
        加载测试数据
        """
        test_files = [
            os.path.join(self.test_data_path, f)
            for f in os.listdir(self.test_data_path)
            if f.endswith(".feather")
        ]
        test_chunks = []
        for f in test_files:
            test_chunk = feather.read_dataframe(f)
            # 创建类型转换字典
            dtype_mapping = {}
            for col in test_chunk.columns:
                if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                    dtype_mapping[col] = "float32"
                else:
                    dtype_mapping[col] = "int32"
            # 应用类型转换
            test_chunk = test_chunk.astype(dtype_mapping)
            test_chunks.append(test_chunk)
        # test_chunks = [feather.read_dataframe(f) for f in test_files]
        self.test_data = pd.concat(test_chunks)
        self.test_data.drop("LogTime", axis=1, inplace=True, errors="ignore")
        self.test_data = self.test_data.sort_index(axis=1)

        train_all = pd.concat([self.train_pos, self.train_neg])
        train_all.drop("LogTime", axis=1, inplace=True)
        train_all = train_all.sort_index(axis=1)

        X = train_all.drop(columns=["label"])
        self.train_data = X

    def plot_feature_distributions(self):
        """
        对比训练集和测试集的特征分布
        """
        # 确保训练集和测试集特征列一致
        common_cols = list(set(self.train_data.columns) & set(self.test_data.columns))
        common_cols = [c for c in self.train_data.columns if c in common_cols]  # 保持顺序

        # 创建保存目录
        output_dir = "feature_distribution_comparison"
        os.makedirs(output_dir, exist_ok=True)

        # 对每个特征进行可视化
        ks_results = {}
        for feature in tqdm(common_cols, desc="Plotting features"):
            plt.figure(figsize=(10, 6))

            # 训练集数据（合并正负样本）
            sns.kdeplot(
                self.train_data[feature],
                color="blue",
                label="Train",
                fill=True,
                alpha=0.3
            )

            # 测试集数据
            sns.kdeplot(
                self.test_data[feature],
                color="red",
                label="Test",
                fill=True,
                alpha=0.3
            )

            # 添加统计检验结果
            stat, p = ks_2samp(
                self.train_data[feature].dropna(),
                self.test_data[feature].dropna()
            )
            ks_results[feature] = (stat, p)
            plt.title(
                f"{feature}\nKS Test: stat={stat:.3f}, p={p:.3f}",
                fontsize=10
            )
            plt.legend()

            # 保存图片
            plt.savefig(
                os.path.join(output_dir, f"{feature}.png"),
                dpi=150,
                bbox_inches="tight"
            )
            plt.close()

        # 特征选择：根据 KS 检验的 p 值选择特征
        # selected_features = [feature for feature, (_, p) in ks_results.items() if p > 0.05]
        # self.train_data = self.train_data[selected_features]
        # self.test_data = self.test_data[selected_features]

    def generate_feature_distribution_report(self, chunk_size=50) -> pd.DataFrame:
        """
        生成训练集和测试集的特征分布对比报告，保存为CSV
        """
        # 确保特征列一致
        common_cols = list(set(self.train_data.columns) & set(self.test_data.columns))
        common_cols = [c for c in self.train_data.columns if c in common_cols]

        sample_train = self.train_data.sample(frac=0.5)
        sample_test = self.test_data.sample(frac=0.5)
        
        report = []
        """
        for feature in common_cols:
            # 数值型特征处理
            if pd.api.types.is_numeric_dtype(self.train_data[feature]):
                # print("处理数值型特征：", feature)
                # 训练集统计量
                train_mean = self.train_data[feature].mean()
                train_std = self.train_data[feature].std()
                train_q25, train_q50, train_q75 = self.train_data[feature].quantile([0.25, 0.5, 0.75])
                
                # 测试集统计量
                test_mean = self.test_data[feature].mean()
                test_std = self.test_data[feature].std()
                test_q25, test_q50, test_q75 = self.test_data[feature].quantile([0.25, 0.5, 0.75])
                
                # KS检验
                ks_stat, ks_p = ks_2samp(
                    self.train_data[feature].dropna(),
                    self.test_data[feature].dropna()
                )
                
                report.append({
                    "feature": feature,
                    "data_type": "numeric",
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "train_q25": train_q25,
                    "train_q50": train_q50,
                    "train_q75": train_q75,
                    "test_mean": test_mean,
                    "test_std": test_std,
                    "test_q25": test_q25,
                    "test_q50": test_q50,
                    "test_q75": test_q75,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p
                })
                
            # 类别型特征处理（假设为字符串或整数类别）
            else:
                # print("处理类别型特征：", feature)
                # 训练集类别分布
                train_counts = self.train_data[feature].value_counts(normalize=True).to_dict()
                # 测试集类别分布
                test_counts = self.test_data[feature].value_counts(normalize=True).to_dict()
                
                # 卡方检验
                contingency_table = pd.crosstab(
                    pd.concat([self.train_data[feature], self.test_data[feature]]),
                    pd.Series(["Train"]*len(self.train_data) + ["Test"]*len(self.test_data))
                )
                chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
                
                report.append({
                    "feature": feature,
                    "data_type": "categorical",
                    "train_top3_categories": str(list(train_counts.keys())[:3]),
                    "train_top3_freq": str([round(v,3) for v in list(train_counts.values())[:3]]),
                    "test_top3_categories": str(list(test_counts.keys())[:3]),
                    "test_top3_freq": str([round(v,3) for v in list(test_counts.values())[:3]]),
                    "chi2_stat": chi2_stat,
                    "chi2_p": chi2_p
                })
        """

        for i in range(0, len(common_cols), chunk_size):
            chunk_cols = common_cols[i:i+chunk_size]
            chunk_report = []

            for feature in tqdm(chunk_cols, desc=f"Processing chunk {i//chunk_size}"):
                # 数值型特征处理
                if pd.api.types.is_numeric_dtype(sample_train[feature]):
                    # 训练集统计量
                    train_mean = sample_train[feature].mean()
                    train_std = sample_train[feature].std()
                    train_q25, train_q50, train_q75 = sample_train[feature].quantile([0.25, 0.5, 0.75])
                    
                    # 测试集统计量
                    test_mean = sample_test[feature].mean()
                    test_std = sample_test[feature].std()
                    test_q25, test_q50, test_q75 = sample_test[feature].quantile([0.25, 0.5, 0.75])
                    
                    # KS检验
                    ks_stat, ks_p = ks_2samp(
                        sample_train[feature].dropna(),
                        sample_test[feature].dropna()
                    )
                    
                    report.append({
                        "feature": feature,
                        "data_type": "numeric",
                        "train_mean": train_mean,
                        "train_std": train_std,
                        "train_q25": train_q25,
                        "train_q50": train_q50,
                        "train_q75": train_q75,
                        "test_mean": test_mean,
                        "test_std": test_std,
                        "test_q25": test_q25,
                        "test_q50": test_q50,
                        "test_q75": test_q75,
                        "ks_stat": ks_stat,
                        "ks_p": ks_p
                    })
                
                # 类别型特征处理
                else:
                    # 优化：仅处理低基数类别特征
                    n_unique_train = sample_train[feature].nunique()
                    n_unique_test = sample_test[feature].nunique()
                    # 跳过高基数特征
                    if n_unique_train > 1000 or n_unique_test > 1000:
                        print(f"Skipping high-cardinality feature: {feature}")
                        continue

                    # 训练集类别分布
                    train_counts = sample_train[feature].value_counts(normalize=True).to_dict()
                    # 测试集类别分布
                    test_counts = sample_test[feature].value_counts(normalize=True).to_dict()
                    
                    # 卡方检验
                    contingency_table = pd.crosstab(
                        pd.concat([sample_train[feature], sample_test[feature]]),
                        pd.Series(["Train"]*len(sample_train) + ["Test"]*len(sample_test))
                    )
                    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
                    
                    report.append({
                        "feature": feature,
                        "data_type": "categorical",
                        "train_top3_categories": str(list(train_counts.keys())[:3]),
                        "train_top3_freq": str([round(v,3) for v in list(train_counts.values())[:3]]),
                        "test_top3_categories": str(list(test_counts.keys())[:3]),
                        "test_top3_freq": str([round(v,3) for v in list(test_counts.values())[:3]]),
                        "chi2_stat": chi2_stat,
                        "chi2_p": chi2_p
                    })

            report.extend(chunk_report)
        
        # 转换为DataFrame并保存
        report_df = pd.DataFrame(report)
        report_df.to_csv("feature_distribution_report.csv", index=False)
        return report_df

    def adversarial_validation(self):
        """
        对抗性验证：用模型判断样本来自训练集还是测试集
        """
        """
        # 合并数据并打标签
        train_data = self.train_data.copy().assign(source=0)
        test_data = self.test_data.copy().assign(source=1)
        combined = pd.concat([train_data, test_data]).sample(frac=1.0)
        
        # 准备特征和标签
        X = combined.drop(columns=["source", "label"], errors="ignore")
        y = combined["source"]
        
        # 训练分类器
        model = LGBMClassifier()
        scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        
        print(f"Adversarial Validation AUC: {np.mean(scores):.3f}")
        if np.mean(scores) > 0.7:
            print("警告：训练集和测试集分布差异显著！")"""
        
        # ========== 分块合并优化 ==========
        chunk_size = 100_000  # 根据可用内存调整（建议设置为可用内存的20%）
        combined_chunks = []
        # 分块合并训练集和测试集（避免一次性加载全部数据）
        for i in tqdm(range(0, max(len(self.train_data), len(self.test_data)), chunk_size), desc="分块合并"):
            train_chunk = self.train_data.iloc[i:i+chunk_size].copy().assign(source=0)
            test_chunk = self.test_data.iloc[i:i+chunk_size].copy().assign(source=1)
            combined_chunks.append(pd.concat([train_chunk, test_chunk]))
        combined = pd.concat(combined_chunks)
        del combined_chunks  # 及时释放中间内存
        gc.collect()
        
        # 准备特征和标签
        print("准备特征和标签...")
        X = combined.drop(columns=["source"])
        y = combined["source"].astype('int8')  # 标签用int8存储
        del combined  # 及时释放原始数据
        gc.collect()

        # ========== 手动交叉验证 ==========
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"Processing fold {fold_idx}/5...")
            
            # 按索引切片避免数据复制
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # ========== 轻量级模型配置 ==========
            model = LGBMClassifier(
                n_estimators=500,          # 减少树的数量
                num_leaves=31,             # 限制叶子节点数
                max_depth=5,               # 降低树深度
                learning_rate=0.1,         # 提高学习率
                verbosity=-1,              # 关闭日志
                n_jobs=-1,                  # 限制并行线程
                random_state=42
            )

            # ========== 增量训练与评估 ==========
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=20)]
            )
            
            # 增量预测（减少内存占用）
            y_pred = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
            fold_auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(fold_auc)
            
            # 及时释放内存
            del X_train, X_val, y_train, y_val, model
            _ = gc.collect()  # 强制垃圾回收

        # ========== 结果输出 ==========
        final_auc = np.mean(auc_scores)
        print(f"Adversarial Validation AUC: {final_auc:.3f}")
        if final_auc > 0.7:
            print("警告：训练集和测试集分布差异显著！")

        # 最终内存清理
        del X, y
        gc.collect()

    def _adversarial_feature_selection(self):
        """
        对抗性验证筛选特征：移除区分训练/测试集能力强的特征
        """
        """
        # 合并数据并打标签
        train_data = self.train_data.copy().assign(source=0)
        test_data = self.test_data.copy().assign(source=1)
        combined = pd.concat([train_data, test_data])#.sample(frac=1.0)
        
        # 训练对抗性模型
        X = combined.drop(columns=["source"])
        y = combined["source"]
        
        adv_model = LGBMClassifier()
        # y_pred = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
        
        # 计算特征重要性
        adv_model.fit(X, y)
        self.plot_feature_importance(adv_model, X, file_name='feature_importance_adversarial.csv')
        importance = pd.Series(adv_model.feature_importances_, index=X.columns)
        
        # 标记高重要性特征（筛选重要性前10%）
        threshold = importance.quantile(0.9)
        self.adversarial_features = importance[importance > threshold].index.tolist()
        print("*******************************************************")
        print(f"Adversarial Feature Selection: {len(self.adversarial_features)} features selected")
        print(self.adversarial_features)
        
        # 移除对抗性特征
        self.train_data = self.train_data.drop(columns=self.adversarial_features)
        self.test_data = self.test_data.drop(columns=self.adversarial_features)"""

        # ========== 分块合并数据 ==========
        combined_chunks = []
        chunk_size = 100_000  # 根据内存调整
        # 分块合并训练集和测试集
        for i in range(0, len(self.train_data), chunk_size):
            train_chunk = self.train_data.iloc[i:i+chunk_size].assign(source=0)
            test_chunk = self.test_data.sample(n=chunk_size).assign(source=1)  # 随机匹配测试数据
            combined_chunks.append(pd.concat([train_chunk, test_chunk]))
        combined = pd.concat(combined_chunks)
        del combined_chunks  # 及时释放中间内存
        gc.collect()
        
        # 训练对抗性模型
        X = combined.drop(columns=["source"])
        y = combined["source"].astype('int8')  # 标签用int8存储
        del combined  # 及时释放原始数据
        gc.collect()

        # 初始化特征重要性存储
        feature_importance = pd.Series(0, index=X.columns)
        
        # 手动实现 5 折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"Processing fold {fold}/5...")
            
            # 分折数据（避免复制整个数据集）
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # ========== 轻量级模型配置 ==========
            adv_model = LGBMClassifier(
                n_estimators=500,          # 减少树的数量
                num_leaves=31,             # 限制叶子节点数
                max_depth=5,              # 降低树深度
                learning_rate=0.1,         # 提高学习率加速收敛
                verbosity=-1,              # 关闭日志输出
                n_jobs=-1                  # 限制并行线程数
            )
            
            # 训练并获取特征重要性
            adv_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=20)])
            fold_importance = pd.Series(adv_model.feature_importances_, index=X.columns)
            feature_importance += fold_importance
            
            # 增量预测（减少内存占用）
            y_pred = adv_model.predict_proba(X_val, num_iteration=adv_model.best_iteration_)[:, 1]
            fold_auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(fold_auc)
            
            # 手动释放内存
            del X_train, X_val, y_train, y_val, adv_model
            gc.collect()
        
        # 计算平均特征重要性
        feature_importance /= 5
        
        # ========== 增量式特征筛选 ==========
        final_auc = np.mean(auc_scores)
        print(f"Adversarial Validation AUC: {final_auc:.3f}")
        if final_auc > 0.7:
            print("警告：训练集和测试集分布差异显著！")
        threshold = feature_importance.quantile(0.9)
        self.adversarial_features = feature_importance[feature_importance > threshold].index.tolist()
        print(f"\n对抗性特征筛选结果（前10%）：{len(self.adversarial_features)}个特征")
        print(list(self.adversarial_features))

        # 移除特征
        self.train_data = self.train_data.drop(columns=self.adversarial_features)
        self.test_data = self.test_data.drop(columns=self.adversarial_features)
        # chunk_size = 10_000
        # for i in range(0, len(self.train_data), chunk_size):
        #     self.train_data.iloc[i:i+chunk_size] = self.train_data.iloc[i:i+chunk_size].drop(columns=self.adversarial_features)
        # for i in range(0, len(self.test_data), chunk_size):
        #     self.test_data.iloc[i:i+chunk_size] = self.test_data.iloc[i:i+chunk_size].drop(columns=self.adversarial_features)
        
        # 最终内存整理
        self.train_data = self.train_data.copy()
        self.test_data = self.test_data.copy()

    def train_with_cv(self, n_splits=5):
        """
        使用五折交叉验证训练模型并输出每一折的评估指标
        """
        # 合并正负样本数据
        train_all = pd.concat([self.train_pos, self.train_neg])
        train_all.drop("LogTime", axis=1, inplace=True)
        train_all = train_all.sort_index(axis=1)

        # 使用筛选后的特征
        X = train_all[self.train_data.columns]
        y = train_all["label"]

        # 初始化五折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        auc_scores = []
        f1_scores = []
        confusion_matrices = []

        # 五折交叉验证
        for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
            print(f"Training fold {fold}...")
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            self.model = LGBMClassifier(**self.model_params)

            # 训练模型
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=20)])

            # 预测概率
            y_pred_prob = self.model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_prob >= 0.5).astype(int)

            # 计算 AUC、F1 score 和 混淆矩阵
            auc = roc_auc_score(y_val, y_pred_prob)
            f1 = f1_score(y_val, y_pred)
            cm = confusion_matrix(y_val, y_pred)
            class_report_lgb = classification_report(y_val, y_pred)

            # 输出当前折的评估指标
            print(f"Fold {fold} - AUC: {auc:.4f}, F1 score: {f1:.4f}")
            print(f"Confusion Matrix for fold {fold}:\n{cm}")
            print(f"Classification Report for fold {fold}:\n{class_report_lgb}")

            # 保存当前折的结果
            auc_scores.append(auc)
            f1_scores.append(f1)
            confusion_matrices.append(cm)

        # 输出五折的平均评估指标
        print("\nAverage Scores across all folds:")
        print(f"Average AUC: {np.mean(auc_scores):.4f}")
        print(f"Average F1 score: {np.mean(f1_scores):.4f}")
        print(f"Average Confusion Matrix:\n{np.mean(confusion_matrices, axis=0)}")

        # 训练模型
        self.model = LGBMClassifier(**self.model_params)
        self.model.fit(X, y)

        # 绘制特征重要性
        self.plot_feature_importance(self.model, X, file_name='feature_importance.csv')

    def plot_feature_importance(self, model_used, X, file_name):
        """
        绘制特征重要性图并将特征重要性保存为CSV文件

        :param X: 训练数据特征
        """
        # 获取特征的重要性
        feature_importance = model_used.feature_importances_

        # 获取特征名称
        feature_names = X.columns

        # 创建数据框来存储特征及其重要性
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })

        # 按照重要性排序
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        # 保存为 CSV 文件
        feature_importance_df.to_csv(file_name, index=False)

        # 绘制特征重要性条形图
        # plt.figure(figsize=(10, 6))
        # plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
        # plt.xlabel('Importance')
        # plt.title('Feature Importance')
        # plt.gca().invert_yaxis()  # 反转Y轴，重要性高的特征在上面
        # plt.show()

        print("Feature importance saved to {}".format(file_name))

    def predict_proba(self) -> Dict[str, List]:
        """
        预测测试数据每个样本被预测为正类的概率, 并返回结果

        :return: 每个样本被预测为正类的概率, 结果是一个 dict, key 为 sn_name, value 为预测结果列表
        """
        result = {}
        for file in os.listdir(self.test_data_path):
            test_df = feather.read_dataframe(os.path.join(self.test_data_path, file))
            # 创建类型转换字典
            dtype_mapping = {}
            for col in test_df.columns:
                if any(kwd in col for kwd in ["_avg_", "_std_"]) or col.startswith("log_happen_frequency"):
                    dtype_mapping[col] = "float32"
                else:
                    dtype_mapping[col] = "int32"
            # 应用类型转换
            test_df = test_df.astype(dtype_mapping)

            # test_df["sn_name"] = test_df.index.get_level_values(0)
            # test_df["log_time"] = test_df.index.get_level_values(1)
            sn_name_column = pd.DataFrame({'sn_name': test_df.index.get_level_values(0)}, index=test_df.index)
            log_time_column = pd.DataFrame({'log_time': test_df.index.get_level_values(1)}, index=test_df.index)
            # 使用 pd.concat 一次性添加新列
            test_df = pd.concat([test_df, sn_name_column, log_time_column], axis=1)

            test_df = test_df[self.model.feature_name_]
            # print("测试使用特征：", list(test_df.columns))
            predict_result = self.model.predict_proba(test_df)

            index_list = list(test_df.index)
            for i in tqdm(range(len(index_list))):
                p_s = predict_result[i][1]

                # 过滤低概率样本, 降低预测结果占用的内存
                if p_s < 0.1:
                    continue

                sn = index_list[i][0]
                sn_t = datetime.fromtimestamp(index_list[i][1])
                result.setdefault(sn, [])
                result[sn].append((sn_t, p_s))
        return result

    def predict(self, threshold: int = 0.5) -> Dict[str, List]:
        """
        获得特定阈值下的预测结果

        :param threshold: 阈值, 默认为 0.5
        :return: 按照阈值筛选后的预测结果, 结果是一个字典, key 为 sn_name, value 为时间戳列表
        """

        # 获取预测概率结果
        result = self.predict_proba()

        # 将预测结果按照阈值进行筛选
        result = {
            sn: [int(sn_t.timestamp()) for sn_t, p_s in pred_list if p_s >= threshold]
            for sn, pred_list in result.items()
        }

        # 过滤空预测结果, 并将预测结果按照时间进行排序
        result = {
            sn: sorted(pred_list) for sn, pred_list in result.items() if pred_list
        }

        return result



if __name__ == "__main__":
    sn_type = "B"  # SN 类型, A 或 B, 这里以 A 类型为例
    test_stage = 1  # 测试阶段, 1 或 2, 这里以 Stage 1 为例

    # 根据测试阶段设置测试数据的时间范围
    if test_stage == 1:
        test_data_range: tuple = ("2024-06-01", "2024-08-01")  # 第一阶段测试数据范围
    else:
        test_data_range: tuple = ("2024-08-01", "2024-10-01")  # 第二阶段测试数据范围

    # 初始化配置类 Config，设置数据路径、特征路径、训练数据路径、测试数据路径等
    config = Config(
        data_path=os.path.join("stage1_feather", f"type_{sn_type}"),  # 原始数据集路径
        feature_path=os.path.join(
            "combined_sn_feature", f"type_{sn_type}"  # 生成的特征数据路径
        ),
        train_data_path=os.path.join(
            "train_data", f"type_{sn_type}"  # 生成的训练数据路径
        ),
        test_data_path=os.path.join(
            "test_data", f"type_{sn_type}_{test_stage}"  # 生成的测试数据路径
        ),
        test_data_range=test_data_range,  # 测试数据时间范围
        ticket_path="stage1_feather/ticket.csv",  # 维修单路径
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] 程序开始运行")

    # 初始化特征工厂类 FeatureFactory，用于处理 SN 文件并生成特征
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在处理SN文件... ***************************")
    feature_factory = FeatureFactory(config)
    feature_factory.process_all_sn()  # 处理所有 SN 文件

    # 初始化正样本数据生成器，生成并保存正样本数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在生成正样本数据... ***************************")
    positive_data_generator = PositiveDataGenerator(config)
    positive_data_generator.generate_and_save_data()

    # 初始化负样本数据生成器，生成并保存负样本数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在生成负样本数据... ***************************")
    negative_data_generator = NegativeDataGenerator(config)
    negative_data_generator.generate_and_save_data()

    # 初始化测试数据生成器，生成并保存测试数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在生成测试数据... ***************************")
    test_data_generator = TestDataGenerator(config)
    test_data_generator.generate_and_save_data()

    # 初始化模型类 MFPmodel，加载训练数据并训练模型
    model = MFPmodel(config)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在加载训练数据和测试数据... ***************************")
    model.load_train_data()  # 加载训练数据
    model.load_test_data()   # 新增：加载测试数据

    # 新增：对比特征分布
    # model.plot_feature_distributions()
    # 生成并保存分布报告
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在生成特征分布报告... ***************************")
    report_df = model.generate_feature_distribution_report()
    # 对抗性验证
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{current_time}] *************************** 正在进行对抗性验证... ***************************")
    # model.adversarial_validation()

    # 对抗性验证筛选特征
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{current_time}] *************************** 正在进行对抗性验证筛选特征... ***************************")
    # model._adversarial_feature_selection()

    # 使用五折交叉验证训练并评估模型
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] *************************** 正在训练模型... ***************************")
    model.train_with_cv(n_splits=5)
    
    result = model.predict() # 使用训练好的模型进行预测

    # 将预测结果转换为提交格式
    submission = []
    for sn in result:  # 遍历每个 SN 的预测结果
        for timestamp in result[sn]:  # 遍历每个时间戳
            submission.append([sn, timestamp, sn_type])  # 添加 SN 名称、预测时间戳和 SN 类型

    # 将提交数据转换为 DataFrame 并保存为 CSV 文件
    submission = pd.DataFrame(
        submission, columns=["sn_name", "prediction_timestamp", "serial_number_type"]
    )
    current_time = datetime.now().strftime("%m%d%H%M")
    submission_file_name = f"submission{current_time}.csv"
    submission.to_csv(submission_file_name, index=False, encoding="utf-8")

    print()
