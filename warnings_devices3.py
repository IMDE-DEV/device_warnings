'''
第三版非示范沟，在第二版的基础上，将雨量预警的结果单独标注出来，雨量初始权重为15，泥位初始权重为10
'''


import pandas as pd
import numpy as np
import os
import math
from datetime import datetime


# ================= 1. 雨量算法引擎 (Rain Risk Scorer) =================

class RainRiskScorer:
    def __init__(self):
        # API 衰减配置
        self.K_DECAY = 0.96  # 小时衰减系数 (24小时后衰减至约 0.37)

    def calculate_single_score(self, current_rain, past_history_list):
        """计算单次得分 (基于行业标准的有效雨量和径流系数)"""
        if not past_history_list:
            past_history_list = [0.0]

        # 1. 计算前期影响雨量 (Antecedent Precipitation, Pa)
        pa_val = 0.0
        for i, rain in enumerate(past_history_list):
            weight = math.pow(self.K_DECAY, i + 1)
            pa_val += rain * weight

        # 2. 行业标准：依据 Pa 判定土壤饱和度，获取产流系数 (alpha)
        if pa_val < 10.0:
            alpha = 0.3  # 干燥 (吸收多)
        elif pa_val < 30.0:
            alpha = 0.6  # 湿润
        else:
            alpha = 0.95  # 饱和 (几乎不吸收，极易致灾)

        # 3. 计算综合致灾雨量 (Effective Rainfall, R_eff)
        r_eff = current_rain + (alpha * pa_val)

        # 4. 阶梯式风险映射 (符合气象预警规范的分段函数)
        if r_eff < 20.0:
            final_score = (r_eff / 20.0) * 40.0  # 0 - 40分 (蓝警以下)
        elif r_eff < 40.0:
            final_score = 40.0 + ((r_eff - 20.0) / 20.0) * 20.0  # 40 - 60分 (黄警区间)
        elif r_eff < 70.0:
            final_score = 60.0 + ((r_eff - 40.0) / 30.0) * 20.0  # 60 - 80分 (橙警区间)
        else:
            final_score = 80.0 + ((r_eff - 70.0) / 30.0) * 20.0  # 80 - 100分 (红警区间)

        return min(100.0, final_score)

    def process_dataframe(self, df, target_time):
        """
        处理DataFrame (时间切片求和法，无视频率)
        返回: (雨量风险分, 数据完整率, 当前1h降雨量)
        """
        try:
            df_past = df[df['datetime'] <= target_time]
            if df_past.empty:
                return None

            hourly_sums = []
            valid_hours_24 = 0

            # 往前倒推 49 个小时的时间切片 (索引0为当前1h，1~48为过去48h)
            for i in range(49):
                end_t = target_time - pd.Timedelta(hours=i)
                start_t = end_t - pd.Timedelta(hours=1)

                # 提取该小时段内的所有数据并求和 (pre 列本身为增量，直接求和代表真实降雨)
                slice_df = df_past[(df_past['datetime'] > start_t) & (df_past['datetime'] <= end_t)]

                if not slice_df.empty:
                    rain_sum = float(slice_df['val'].sum())
                    if i < 24: valid_hours_24 += 1  # 统计近24h的数据完整性
                else:
                    rain_sum = 0.0

                hourly_sums.append(rain_sum)

            # 数据完整率 (惩罚系数)
            completeness = max(0.01, min(1.0, valid_hours_24 / 24.0))

            current_rain = hourly_sums[0]
            past_history = hourly_sums[1:]  # 过去48小时列表，按照倒序排列，完全吻合底层的Pa衰减公式

            score = self.calculate_single_score(current_rain, past_history)

            return score, completeness, current_rain

        except Exception:
            return None


# ================= 2. 泥位算法引擎 (Mud Risk Scorer) =================

class MudRiskScorer:
    def __init__(self):
        self.WINDOW_SIZE = 150  # 滚动窗口大小 (计算历史均值和噪声用)
        self.MIN_STD = 0.01  # 最小标准差防噪 (防止完全死水除以0)

    def process_dataframe(self, df, target_time):
        """处理DataFrame，返回: (泥位风险分, 当前背景噪声sigma)"""
        try:
            df = df.sort_values('datetime')
            df_past = df[df['datetime'] <= target_time]

            if len(df_past) < 2:
                return None

            val = df_past['val']

            # 1. 计算滑动统计量
            rolling_mean = val.rolling(window=self.WINDOW_SIZE, min_periods=1).mean()
            rolling_std = val.rolling(window=self.WINDOW_SIZE, min_periods=1).std()

            # 2. 噪声抑制底限
            rolling_std = rolling_std.apply(lambda x: max(x, self.MIN_STD))

            # 3. 计算 Z-Score 异常偏离度
            z_scores = (val - rolling_mean).abs() / rolling_std

            # 4. 获取目标时刻的最新的 Z-Score 和 Sigma
            current_z = z_scores.iloc[-1]
            current_sigma = rolling_std.iloc[-1]

            if pd.isna(current_z) or pd.isna(current_sigma):
                return None

            # 5. 映射分数 (保持原版判定逻辑不变)
            raw_score = (current_z - 2.0) * 20.0
            final_score = max(0.0, min(100.0, raw_score))

            return final_score, current_sigma

        except Exception:
            return None


# ================= 3. 数据加载中间件 (Data Processor) =================

class DataProcessor:
    def __init__(self, data_dirs):
        self.dirs = data_dirs

    def load_data(self, device_id, dtype):
        dir_path = self.dirs.get(dtype)
        if not dir_path: return None
        path = os.path.join(dir_path, f"{device_id}.csv")
        if not os.path.exists(path): return None

        try:
            # 自动处理不同编码
            try:
                df = pd.read_csv(path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='gbk')

            df.columns = df.columns.str.lower()

            if dtype == 'YL':
                # 【修改点1】：替换 pre_24h，直接读取 pre 增量列
                if 'pre' in df.columns:
                    df['val'] = pd.to_numeric(df['pre'], errors='coerce')
                else:
                    return None
            else:
                if 'value' in df.columns:
                    df['val'] = pd.to_numeric(df['value'], errors='coerce')
                else:
                    return None

            df['datetime'] = pd.to_datetime(df['observtime'], errors='coerce')
            df = df.dropna(subset=['datetime', 'val'])
            return df[['datetime', 'val']]
        except Exception:
            return None


# ================= 4. 融合决策大脑 (Risk Fusion Engine) =================

class RiskFusionEngine:
    def __init__(self, device_table_path, data_dirs):
        self.device_table_path = device_table_path
        self.processor = DataProcessor(data_dirs)
        self.rain_scorer = RainRiskScorer()
        self.mud_scorer = MudRiskScorer()

    def run(self, target_time=None):
        if target_time is None:
            target_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        else:
            target_time = pd.to_datetime(target_time).replace(minute=0, second=0, microsecond=0)

        print(f"计算目标时刻: {target_time}")

        try:
            devices_df = pd.read_csv(self.device_table_path, encoding='utf-8-sig')
        except Exception as e:
            raise ValueError("无法读取设备表")

        # 筛选非示范沟且在线的设备
        target_devices = devices_df[
            (devices_df['demo'] == 0) &
            (devices_df['type'].isin(['NW', 'YL'])) &
            (devices_df['is_online'] == 1)
            ]

        results = []
        grouped = target_devices.groupby('basinCode')
        print(f"开始处理，共 {len(grouped)} 个非示范沟流域...")

        for basin_code, group in grouped:
            rain_results = []  # 存储 (score, dynamic_weight, current_rain, device_id)
            mud_results = []  # 存储 (score, dynamic_weight)

            for _, row in group.iterrows():
                device_id = row['device']
                dtype = row['type']
                init_weight = float(row.get('init_weight', 15.0 if dtype == 'YL' else 10.0))

                if dtype == 'YL':
                    df = self.processor.load_data(device_id, 'YL')
                    if df is not None:
                        res = self.rain_scorer.process_dataframe(df, target_time)
                        if res is not None:
                            score, completeness, rain_1h = res
                            dynamic_weight = init_weight * completeness
                            # 把单站降雨量一起记下来
                            rain_results.append((score, dynamic_weight, rain_1h, device_id))

                elif dtype == 'NW':
                    df = self.processor.load_data(device_id, 'NW')
                    if df is not None:
                        res = self.mud_scorer.process_dataframe(df, target_time)
                        if res is not None:
                            score, current_sigma = res
                            epsilon = 0.05
                            confidence = epsilon / (current_sigma + epsilon)
                            dynamic_weight = init_weight * confidence
                            mud_results.append((score, dynamic_weight))

            # ================= 全动态置信度加权融合 =================
            sum_mud_weighted_scores = 0.0
            sum_mud_weights = 0.0
            final_mud = 0.0

            if mud_results:
                final_mud = max([x[0] for x in mud_results])
                for s_mud, w_mud in mud_results:
                    sum_mud_weighted_scores += s_mud * w_mud
                    sum_mud_weights += w_mud

            final_rain = 0.0
            rain_details = []

            if rain_results:
                # 代理策略 (取得分最高的那台雨量计代表流域)
                best_rain = max(rain_results, key=lambda x: x[0])
                final_rain = best_rain[0]
                rain_weight = best_rain[1]

                sum_weighted_scores = sum_mud_weighted_scores + final_rain * rain_weight
                sum_weights = sum_mud_weights + rain_weight

                # 记录所有雨量计明细
                for r in rain_results:
                    rain_details.append(f"{r[3]}:{round(r[2], 1)}mm")
            else:
                sum_weighted_scores = sum_mud_weighted_scores
                sum_weights = sum_mud_weights

            # 综合最终得分计算
            weighted_score = sum_weighted_scores / sum_weights if sum_weights > 0 else 0.0

            # 【修改点3：判断预警是否由雨量触发】
            # 我们通过计算“纯泥位计分数”来进行对比剥离。
            mud_only_score = sum_mud_weighted_scores / sum_mud_weights if sum_mud_weights > 0 else 0.0

            def get_warning_level(sc):
                if sc >= 80:
                    return 4, "红色"
                elif sc >= 60:
                    return 3, "橙色"
                elif sc >= 40:
                    return 2, "黄色"
                elif sc >= 20:
                    return 1, "蓝色"
                else:
                    return 0, "正常"

            final_lvl, final_lvl_str = get_warning_level(weighted_score)
            mud_lvl, _ = get_warning_level(mud_only_score)

            # 如果最终报警了(>0)，且等级高于纯泥位的等级，说明是雨量强行拉高的报警！
            if final_lvl > 0 and final_lvl > mud_lvl:
                final_lvl_str += " (由雨量预警导致)"

            results.append({
                '流域编号': basin_code,
                '各站1h降雨明细': " | ".join(rain_details) if rain_details else "无数据",
                '雨量分': round(final_rain, 1),
                '泥位分': round(final_mud, 1),
                '综合预警分': round(weighted_score, 1),
                '最终预警等级': final_lvl_str,
                '有效雨量计': len(rain_results),
                '有效泥位计': len(mud_results)
            })

        return pd.DataFrame(results)


# ================= 5. 主程序启动入口 =================
if __name__ == "__main__":

    # 将字典统一化管理 (与示范沟保持风格一致)
    DATA_DIRS = {
        'YL': "雨量计数据",
        'NW': "泥位计数据"
    }
    DEVICE_TABLE = "code_test/流域对应表.csv"

    try:
        # 实例化融合引擎
        engine = RiskFusionEngine(DEVICE_TABLE, DATA_DIRS)

        # 设定计算目标时刻
        target = pd.to_datetime("2025-07-15 18:00:00")

        # 执行运算
        result_df = engine.run(target_time=target)

        print("\n================== 预警计算结果 (非示范沟) ==================")
        if result_df.empty:
            print("没有计算出任何结果，请检查数据文件匹配情况。")
        else:
            print(result_df.to_string(index=False))
            result_df.to_csv("最终预警结果_非示范沟.csv", index=False, encoding='utf-8-sig')
            print("\n✔️ 结果已成功保存至 '最终预警结果_非示范沟.csv'")

    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        import traceback

        traceback.print_exc()