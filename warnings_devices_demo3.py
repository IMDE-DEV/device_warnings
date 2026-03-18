'''
第三版示范沟，在第二版的基础上，将雨量预警的结果单独标注出来，不算做临灾预警指标
'''


import pandas as pd
import numpy as np
import os
from datetime import datetime

# 1. ===== 导入地震动模块 API =====
from seismic_engine import get_seismic_status

# ================= 0. 全局常量定义 =================
LEVEL_MAP = {0: "正常", 1: "蓝色", 2: "黄色", 3: "橙色", 4: "红色"}


# ================= 1. 示范沟专属雨量算法 (双时间窗) =================
class DemoRainScorer:
    def __init__(self):
        self.HAS_RAIN_6H_THRESHOLD = 5.0
        self.THRESHOLDS = {
            'blue': 8.0,
            'yellow': 18.0,
            'orange': 28.0,
            'red': 45.0
        }

    def process_dataframe(self, df, target_time):
        try:
            df_past = df[df['datetime'] <= target_time]
            if df_past.empty:
                return False, 0, 0.0

            # 6小时降雨判定“是否有雨”
            start_time_6h = target_time - pd.Timedelta(hours=6)
            df_6h = df_past[df_past['datetime'] > start_time_6h]
            rain_6h_sum = float(df_6h['val'].sum()) if not df_6h.empty else 0.0
            has_rain = rain_6h_sum >= self.HAS_RAIN_6H_THRESHOLD

            # 1小时降雨判定“单站预警等级”
            start_time_1h = target_time - pd.Timedelta(hours=1)
            df_1h = df_past[df_past['datetime'] > start_time_1h]
            rain_1h_sum = float(df_1h['val'].sum()) if not df_1h.empty else 0.0

            if rain_1h_sum >= self.THRESHOLDS['red']:
                level = 4
            elif rain_1h_sum >= self.THRESHOLDS['orange']:
                level = 3
            elif rain_1h_sum >= self.THRESHOLDS['yellow']:
                level = 2
            elif rain_1h_sum >= self.THRESHOLDS['blue']:
                level = 1
            else:
                level = 0

            return has_rain, level, rain_1h_sum
        except Exception:
            return False, 0, 0.0


# ================= 2. 泥位算法引擎 (高灵敏度) =================
class MudRiskScorer:
    def __init__(self):
        self.WINDOW_SIZE = 150
        self.MIN_STD = 0.01

    def process_dataframe(self, df, target_time):
        try:
            df_past = df[df['datetime'] <= target_time].sort_values('datetime')
            if len(df_past) < 2: return 0

            val = df_past['val']
            rolling_mean = val.rolling(window=self.WINDOW_SIZE, min_periods=1).mean()
            rolling_std = val.rolling(window=self.WINDOW_SIZE, min_periods=1).std()
            rolling_std = rolling_std.apply(lambda x: max(x, self.MIN_STD))

            z_scores = (val - rolling_mean).abs() / rolling_std
            current_z = z_scores.iloc[-1]
            if pd.isna(current_z): return 0

            # 高灵敏度公式
            raw_score = (current_z - 1.5) * 20.0
            score = max(0.0, min(100.0, raw_score))

            if score >= 80:
                return 4
            elif score >= 60:
                return 3
            elif score >= 40:
                return 2
            elif score >= 20:
                return 1
            else:
                return 0
        except Exception:
            return 0


# ================= 3. 新增设备 Mock 接口 =================
class SeismometerScorer:
    def __init__(self, root_folder, mapping_csv_path):
        self.root_folder = root_folder
        self.mapping_csv_path = mapping_csv_path

    def get_status(self, device_id, target_time):
        try:
            if not self.root_folder: return None
            # print("root_folder:", self.root_folder)
            device_folder = os.path.join(self.root_folder, str(device_id))
            # print("device_folder:", device_folder)
            if not os.path.exists(device_folder): return None

            status, minute_results = get_seismic_status(
                device_folder=device_folder,
                mapping_csv_path=self.mapping_csv_path,
                target_time=target_time,
                minutes=30,
                sampling_rate=100,
                mode='any_minute',
                return_minute_results=True
            )
            # print(f"地震动设备 {device_id} 在 {target_time} 的状态: {status}")
            # print("minute_results:",minute_results)
            return status
        except Exception:
            return None


class VideoScorer:
    def get_status(self, device_id, target_time):
        try:
            # TODO: 接入视频接口, 返回 (置信度, "高/中/低/无")
            return None
        except Exception:
            return None

        # ================= 4. 数据加载中间件 =================


class DataProcessor:
    def __init__(self, data_dirs):
        self.dirs = data_dirs

    def load_data(self, device_id, dtype):
        dir_path = self.dirs.get(dtype)
        if not dir_path: return None
        path = os.path.join(dir_path, f"{device_id}.csv")
        if not os.path.exists(path): return None

        try:
            try:
                df = pd.read_csv(path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='gbk')
            df.columns = df.columns.str.lower()

            if dtype == 'YL':
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


# ================= 5. 示范沟逻辑决策大脑 =================
class DemoRiskFusionEngine:
    def __init__(self, device_table_path, data_dirs):
        self.device_table_path = device_table_path
        self.processor = DataProcessor(data_dirs)
        self.rain_scorer = DemoRainScorer()
        self.mud_scorer = MudRiskScorer()
        dzd_root = data_dirs.get('DZD')
        self.dzd_scorer = SeismometerScorer(root_folder=dzd_root, mapping_csv_path=device_table_path)
        self.video_scorer = VideoScorer()

    def run(self, target_time=None):
        if target_time is None: target_time = datetime.now()

        try:
            devices_df = pd.read_csv(self.device_table_path, encoding='utf-8-sig')
        except Exception as e:
            raise ValueError(f"无法读取设备对应表: {e}")

        demo_devices = devices_df[(devices_df['demo'] == 1) & (devices_df['is_online'] == 1)]
        results = []

        for basin_code, group in demo_devices.groupby('basinCode'):
            basin_has_rain = False
            max_basin_rain = 0.0
            rain_details = []

            yl_levels, nw_levels, dzd_status_list, video_status_list = [], [], [], []
            dzd_warning = False

            for _, row in group.iterrows():
                device_id, dtype = row['device'], row['type']

                if dtype == 'YL':
                    df = self.processor.load_data(device_id, 'YL')
                    if df is not None:
                        has_rain, level, rain_1h = self.rain_scorer.process_dataframe(df, target_time)
                        if has_rain: basin_has_rain = True
                        yl_levels.append(level)
                        max_basin_rain = max(max_basin_rain, rain_1h)
                        rain_details.append(f"{device_id}:{round(rain_1h, 1)}mm")

                elif dtype == 'NW':
                    df = self.processor.load_data(device_id, 'NW')
                    if df is not None:
                        nw_levels.append(self.mud_scorer.process_dataframe(df, target_time))

                elif dtype == 'DZD':
                    status = self.dzd_scorer.get_status(device_id, target_time)
                    # print(f"地震动设备 {device_id} 在 {target_time} 的状态: {status}")

                    if status is not None:
                        dzd_status_list.append(status)
                        if status == 1: dzd_warning = True

                elif dtype == 'SP':
                    status = self.video_scorer.get_status(device_id, target_time)
                    if status is not None: video_status_list.append(status)

            # --- 统计各类数据 ---
            yl_counts = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
            for lvl in yl_levels: yl_counts[lvl] += 1
            max_nw_level = max(nw_levels) if nw_levels else 0

            # 累计高级别雨量计数量 (向下兼容)
            yl_ge_4 = yl_counts[4]
            yl_ge_3 = yl_counts[4] + yl_counts[3]
            yl_ge_2 = yl_counts[4] + yl_counts[3] + yl_counts[2]
            yl_ge_1 = yl_counts[4] + yl_counts[3] + yl_counts[2] + yl_counts[1]

            # --- 核心逻辑判定树 ---
            init_level = 0
            final_level = 0
            is_rain_only_trigger = False
            video_action = "无干预(无数据)"

            # 判断是否为“只有视频设备”的流域 (物理设备无有效数据)
            is_video_only_basin = (len(yl_levels) == 0 and len(nw_levels) == 0 and len(dzd_status_list) == 0)

            if is_video_only_basin and len(video_status_list) > 0:
                # 场景A：仅有视频设备
                best_video = max(video_status_list, key=lambda x: x[0])
                confidence, danger_str = best_video

                if confidence > 0.5:
                    if danger_str == "高":
                        final_level = 4
                    elif danger_str == "中":
                        final_level = 3
                    elif danger_str == "低":
                        final_level = 2
                    elif danger_str == "无":
                        final_level = 1
                    else:
                        final_level = 0
                    video_action = f"单一视频流域直接判定 (置信度{confidence}, 判{danger_str})"
                else:
                    final_level = 0
                    video_action = f"单一视频流域，置信度过低({confidence})，不报警"

                init_level = final_level

            else:
                # 场景B：常规多源设备联合判定

                # 1. 拆分“协同条件”
                is_synergy_4 = (dzd_warning and basin_has_rain) or (max_nw_level == 4 and basin_has_rain)
                is_synergy_3 = (dzd_warning and not basin_has_rain) or (max_nw_level == 4 and not basin_has_rain) or (
                            max_nw_level == 3 and basin_has_rain)
                is_synergy_2 = (max_nw_level == 2 and basin_has_rain) or (max_nw_level == 3 and not basin_has_rain)
                is_synergy_1 = (max_nw_level == 1 and basin_has_rain) or (max_nw_level == 2 and not basin_has_rain)

                # 2. 严格按优先级的决策梯队
                if is_synergy_4:
                    init_level = 4
                elif yl_ge_4 >= 1:
                    init_level = 4
                    is_rain_only_trigger = True
                elif is_synergy_3:
                    init_level = 3
                elif yl_ge_3 >= 1:
                    init_level = 3
                    is_rain_only_trigger = True
                elif is_synergy_2:
                    init_level = 2
                elif yl_ge_2 >= 1:
                    init_level = 2
                    is_rain_only_trigger = True
                elif is_synergy_1:
                    init_level = 1
                elif yl_ge_1 >= 1:
                    init_level = 1
                    is_rain_only_trigger = True

                final_level = init_level

                # 3. 视频二次干预升降级
                if video_status_list:
                    best_video = max(video_status_list, key=lambda x: x[0])
                    confidence, danger_str = best_video
                    if confidence > 0.5:
                        if danger_str in ["中", "高"]:
                            final_level = min(4, init_level + 1)
                            video_action = f"视频升级 (置信度{confidence}, 判{danger_str})"
                        elif danger_str in ["低", "无"]:
                            final_level = max(0, init_level - 1)
                            video_action = f"视频降级 (置信度{confidence}, 判{danger_str})"
                        else:
                            video_action = f"置信度达标但等级未知({danger_str})"
                    else:
                        video_action = f"视频置信度过低({confidence})，维持原判"

            # --- 组装输出结果 ---
            final_level_str = LEVEL_MAP[final_level]
            # 如果判定主要是因为雨量计单点暴雨引起的，且最终依然处于报警状态，则打上标签
            if is_rain_only_trigger and final_level > 0:
                final_level_str += " (由雨量预警导致)"

            results.append({
                '流域编号': basin_code,
                '是否有雨': '是' if basin_has_rain else '否',
                '流域最大1h降雨(mm)': round(max_basin_rain, 1),
                '各站1h降雨明细': " | ".join(rain_details) if rain_details else "无数据",
                '雨量计预警数(红/橙/黄/蓝)': f"{yl_counts[4]}/{yl_counts[3]}/{yl_counts[2]}/{yl_counts[1]}",
                '泥位最高等级': LEVEL_MAP[max_nw_level],
                '地震动是否报警': '是' if dzd_warning else '否',
                '初判预警等级': LEVEL_MAP[init_level],
                '视频干预说明': video_action,
                '最终预警等级': final_level_str,  # <--- 在这里加入了后缀
                '有效设备数(雨/泥/震/视)': f"{len(yl_levels)}/{len(nw_levels)}/{len(dzd_status_list)}/{len(video_status_list)}"
            })

        return pd.DataFrame(results)


# ================= 测试启动代码 =================
if __name__ == "__main__":
    DATA_DIRS = {
        'YL': "雨量计数据",
        'NW': "泥位计数据",
        'DZD': "地震动数据"
    }
    DEVICE_TABLE = r"D:\短期工作\2026_03\系统临灾预警\code_test\流域对应表.csv"

    try:
        engine = DemoRiskFusionEngine(DEVICE_TABLE, DATA_DIRS)
        target = pd.to_datetime("2025-07-04 20:50:00")

        print(f"计算目标时刻: {target}")
        result_df = engine.run(target_time=target)

        print("\n================== 示范沟预警结果 ==================")
        if result_df.empty:
            print("未找到示范沟设备或无有效数据。")
        else:
            print(result_df.to_string(index=False))
            result_df.to_csv("最终预警结果_示范沟.csv", index=False, encoding='utf-8-sig')
            print("\n✔️ 结果已成功保存至 '最终预警结果_示范沟.csv'")

    except Exception as e:
        print(f"\n❌ 程序主逻辑崩溃: {str(e)}")
        import traceback

        traceback.print_exc()