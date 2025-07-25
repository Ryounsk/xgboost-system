import pandas as pd
import numpy as np
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from datetime import datetime
import requests
import json
from bs4 import BeautifulSoup
import re

# B站API和请求头
BASE_URL = "https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all&pn={}&ps=50"
DETAIL_API = "https://api.bilibili.com/x/web-interface/view?bvid={}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": "https://www.bilibili.com"
}

# -------------------------- 1. 数据采集模块 --------------------------
class DataCollector:
    """B站排行榜数据采集模块，支持直接爬取和定时采集"""
    def __init__(self, save_path="bilibili_data/raw_data.csv"):
        self.save_path = save_path
        # 确保数据目录存在
        if not os.path.exists("bilibili_data"):
            os.makedirs("bilibili_data")

    def get_video_list(self, page_num):
        """获取B站排行榜视频列表"""
        url = BASE_URL.format(page_num)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["code"] == 0 and "data" in data and "list" in data["data"]:
                return data["data"]["list"]
            else:
                print(f"获取第{page_num}页视频列表失败: {data.get('message', '未知错误')}")
                return []
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            return []

    def parse_video_detail(self, bvid):
        """使用API解析视频详情，提取点赞量和播放量"""
        url = DETAIL_API.format(bvid)
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data["code"] == 0 and "data" in data:
                video_data = data["data"]
                return {
                    "title": video_data.get("title", "未知标题"),
                    "likes": video_data.get("stat", {}).get("like", 0),
                    "views": video_data.get("stat", {}).get("view", 0),
                    "bvid": bvid
                }
            else:
                print(f"获取视频 {bvid} 详情失败: {data.get('message', '未知错误')}")
                return {
                    "title": "获取失败",
                    "likes": 0,
                    "views": 0,
                    "bvid": bvid
                }
        except Exception as e:
            print(f"解析视频 {bvid} 详情出错: {e}")
            return {
                "title": "解析失败",
                "likes": 0,
                "views": 0,
                "bvid": bvid
            }

    def save_to_csv(self, video_info, output_file="raw_data.csv"):
        """将视频信息保存到CSV文件"""
        df = pd.DataFrame([video_info])
        df.to_csv(os.path.join("bilibili_data", output_file), index=False, mode="a", header=not os.path.exists(self.save_path))

    def fetch_data(self, max_pages=5):
        """直接爬取B站排行榜数据"""
        processed_count = 0
        for page in range(1, max_pages + 1):
            print(f"正在处理第 {page} 页...")
            video_list = self.get_video_list(page)

            if not video_list:
                print(f"第 {page} 页没有获取到视频")
                continue

            print(f"第 {page} 页获取到 {len(video_list)} 个视频")

            for video in video_list:
                bvid = video.get("bvid", "")
                if not bvid:
                    continue

                # 解析视频详情
                detail = self.parse_video_detail(bvid)

                # 保存数据
                self.save_to_csv(detail)
                processed_count += 1

                print(f"已处理 {processed_count}/{max_pages*50}: {detail['title']} - 点赞: {detail['likes']}, 播放: {detail['views']}")

                # 控制爬取速度
                time.sleep(1)

            # 每页处理完后休息一下
            time.sleep(3)

        print(f"爬取完成！共获取 {processed_count} 个视频信息")
        df = pd.read_csv(self.save_path)
        return df

    def auto_collect(self, interval=3600, max_pages=5):
        """定时自动采集数据（每 interval 秒一次）"""
        def collect_loop():
            while True:
                print(f"开始自动采集（{datetime.now()}）")
                self.fetch_data(max_pages)
                print(f"自动采集完成，下次采集将在{interval/3600}小时后")
                time.sleep(interval)

        # 启动后台线程执行定时任务
        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()
        return thread


# -------------------------- 2. 数据预处理模块 --------------------------
class DataPreprocessor:
    """数据清洗与特征工程模块"""
    def __init__(self, input_path="bilibili_data/raw_data.csv", output_path="bilibili_data/processed_data.csv"):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self):
        """加载原始数据"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError("原始数据文件不存在，请先采集数据")
        return pd.read_csv(self.input_path)

    def clean_data(self, df):
        """处理缺失值、异常值"""
        # 去除重复数据
        df = df.drop_duplicates()

        # 处理缺失值
        numeric_cols = ["likes", "views"]
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())  # 数值型用中位数填充

        # 处理异常值（根据业务逻辑过滤）
        df = df[df["views"] >= 0]
        df = df[df["likes"] >= 0]

        return df

    def feature_engineering(self, df):
        """提取时间特征、编码分类特征"""
        # 计算互动率特征
        df["like_rate"] = df["likes"] / (df["views"] + 1)  # 点赞率（+1避免除零）

        # 选择最终特征
        features = [
            "views", "likes", "like_rate"
        ]
        target_like = "likes"
        target_view = "views"

        # 分离特征和目标变量
        X = df[features]
        y_like = df[target_like]
        y_view = df[target_view]

        # 保存处理后的数据
        df.to_csv(self.output_path, index=False)
        return X, y_like, y_view, df


# -------------------------- 3. 模型训练与预测模块 --------------------------
class XGBoostModel:
    """基于 XGBoost 的多特征融合模型，支持版本兼容性处理"""
    
    def __init__(self):
        self.model_like = None
        self.model_view = None
        self.features = None
        self.xgb_version = None
        self._check_xgb_version()  # 初始化时检查 XGBoost 版本
    
    def _check_xgb_version(self):
        """检查 XGBoost 版本，确定支持的特性"""
        try:
            import xgboost as xgb
            self.xgb_version = xgb.__version__
            print(f"当前 XGBoost 版本: {self.xgb_version}")
            
            # 检查是否支持 early_stopping_rounds
            self.supports_early_stopping = hasattr(
                xgb.XGBRegressor().fit, 
                'early_stopping_rounds'
            )
            print(f"支持 early_stopping: {self.supports_early_stopping}")
            
        except Exception as e:
            print(f"版本检查失败: {e}")
            self.supports_early_stopping = False
    
    def _create_model(self):
        """创建 XGBoost 模型实例，根据版本动态调整参数"""
        base_params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42
        }
        
        # 尝试使用 GPU 加速（如果可用）
        try:
            import xgboost as xgb
            if xgb.rabit.get_rank() == 0:  # 检查是否有 GPU 支持
                base_params["tree_method"] = "gpu_hist"
                print("启用 GPU 加速训练")
        except:
            base_params["tree_method"] = "auto"
            print("使用 CPU 训练")
            
        return base_params
    
    def train(self, X, y_like, y_view, test_size=0.2):
        """训练模型（分别预测点赞量和播放量）"""
        # 动态创建模型
        model_params = self._create_model()
        
        # 导入 XGBoost 并创建模型实例
        import xgboost as xgb
        self.model_like = xgb.XGBRegressor(**model_params)
        self.model_view = xgb.XGBRegressor(**model_params)
        
        # 手动划分训练集和测试集
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_like_train, y_like_test = y_like[:train_size], y_like[train_size:]
        y_view_train, y_view_test = y_view[:train_size], y_view[train_size:]
        
        # 根据版本选择训练方式
        if self.supports_early_stopping:
            print("使用 early_stopping 训练模型...")
            
            # 训练点赞量预测模型
            self.model_like.fit(
                X_train, y_like_train,
                eval_set=[(X_test, y_like_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # 训练播放量预测模型
            self.model_view.fit(
                X_train, y_view_train,
                eval_set=[(X_test, y_view_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
        else:
            print("XGBoost 版本不支持 early_stopping，使用标准训练...")
            
            # 训练点赞量预测模型
            self.model_like.fit(X_train, y_like_train, verbose=False)
            
            # 训练播放量预测模型
            self.model_view.fit(X_train, y_view_train, verbose=False)
        
        # 评估模型性能
        like_pred = self.model_like.predict(X_test)
        view_pred = self.model_view.predict(X_test)
        
        # 计算评估指标
        metrics = self._calculate_metrics(y_like_test, like_pred, y_view_test, view_pred)
        
        self.features = X.columns
        return metrics, (X_test, y_like_test, y_view_test, like_pred, view_pred)
    
    def _calculate_metrics(self, y_true_like, y_pred_like, y_true_view, y_pred_view):
        """计算模型评估指标"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        return {
            "like_mae": mean_absolute_error(y_true_like, y_pred_like),
            "like_rmse": np.sqrt(mean_squared_error(y_true_like, y_pred_like)),
            "view_mae": mean_absolute_error(y_true_view, y_pred_view),
            "view_rmse": np.sqrt(mean_squared_error(y_true_view, y_pred_view))
        }
    
    def predict(self, X_new):
        """使用训练好的模型预测新数据"""
        if self.features is None:
            raise ValueError("模型尚未训练，请先调用 train 方法")
        
        # 确保输入特征与训练时一致
        if not all(feature in X_new.columns for feature in self.features):
            missing_features = [f for f in self.features if f not in X_new.columns]
            raise ValueError(f"输入数据缺少特征: {missing_features}")
        
        X_new = X_new[self.features]
        like_pred = self.model_like.predict(X_new)
        view_pred = self.model_view.predict(X_new)
        return like_pred, view_pred
    
    def get_feature_importance(self):
        """获取特征重要性排序"""
        if self.features is None:
            raise ValueError("模型尚未训练，请先调用 train 方法")
        
        importance_like = pd.DataFrame({
            "特征": self.features,
            "重要性（点赞量）": self.model_like.feature_importances_
        }).sort_values(by="重要性（点赞量）", ascending=False)
        
        importance_view = pd.DataFrame({
            "特征": self.features,
            "重要性（播放量）": self.model_view.feature_importances_
        }).sort_values(by="重要性（播放量）", ascending=False)
        
        return importance_like, importance_view


# -------------------------- 4. 结果展示模块（GUI） --------------------------
class ResultVisualizer:
    """结果可视化与交互界面（基于 Tkinter）"""
    def __init__(self, root):
        self.root = root
        self.root.title("B站排行榜数据分析系统")
        self.root.geometry("1000x600")

        # 初始化各模块实例
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.model = XGBoostModel()

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        """创建 GUI 界面组件"""
        # 标签页布局
        tab_control = ttk.Notebook(self.root)

        # 1. 数据采集标签页
        tab_collect = ttk.Frame(tab_control)
        tab_control.add(tab_collect, text="数据采集")

        ttk.Button(tab_collect, text="手动采集数据", command=self.run_manual_collect).pack(pady=10)
        ttk.Button(tab_collect, text="启动自动采集", command=self.run_auto_collect).pack(pady=10)
        self.collect_status = tk.Text(tab_collect, height=10, width=80)
        self.collect_status.pack(pady=10)

        # 2. 数据预处理标签页
        tab_preprocess = ttk.Frame(tab_control)
        tab_control.add(tab_preprocess, text="数据预处理")

        ttk.Button(tab_preprocess, text="执行数据清洗", command=self.run_preprocess).pack(pady=10)
        self.preprocess_status = tk.Text(tab_preprocess, height=10, width=80)
        self.preprocess_status.pack(pady=10)

        # 3. 模型训练标签页
        tab_train = ttk.Frame(tab_control)
        tab_control.add(tab_train, text="模型训练")

        ttk.Button(tab_train, text="开始训练模型", command=self.run_train).pack(pady=10)
        self.train_status = tk.Text(tab_train, height=10, width=80)
        self.train_status.pack(pady=10)

        # 4. 结果展示标签页
        tab_result = ttk.Frame(tab_control)
        tab_control.add(tab_result, text="结果展示")

        ttk.Button(tab_result, text="显示特征重要性", command=self.show_importance).pack(pady=10)
        ttk.Button(tab_result, text="显示预测对比", command=self.show_prediction_comparison).pack(pady=10)
        self.result_fig = None

        tab_control.pack(expand=1, fill="both")

    def run_manual_collect(self):
        """手动触发数据采集"""
        def collect():
            self.collect_status.insert(tk.END, "开始手动采集数据...\n")
            df = self.collector.fetch_data(max_pages=5)
            self.collect_status.insert(tk.END, f"手动采集完成，共采集{len(df)}条数据\n")

        threading.Thread(target=collect, daemon=True).start()

    def run_auto_collect(self):
        """启动自动采集"""
        self.collector.auto_collect(interval=3600)
        self.collect_status.insert(tk.END, "自动采集已启动，将每小时采集一次数据\n")

    def run_preprocess(self):
        """执行数据预处理"""
        def preprocess():
            try:
                df = self.preprocessor.load_data()
                self.preprocess_status.insert(tk.END, f"加载原始数据，共{len(df)}条\n")

                df_cleaned = self.preprocessor.clean_data(df)
                self.preprocess_status.insert(tk.END, f"数据清洗完成，剩余{len(df_cleaned)}条\n")

                X, y_like, y_view, df_processed = self.preprocessor.feature_engineering(df_cleaned)
                self.preprocess_status.insert(tk.END, f"特征工程完成，提取{X.shape[1]}个特征\n")

            except Exception as e:
                self.preprocess_status.insert(tk.END, f"预处理出错：{str(e)}\n")

        threading.Thread(target=preprocess, daemon=True).start()

    def run_train(self):
        """执行模型训练"""
        def train():
            try:
                # 加载预处理后的数据
                df = pd.read_csv(self.preprocessor.output_path)
                X, y_like, y_view, _ = self.preprocessor.feature_engineering(df)  # 复用特征工程逻辑

                # 训练模型
                metrics, test_data = self.model.train(X, y_like, y_view)
                self.test_data = test_data

                # 显示训练结果
                self.train_status.insert(tk.END, "模型训练完成，性能指标：\n")
                self.train_status.insert(tk.END, f"点赞量预测 MAE：{metrics['like_mae']:.2f}\n")
                self.train_status.insert(tk.END, f"点赞量预测 RMSE：{metrics['like_rmse']:.2f}\n")
                self.train_status.insert(tk.END, f"播放量预测 MAE：{metrics['view_mae']:.2f}\n")
                self.train_status.insert(tk.END, f"播放量预测 RMSE：{metrics['view_rmse']:.2f}\n")

            except Exception as e:
                self.train_status.insert(tk.END, f"模型训练出错：{str(e)}\n")

        threading.Thread(target=train, daemon=True).start()

    def show_importance(self):
        """展示特征重要性柱状图"""
        try:
            imp_like, imp_view = self.model.get_feature_importance()

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.barplot(x="重要性（点赞量）", y="特征", data=imp_like)
            plt.title("影响点赞量的特征重要性")

            plt.subplot(1, 2, 2)
            sns.barplot(x="重要性（播放量）", y="特征", data=imp_view)
            plt.title("影响播放量的特征重要性")

            plt.show()
        except Exception as e:
            messagebox.showerror("错误", f"显示特征重要性出错：{str(e)}")

    def show_prediction_comparison(self):
        """显示预测值与真实值的对比图"""
        try:
            if not hasattr(self, 'test_data'):
                messagebox.showerror("错误", "请先训练模型")
                return

            _, y_like_test, y_view_test, like_pred, view_pred = self.test_data

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.scatter(y_like_test, like_pred)
            plt.xlabel('真实点赞量')
            plt.ylabel('预测点赞量')
            plt.title('点赞量预测对比')

            plt.subplot(1, 2, 2)
            plt.scatter(y_view_test, view_pred)
            plt.xlabel('真实播放量')
            plt.ylabel('预测播放量')
            plt.title('播放量预测对比')

            plt.show()
        except Exception as e:
            messagebox.showerror("错误", f"显示预测对比出错：{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    visualizer = ResultVisualizer(root)
    root.mainloop()