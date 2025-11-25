from ultralytics import YOLO
from ultralytics.engine.results import Results
import logging
import os

def evaluate_final_generalization(model_path, test_data_path, split="test"):
    """
    评估模型在测试集上的泛化性能
    Args:
        model_path: 训练好的模型权重路径（通常是best.pt）
        test_data_path: 测试集配置文件（如test_dataset.yaml）
        split: 数据集分割方式，固定为"test"以使用测试集
    Returns:
        metrics: 评估指标字典
    """
    # 加载训练好的模型
    model = YOLO(model_path)

    # 在测试集上运行评估
    metrics = model.val(
        data=test_data_path,
        split=split,
        plots=True,  # 生成可视化结果
        verbose=True  # 输出详细日志
    )

    return metrics


# 评估最佳模型在测试集上的泛化性能
if __name__ == "__main__":
    # 训练好的最佳模型权重（通常保存在runs/train/exp/weights/best.pt）
    model_path = "/home/work/yolov8_zhao/runs/detect/my_yolov8_cbam_SEnet_160_160/weights/last.pt"
    # 测试集配置文件（需提前定义，包含test集路径）
    test_data_path = "./ultralytics/cfg/datasets/my_dataset.yaml"

    # 执行评估
    test_metrics = evaluate_final_generalization(model_path, test_data_path)