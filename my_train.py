import time
import torch

from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    start_time = time.time()
    model = YOLO('./ultralytics/cfg/models/v8/yolov8.yaml')
    # Train the model
    model.train(epochs=300, batch=64,data='/home/work/yolov8_zhao/ultralytics/cfg/datasets/my_dataset.yaml')
    #epochs表示训练的轮数，batch表示每次加载图片的数量，data是数据集的路径这里使用的是之前创建的yaml文件

    end_time = time.time()
    total_seconds = end_time - start_time
    # 转换为 小时:分钟:秒 格式（更易读）
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成，耗时：{int(hours)}:{int(minutes)}:{int(seconds)}")

# from ultralytics import YOLO
#
# # 加载修改后的配置文件
# model = YOLO("ultralytics/cfg/models/v8/my_yolov8_cbam.yaml")
# # 打印网络结构信息（查看是否有CBAM模块）
# model.info()