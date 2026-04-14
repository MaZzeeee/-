import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- 1. 设置随机种子 ---
def set_random_seed(seed=42):
    """
    固定所有可能的随机源，确保实验结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为CPU设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置种子
        torch.cuda.manual_seed_all(seed) # 为所有GPU设置种子
        # 确保CUDA的卷积操作是确定性的（可能会降低速度，但保证精度一致）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 2. 数据加载与预处理 ---
def load_data(file_path):
    # 读取Excel，假设第一行是表头
    df = pd.read_excel(file_path, header=0)
    
    # 根据文档描述，前4列是成分，后面700列是光谱 (1100-2498nm)
    X = df.iloc[:, 4:704].values.astype(np.float32)
    y = df.iloc[:, 0:4].values.astype(np.float32)
    
    return X, y

# --- 3. 深度学习模型设计 (1D-CNN) ---
class SpectralCNN(nn.Module):
    def __init__(self, input_channels=700):
        super(SpectralCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 87, 256), # 87 = 700经过卷积池化后的计算长度
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # (Batch, 1, 700)
        x = self.features(x)
        x = x.view(x.size(0), -1) # 展平
        x = self.regressor(x)
        return x

# --- 4. 主程序 ---
def main():
    # 1. 设置随机种子 (修复点)
    set_random_seed(42) # 固定为42，你可以指定任何数字
    
    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. 加载数据
    try:
        X, y = load_data('D:\桌面\数据技术基础\problem1\作业\玉米的近红外光谱数据.xlsx')
        print(f"数据加载成功，样本数: {X.shape[0]}")
    except FileNotFoundError:
        print("错误：未找到数据文件 '玉米的近红外光谱数据.xlsx'")
        print("请确保文件在当前目录下，或修改文件路径。")
        return

    # 4. 数据归一化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42, shuffle=True)
    
    # 转换为Tensor
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 6. 初始化模型
    model = SpectralCNN().to(device)
    criterion = nn.MSELoss()
    # 使用L2正则化 (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 

    # 7. 训练循环
    epochs = 200
    train_losses = []

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}')

    # 8. 评估与保存结果 (保持不变)
    model.eval()
    y_pred_scaled = []
    y_true_scaled = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_pred_scaled.extend(outputs.cpu().numpy())
            y_true_scaled.extend(labels.cpu().numpy())
    
    # 反归一化
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled))
    y_true = scaler_y.inverse_transform(np.array(y_true_scaled))
    
    # 计算MSE
    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    print("\n测试集上的均方误差 (MSE)：")
    components = ['Moisture', 'Oil', 'Protein', 'Starch']
    for i, comp in enumerate(components):
        print(f"{comp}: {mse[i]:.4f}")

    # --- 9. 绘图与保存 ---
    # (此处代码与之前一致，省略以节省空间，实际使用时请保留)
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'Sample_ID': range(len(y_true)),
        'True_Moisture': y_true[:, 0], 'Pred_Moisture': y_pred[:, 0],
        'True_Oil': y_true[:, 1], 'Pred_Oil': y_pred[:, 1],
        'True_Protein': y_true[:, 2], 'Pred_Protein': y_pred[:, 2],
        'True_Starch': y_true[:, 3], 'Pred_Starch': y_pred[:, 3],
    })
    result_df.to_csv('test_predictions.csv', index=False)
    print("\n预测结果已保存。")

if __name__ == '__main__':
    main()