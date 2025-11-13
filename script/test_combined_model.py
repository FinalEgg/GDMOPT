import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.combined import CombinedModel, CombinedConfig

def test_combined_model():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义模型参数
    state_dim = 10  # 状态空间维度
    connection_dim = 5  # 连接矩阵维度（无人机-基站连接数量）
    power_dim = 5  # 功率分配维度
    threshold = 0.5  # 阈值
    
    # 创建配置实例
    config = CombinedConfig()
    config.update(
        actor_hidden_dim=64,
        diffusion_hidden_dim=64,
        diffusion_timesteps=5,
        threshold=threshold,
        max_power=1.0
    )
    
    # 创建模型实例
    model = CombinedModel(
        state_dim=state_dim,
        connection_dim=connection_dim,
        power_dim=power_dim,
        config=config
    )
    
    print("模型创建成功！")
    
    # 生成测试数据
    batch_size = 2
    test_state = torch.randn(batch_size, state_dim)
    
    print(f"测试状态形状: {test_state.shape}")
    
    # 测试基本前向传播
    print("\n1. 测试基本前向传播:")
    connection_matrix, power_allocation = model(test_state)
    
    print(f"连接矩阵形状: {connection_matrix.shape}")
    print(f"连接矩阵内容:\n{connection_matrix.detach().numpy()}")
    print(f"功率分配形状: {power_allocation.shape}")
    print(f"功率分配内容:\n{power_allocation.detach().numpy()}")
    
    # 测试不同的输出模式
    print("\n2. 测试不同的输出模式:")
    
    # 只获取连接矩阵
    connection_only = model(test_state, output_mode='connection')
    print(f"只获取连接矩阵形状: {connection_only.shape}")
    
    # 只获取功率分配
    power_only = model(test_state, output_mode='power')
    print(f"只获取功率分配形状: {power_only.shape}")
    
    # 获取连接概率和连接矩阵
    connection_prob, connection_mat = model(test_state, output_mode='prob_and_connection')
    print(f"连接概率形状: {connection_prob.shape}")
    print(f"连接矩阵形状: {connection_mat.shape}")
    
    # 测试自定义阈值
    print("\n3. 测试自定义阈值:")
    custom_threshold = 0.7
    connection_matrix_high, _ = model(test_state, threshold=custom_threshold)
    print(f"使用较高阈值 ({custom_threshold}) 的连接矩阵:\n{connection_matrix_high.detach().numpy()}")
    
    # 测试软阈值
    print("\n4. 测试软阈值:")
    model.set_soft_threshold(True, temperature=0.1)
    model.train()  # 设置为训练模式以启用软阈值
    soft_connection = model(test_state, output_mode='connection')
    print(f"软阈值连接输出:\n{soft_connection.detach().numpy()}")
    
    # 测试二进制连接获取（无论软阈值设置如何）
    binary_connection = model.get_binary_connection(test_state)
    print(f"强制二进制连接矩阵:\n{binary_connection.detach().numpy()}")
    
    # 测试批量处理
    print("\n5. 测试批量处理:")
    # 创建更大的批次
    larger_batch = torch.randn(5, state_dim)
    batch_connections, batch_powers = model.batch_forward(larger_batch)
    print(f"批量处理连接矩阵形状: {batch_connections.shape}")
    print(f"批量处理功率分配形状: {batch_powers.shape}")
    
    # 测试不同阈值的批量处理
    print("\n6. 测试不同阈值的批量处理:")
    different_thresholds = torch.tensor([0.3, 0.5, 0.7, 0.2, 0.9])
    batch_connections_diff_thresh, _ = model.batch_forward(larger_batch, thresholds=different_thresholds)
    print(f"不同阈值批量处理连接矩阵:\n{batch_connections_diff_thresh.detach().numpy()}")
    
    # 测试损失计算
    print("\n7. 测试损失计算:")
    target_connection = torch.randint(0, 2, (batch_size, connection_dim)).float()
    target_power = torch.rand(batch_size, power_dim)
    loss = model.loss(test_state, target_connection, target_power)
    print(f"损失值: {loss.item()}")
    
    # 测试sample方法
    print("\n8. 测试sample方法:")
    sampled_connection, sampled_power = model.sample(test_state)
    print(f"采样连接矩阵:\n{sampled_connection.detach().numpy()}")
    print(f"采样功率分配:\n{sampled_power.detach().numpy()}")
    
    print("\n所有测试完成！模型功能正常。")

if __name__ == "__main__":
    test_combined_model()