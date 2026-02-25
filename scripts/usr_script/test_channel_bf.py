import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from envs.core.channel_bf import ChannelModelBF

class MockConfig:
    def __init__(self):
        self.XI1 = 9.61
        self.XI2 = 0.16
        self.ALPHA1 = 2.0  # LoS path loss exponent
        self.ALPHA2 = 3.5  # NLoS path loss exponent
        self.TAU_P = 10
        self.pu = 0.1 # Pilot power
        self.BEAM_EXPONENT = 8.0 # Beam width

def test_beamforming_gain():
    print("="*30)
    print("测试 ChannelModelBF 波束赋形增益")
    print("="*30)
    
    config = MockConfig()
    channel_model = ChannelModelBF(config)
    
    # 场景 1: 单基站，单无人机 (完美对准)
    # 基站位于 (0,0), 无人机位于 (100, 0, 0)
    # 波束指向无人机
    print("\n[场景 1] 单基站，单无人机 (对准)")
    bs_pos = np.array([[0, 0]])
    uav_pos = np.array([[100, 0, 0]])
    beam_matrix = np.array([[True]]) # 基站0 指向 无人机0
    
    bf_gain = channel_model.calculate_beamforming_gain(bs_pos, uav_pos, beam_matrix)
    print(f"基站位置: {bs_pos}, 无人机位置: {uav_pos}")
    print(f"波束矩阵: {beam_matrix}")
    print(f"波束赋形增益: {bf_gain[0,0]:.4f}")
    
    # 期望: 完美对准时增益应为 1.0 (cos(0)^8 = 1)
    if np.isclose(bf_gain[0,0], 1.0):
        print(">> 通过: 完美对准时增益为 1.0。")
    else:
        print(f">> 失败: 期望 1.0，实际得到 {bf_gain[0,0]}")

    # 场景 2: 单基站，多无人机 (干扰检查)
    # 基站位于 (0,0)
    # 无人机1 位于 (100, 0, 0)  (目标)
    # 无人机2 位于 (0, 100, 0)  (受害者, 90度方向)
    # 无人机3 位于 (100, 100, 0) (受害者, 45度方向)
    print("\n[场景 2] 单基站，多无人机 (干扰)")
    bs_pos = np.array([[0, 0]])
    uav_pos = np.array([
        [100, 0, 0],   # UAV0: 0 度
        [0, 100, 0],   # UAV1: 90 度
        [70.7, 70.7, 0] # UAV2: ~45 度
    ])
    
    # 基站仅指向 UAV0
    beam_matrix = np.array([[True, False, False]]) 
    
    bf_gain = channel_model.calculate_beamforming_gain(bs_pos, uav_pos, beam_matrix)
    
    print(f"波束目标: UAV0")
    print(f"UAV0 (0 度) 的增益: {bf_gain[0,0]:.4f}")
    print(f"UAV1 (90 度) 的增益: {bf_gain[0,1]:.4f}")
    print(f"UAV2 (45 度) 的增益: {bf_gain[0,2]:.4f}")
    
    # 检查
    if np.isclose(bf_gain[0,0], 1.0):
        print(">> 通过: 目标 UAV0 增益为 1.0")
    
    # Cos(90) = 0 -> Gain 0
    if np.isclose(bf_gain[0,1], 0.0, atol=1e-5):
        print(">> 通过: 90度方向受害者 UAV1 增益接近 0")
        
    # Cos(45) ~= 0.707. 0.707^8 ~= 0.0625. 应为较小的非零值
    expected_45 = (np.cos(np.radians(45)))**8
    if np.isclose(bf_gain[0,2], expected_45, atol=0.01):
        print(f">> 通过: 45度方向受害者 UAV2 增益为 {bf_gain[0,2]:.4f} (期望 ~{expected_45:.4f})")
    else:
        print(f">> 失败: 45度方向增益不符合预期。得到 {bf_gain[0,2]}，期望 {expected_45}")

    # 场景 3: 与路径损耗 (大尺度衰落) 集成
    print("\n[场景 3] 与路径损耗集成测试")
    # 设置同场景 2，但检查最终的 Beta 值
    # 我们比较有无波束赋形时的 Beta
    
    # 无波束赋形 (传 None)
    beta_no_bf, _, _ = channel_model.calculate_large_scale_fading(bs_pos, uav_pos, beam_matrix=None)
    # 有波束赋形
    beta_with_bf, _, _ = channel_model.calculate_large_scale_fading(bs_pos, uav_pos, beam_matrix=beam_matrix)
    
    print(f"Beta (无BF)  UAV0: {beta_no_bf[0,0]:.2e}")
    print(f"Beta (有BF) UAV0: {beta_with_bf[0,0]:.2e}")
    
    print(f"Beta (无BF)  UAV1: {beta_no_bf[0,1]:.2e}")
    print(f"Beta (有BF) UAV1: {beta_with_bf[0,1]:.2e}")
    
    ratio0 = beta_with_bf[0,0] / beta_no_bf[0,0]
    ratio1 = beta_with_bf[0,1] / beta_no_bf[0,1]
    
    print(f"比率 UAV0 (目标): {ratio0:.4f}")
    print(f"比率 UAV1 (受害者): {ratio1:.4f}")
    
    if np.isclose(ratio0, 1.0) and ratio1 < 0.01:
        print(">> 通过: 目标功率保持，受害者被抑制。")
    else:
        print(">> 失败: 功率比率不符合预期。")

    # 场景 4: 单基站多波束 (多用户)
    print("\n[场景 4] 单基站多波束")
    # 基站位于 (0,0)
    # UAV0 位于 (100, 0, 0)
    # UAV1 位于 (-100, 0, 0) (180度方向)
    bs_pos = np.array([[0, 0]])
    uav_pos = np.array([
        [100, 0, 0],   # 0 度
        [-100, 0, 0]   # 180 度
    ])
    beam_matrix = np.array([[True, True]]) # 基站服务两个用户
    
    bf_gain = channel_model.calculate_beamforming_gain(bs_pos, uav_pos, beam_matrix)
    print(f"波束矩阵: 两者均为 True")
    print(f"增益 UAV0: {bf_gain[0,0]:.4f}")
    print(f"增益 UAV1: {bf_gain[0,1]:.4f}")
    
    # 对于 UAV0: 接收发给自己的波束 (增益 1) + 发给 UAV1 的波束 (增益 0, 背瓣为 0)
    # 对于 UAV1: 接收发给自己的波束 (增益 1) + 发给 UAV0 的波束 (增益 0)
    if np.isclose(bf_gain[0,0], 1.0) and np.isclose(bf_gain[0,1], 1.0):
        print(">> 通过: 两个用户均获得全增益 (空间分离)。")
    else:
        print(f">> 失败: 增益 {bf_gain}")

if __name__ == "__main__":
    test_beamforming_gain()
