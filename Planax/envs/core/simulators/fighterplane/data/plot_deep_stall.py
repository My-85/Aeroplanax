import numpy as np
import matplotlib.pyplot as plt
import os

# === 1. 设置数据路径 ===
BASE_DIR = r"F:\研究生\清华课题组\neuralplane_jax\20251215最新代码库\Planax\envs\core\simulators\fighterplane\data"

# 定义维度数据
ALPHA = np.array([-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 
                  30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0])
# Beta 索引: -30, ..., 0, ..., 30。0.0 是第 9 个索引 (index=9)
BETA_ZERO_IDX = 9 
NUM_BETA = 19
NUM_ALPHA = 20

# === 2. 定义读取函数 (带切片) ===
def read_cm_at_beta_zero(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    # 读取原始数据
    raw_data = np.loadtxt(filepath)
    # 展平以防万一
    flat_data = raw_data.flatten()
    
    # 根据 aero_data.py，数据通常是 (DH, BETA, ALPHA) 或 (BETA, ALPHA)
    # 我们先假设这个文件包含了一个 DH 层面的数据，即 Shape=(19, 20) = 380 个点
    # 或者它包含了所有 5 个 DH，即 1900 个点。
    
    # 检查大小
    if flat_data.size == 1900: 
        # 这是一个完整的 CM 表 (5, 19, 20)
        # 我们需要根据 DH 索引提取
        return flat_data.reshape((5, 19, 20))
    elif flat_data.size == 380:
        # 这是一个单层 DH 的表 (19, 20)
        return flat_data.reshape((19, 20))
    else:
        print(f"Warning: Unexpected data size {flat_data.size}. Expected 380 or 1900.")
        return None

# === 3. 读取数据 ===
# 既然 aero_data.py 里是 `Cm = safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat')` 
# 且后面有 reshape((5, 19, 20))，说明这一个文件里就包含了所有 5 个舵偏角的数据！
# 也就是说，我们只需要这就一个文件，就能画出所有线！

CM_FILE = "CM0120_ALPHA1_BETA1_DH1_101.dat"
cm_data_full = read_cm_at_beta_zero(CM_FILE)

if cm_data_full is not None:
    # 提取 Beta=0 (index 9) 的数据
    # cm_data_full shape is (5, 19, 20) -> (DH, Beta, Alpha)
    
    # DH1 = [-25, -10, 0, +10, +25]
    # Index 0: -25 (Push)
    # Index 2: 0   (Neutral)
    # Index 4: +25 (Pull)
    
    cm_push = cm_data_full[0, BETA_ZERO_IDX, :]
    cm_neutral = cm_data_full[2, BETA_ZERO_IDX, :]
    cm_pull = cm_data_full[4, BETA_ZERO_IDX, :]

    # === 4. 绘图 ===
    plt.figure(figsize=(10, 6))

    # 参考线
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(60, color='gray', linewidth=1, linestyle=':', label='Deep Stall Region (~60°)')

    # 曲线
    plt.plot(ALPHA, cm_push, 'g-o', label='Elevator = -25° (Full Push / Recovery)', linewidth=2)
    plt.plot(ALPHA, cm_neutral, 'b-o', label='Elevator = 0° (Neutral)', linewidth=2)
    plt.plot(ALPHA, cm_pull, 'r-o', label='Elevator = +25° (Full Pull)', linewidth=2)

    # 标注
    plt.title("F-16 Pitching Moment ($C_m$) vs Alpha ($\\alpha$) @ $\\beta=0$\n(Correctly Sliced Data)", fontsize=14)
    plt.xlabel("Angle of Attack $\\alpha$ (deg)", fontsize=12)
    plt.ylabel("Pitching Moment Coefficient $C_m$", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 锁定点标注
    # 只要绿线在 60 度附近 > 0 或者与 0 轴相交且斜率为负，就是锁定点
    plt.annotate('Deep Stall Trim Point', xy=(60, 0), xytext=(65, 0.15),
                 arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontsize=12)

    plt.xlim([-25, 95])
    plt.ylim([-0.8, 0.4]) 

    output_path = os.path.join(BASE_DIR, "deep_stall_corrected.png")
    # plt.savefig(output_path)
    plt.show()
    print("Corrected plot generated.")
else:
    print("Failed to load data structure.")