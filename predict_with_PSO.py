import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import sys
import os

# 添加环境目录到系统路径，以便导入env模块
from env import config as cnf
from env.utility import calc_util_obs, arr2mat
# Ensure test directory is in the same level as this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_case import fixed_grid_environment, random_users_environment, corner_base_stations_environment

def genetic_algorithm(fitness_function, dimensions, bounds=(0.0, 1.0), 
                      population_size=50, generations=100, 
                      crossover_prob=0.7, mutation_prob=0.2, mutation_sigma=0.1,
                      tournament_size=3, elite_size=1):
    """
    Standard Genetic Algorithm Implementation
    
    Parameters:
        fitness_function: Function that takes an individual and returns fitness value
        dimensions: Problem dimensions (number of genes in individual)
        bounds: Range for each gene, tuple (min, max)
        population_size: Size of population
        generations: Maximum number of iterations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        mutation_sigma: Standard deviation for Gaussian mutation
        tournament_size: Size for tournament selection
        elite_size: Number of elite individuals
    
    Returns:
        best_individual: Best individual found
        best_fitness: Fitness value of best individual
        stats: Statistics information
    """
    # Create maximization problem (we want to maximize network utility)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define genes and individuals
    toolbox.register("attr_float", random.uniform, bounds[0], bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_float, n=dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operations
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation_sigma, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    # Create population
    pop = toolbox.population(n=population_size)
    
    # Statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    
    # Hall of fame to record elite individuals
    hof = tools.HallOfFame(elite_size)
    
    # Run algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, 
                                     mutpb=mutation_prob, ngen=generations, 
                                     stats=stats, halloffame=hof, verbose=True)
    
    # Return best individual and fitness
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    
    return best_individual, best_fitness, logbook

def normalize_power_allocation(individual):
    """
    Normalize power allocation values so that sum equals 1 for each AP
    
    Parameters:
        individual: Power allocation individual from GA
        
    Returns:
        normalized_individual: Individual with normalized power values
    """
    # Create a copy to avoid modifying the original
    norm_individual = np.array(individual, dtype=float)
    
    # Replace negative values with 0
    norm_individual[norm_individual < 0] = 0
    
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # Calculate dimensions for each allocation matrix
    g_alloc_dim = n_g * (n_u + n_a)  # Ground AP allocations
    a_alloc_dim = n_a * n_u           # Aerial AP allocations
    
    # Extract portions for ground and aerial APs
    g_alloc = norm_individual[:g_alloc_dim].reshape(n_g, n_u + n_a)
    a_alloc = norm_individual[g_alloc_dim:].reshape(n_a, n_u)
    
    # Normalize ground AP allocations (row-wise)
    for i in range(n_g):
        row_sum = np.sum(g_alloc[i, :])
        if row_sum > 0:
            g_alloc[i, :] = g_alloc[i, :] / row_sum
    
    # Normalize aerial AP allocations (row-wise)
    for i in range(n_a):
        row_sum = np.sum(a_alloc[i, :])
        if row_sum > 0:
            a_alloc[i, :] = a_alloc[i, :] / row_sum
    
    # Flatten and combine
    norm_individual[:g_alloc_dim] = g_alloc.flatten()
    norm_individual[g_alloc_dim:] = a_alloc.flatten()
    
    return norm_individual

def fitness_wrapper(individual, actual_position):
    """
    Wrapper for calc_util_obs function as fitness function
    
    Parameters:
        individual: Individual from GA (power allocation)
        actual_position: Actual system positions
        
    Returns:
        utility: Utility value as fitness
    """
    # Normalize power allocation values
    normalized_individual = normalize_power_allocation(individual)
    
    # Calculate utility using normalized values
    utility, capacity, punishment, link_cost = calc_util_obs(normalized_individual, actual_position)
    return utility,  # Note: return as tuple

def plot_evolution(logbook, title="Optimization Process"):
    """Plot maximum, average, and standard deviation during evolution"""
    gen = logbook.select("gen")
    max_values = logbook.select("max")
    avg_values = logbook.select("avg")
    std_values = logbook.select("std")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gen, max_values, 'b-', label='Maximum')
    ax.plot(gen, avg_values, 'r-', label='Average')
    ax.fill_between(gen, [avg - std for avg, std in zip(avg_values, std_values)],
                   [avg + std for avg, std in zip(avg_values, std_values)],
                   alpha=0.2, color='red')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Network Utility)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('ga_evolution.png')
    plt.show()

def denormalize_state(normalized_state):
    """
    将归一化的状态转换为实际的物理单位状态
    
    参数:
        normalized_state: 由test_case.py生成的归一化状态
        
    返回:
        actual_position: 反归一化后的实际物理位置
    """
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # 位置信息的长度
    pos_len = 3 * n_a + 2 * n_g + 2 * n_u
    
    # 提取位置部分
    norm_position = normalized_state[:pos_len]
    
    # 创建实际位置数组
    actual_position = np.empty_like(norm_position, dtype=float)
    
    # 无人机坐标反归一化 (x, y, h)
    actual_position[:n_a] = norm_position[:n_a] * cnf.MAX_X  # x坐标
    actual_position[n_a:2*n_a] = norm_position[n_a:2*n_a] * cnf.MAX_Y  # y坐标
    actual_position[2*n_a:3*n_a] = norm_position[2*n_a:3*n_a] * cnf.MAX_H  # 高度
    
    # 地面基站坐标反归一化 (x, y)
    actual_position[3*n_a:3*n_a+n_g] = norm_position[3*n_a:3*n_a+n_g] * cnf.MAX_X  # x坐标
    actual_position[3*n_a+n_g:3*n_a+2*n_g] = norm_position[3*n_a+n_g:3*n_a+2*n_g] * cnf.MAX_Y  # y坐标
    
    # 用户坐标反归一化 (x, y)
    actual_position[3*n_a+2*n_g:3*n_a+2*n_g+n_u] = norm_position[3*n_a+2*n_g:3*n_a+2*n_g+n_u] * cnf.MAX_X  # x坐标
    actual_position[3*n_a+2*n_g+n_u:] = norm_position[3*n_a+2*n_g+n_u:] * cnf.MAX_Y  # y坐标
    
    return actual_position

def plot_network_topology(position, power_alloc_action):
    """
    Plot network topology showing UAVs, ground base stations, users, and power allocations.
    
    Parameters:
        position: Array containing positions of all APs and users
        power_alloc_action: Optimal power allocation matrix
    """
    n_a = cnf.NUM_A_AP
    n_g = cnf.NUM_G_AP
    n_u = cnf.NUM_USERS
    
    # Normalize power allocation
    power_alloc_action = normalize_power_allocation(power_alloc_action)
    
    # Split position information
    aerial_end = 3 * n_a
    ground_end = aerial_end + 2 * n_g
    user_end = ground_end + 2 * n_u

    # Extract coordinates for each node type
    aerial_pos = position[:aerial_end].reshape(3, n_a)  # [x_a, y_a, h_a]
    x_a, y_a, h_a = aerial_pos
    ground_pos = position[aerial_end:ground_end].reshape(2, n_g)  # [x_g, y_g]
    x_g, y_g = ground_pos
    user_pos = position[ground_end:user_end].reshape(2, n_u)  # [x_u, y_u]
    x_u, y_u = user_pos
    
    # Convert power allocation to matrix form
    G_beta, G_eta, A_beta, A_eta = arr2mat(power_alloc_action, position)
    
    # Create plot window
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('Network Topology and Optimal Power Allocation', fontsize=16)
    ax.set_xlabel('X-coordinate (m)', fontsize=14)
    ax.set_ylabel('Y-coordinate (m)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, cnf.MAX_X)
    ax.set_ylim(0, cnf.MAX_Y)
    
    # Plot ground APs (red triangles)
    ax.scatter(x_g, y_g, c='red', marker='^', s=200, label='Ground Base Station')
    for idx, (xi, yi) in enumerate(zip(x_g, y_g)):
        ax.text(xi, yi - 25, f'G{idx}', fontsize=12, ha='center', va='top', color='red')
    
    # Plot users (green circles)
    ax.scatter(x_u, y_u, c='green', marker='o', s=150, label='User')
    for idx, (xi, yi) in enumerate(zip(x_u, y_u)):
        ax.text(xi, yi - 25, f'U{idx}', fontsize=12, ha='center', va='top', color='green')
    
    # Plot UAVs (blue squares)
    ax.scatter(x_a, y_a, c='blue', marker='s', s=150, label='UAV')
    for idx, (xi, yi, hi) in enumerate(zip(x_a, y_a, h_a)):
        ax.text(xi, yi - 30, f'A{idx}\nH:{hi:.1f}m', fontsize=12, ha='center', va='top', color='blue')
    
    # Draw connections: ground AP to user (red dashed lines)
    connection_threshold = 0.05  # Increased threshold to reduce visual clutter
    for g in range(n_g):
        for u in range(n_u):
            if G_eta[g, u] > connection_threshold:
                ax.plot([x_g[g], x_u[u]], [y_g[g], y_u[u]], 'r--', alpha=0.5, linewidth=1.5)
                mid_x = (x_g[g] + x_u[u]) / 2
                mid_y = (y_g[g] + y_u[u]) / 2
                ax.text(mid_x, mid_y, f'{G_eta[g, u]:.2f}', fontsize=10, 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw connections: UAV to user (blue dashed lines)
    for a in range(n_a):
        for u in range(n_u):
            if A_eta[a, u] > connection_threshold:
                ax.plot([x_a[a], x_u[u]], [y_a[a], y_u[u]], 'b--', alpha=0.5, linewidth=1.5)
                mid_x = (x_a[a] + x_u[u]) / 2
                mid_y = (y_a[a] + y_u[u]) / 2
                ax.text(mid_x, mid_y, f'{A_eta[a, u]:.2f}', fontsize=10, 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw connections: ground AP to UAV (green dashed lines)
    for g in range(n_g):
        for a in range(n_a):
            uav_col = n_u + a  # First n_u columns are users, next n_a columns are UAVs
            if G_eta[g, uav_col] > connection_threshold:
                ax.plot([x_g[g], x_a[a]], [y_g[g], y_a[a]], 'g--', alpha=0.5, linewidth=1.5)
                mid_x = (x_g[g] + x_a[a]) / 2
                mid_y = (y_g[g] + y_a[a]) / 2
                ax.text(mid_x, mid_y, f'{G_eta[g, uav_col]:.2f}', fontsize=10, 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend with connection types
    handles, labels = ax.get_legend_handles_labels()
    # Add connection line types to legend
    handles.extend([
        plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5),
        plt.Line2D([0], [0], color='blue', linestyle='--', lw=1.5),
        plt.Line2D([0], [0], color='green', linestyle='--', lw=1.5)
    ])
    labels.extend([
        'Ground BS to User Connection',
        'UAV to User Connection',
        'Ground BS to UAV Connection'
    ])
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    # Calculate cluster partitioning
    from env.utility import calc_cluster
    adj_matrix, cluster_labels = calc_cluster(A_eta, G_eta)
    
    # Display nodes in different clusters
    clusters = {}
    for idx in range(n_g + n_a + n_u):
        if idx < n_g:
            node_type = f'G{idx}'
        elif idx < n_g + n_a:
            node_type = f'A{idx - n_g}'
        else:
            node_type = f'U{idx - n_g - n_a}'
        
        cluster_id = cluster_labels[idx]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node_type)
    
    # Add cluster information at the bottom of the chart
    cluster_text = "Network Cluster Partitioning:\n"
    for cid, nodes in clusters.items():
        cluster_text += f"Cluster {cid+1}: {', '.join(nodes)}\n"
    
    plt.figtext(0.5, 0.02, cluster_text, ha='center', fontsize=12, 
               bbox=dict(facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Leave space for cluster information
    plt.savefig('network_topology.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def main():
    # 选择一个测试场景
    scenario = 1  # 1: 固定网格, 2: 随机用户, 3: 角落基站
    scenario_name = ["Fixed Grid", "Random Users", "Corner Base Stations"][scenario-1]
    
    # 获取归一化的测试场景状态
    if scenario == 1:
        normalized_state = fixed_grid_environment()
    elif scenario == 2:
        normalized_state = random_users_environment()
    else:  # scenario == 3
        normalized_state = corner_base_stations_environment()
    
    # 将归一化状态转换为实际物理单位状态
    actual_position = denormalize_state(normalized_state)
    
    print(f"Using {scenario_name} scenario")
    print(f"Actual position shape: {actual_position.shape}")
    
    # 计算功率分配矩阵的维度
    # 地面AP到用户和无人机的功率分配: n_g × (n_u + n_a)
    # 无人机到用户的功率分配: n_a × n_u
    power_alloc_dimensions = cnf.NUM_G_AP * (cnf.NUM_USERS + cnf.NUM_A_AP) + cnf.NUM_A_AP * cnf.NUM_USERS
    print(f"Power allocation dimensions: {power_alloc_dimensions}")
    
    # 创建适应度函数的包装器
    def fitness_func(individual):
        return fitness_wrapper(individual, actual_position)
    
    # 运行遗传算法
    best_individual, best_fitness, logbook = genetic_algorithm(
        fitness_function=fitness_func,
        dimensions=power_alloc_dimensions,
        bounds=(0.0, 1.0),  # 功率分配范围 [0,1]
        population_size=150,  # 较大的种群大小以适应搜索空间
        generations=500,      # 更多迭代次数
        crossover_prob=0.7,
        mutation_prob=0.3,    # 较高的变异概率以增加多样性
        mutation_sigma=0.1,   # 高斯变异的标准差
        tournament_size=3,
        elite_size=5          # 保留更多的精英个体
    )
    
    # 归一化最佳个体
    best_individual = normalize_power_allocation(best_individual)
    
    # 转换最佳个体为结构化矩阵进行分析
    G_beta, G_eta, A_beta, A_eta = arr2mat(best_individual, actual_position)
    
    # 计算最佳解决方案的指标
    utility, capacity, punishment, link_cost = calc_util_obs(best_individual, actual_position)
    
    # 打印结果
    print("\nOptimization Results:")
    print(f"Best fitness value: {best_fitness}")
    print(f"Total network capacity: {capacity}")
    print(f"Penalty term: {punishment}")
    print(f"Link cost: {link_cost}")
    
    # 绘制进化过程
    plot_evolution(logbook, title=f"Network Power Allocation Optimization - {scenario_name} Scenario")
    
    # 绘制网络拓扑
    plot_network_topology(actual_position, best_individual)

if __name__ == "__main__":
    # 清除已存在的creator定义（避免多次运行时出错）
    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individual' in creator.__dict__:
        del creator.Individual
        
    main()