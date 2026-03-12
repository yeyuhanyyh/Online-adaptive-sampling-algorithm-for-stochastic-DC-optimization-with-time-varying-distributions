% 参数设置
clear;
rng(42); % 固定随机种子以便复现
dim = 20; % x 的维度
beta_opt = [15; -12; 0; 0; 0; 10; 0; 0; 0; 0; 0; 10; 0; 0; 0; 0; -10; 0; 0; 0]; % 稀疏最优解
beta0 = zeros(dim, 1); % 初始化点
mu = 1; % 二次近端项系数
lambda = 0.01; % 正则化系数
alpha = 1; % 与目标函数相关的参数
error_threshold = 1e-100; % 收敛阈值
delta_t_max = 100; % 分布扰动的最大值
max_runtime = 3; % 最大运行时间（秒）

% 存储结果
max_iter = 1e6; % 预设最大迭代次数（防止无限循环）
beta3 = zeros(dim, max_iter);
beta3(:, 1) = beta0;
stepsize3 = zeros(1, max_iter); 
norms3 = zeros(1, max_iter); % 存储步长二范数
total_samples3 = 0; % 总采样数
beta_distances3 = zeros(1, max_iter); % 当前解与最优解的距离
elapsed_time3 = zeros(1, max_iter); % 每次迭代的累计时间

% 初始化存储样本点
xi_all = [];
yi_all = [];

% 全局计时器启动
global_start_time = tic;

% 运行自适应样本大小近似点梯度法
for k = 1:max_iter
    % 每次迭代计时器
    iteration_start_time = tic;
    
    % 分布扰动 delta_t 随时间衰减
    delta_t = delta_t_max * k^(-2); % 偏移随迭代次数的二次衰减
    if mod(k, 2) == 1
        delta_t = delta_t; % 奇数次，保留 delta_t
    else
        delta_t = -delta_t; % 偶数次，取相反数
    end

    % 每次采样一个新点
    [xi_new, yi_new] = generate_samples(1, beta_opt + delta_t);

    % 将新样本点加入样本集
    xi_all = [xi_all; xi_new];
    yi_all = [yi_all; yi_new];

    % 更新总采样次数
    total_samples3 = total_samples3 + 1;

    % 构建目标函数 g(beta)（基于所有采样过的样本点）
    g_k_func = @(beta) compute_g(beta, xi_all, yi_all, alpha, lambda);

    % 计算 h 的次梯度
    h_grad = h_subgradient(beta3(:, k), alpha, lambda);

    % 求解优化子问题
    beta3(:, k+1) = solve_subproblem(beta3(:, k), g_k_func, h_grad, mu);

    % 更新
    d_k = beta3(:, k+1) - beta3(:, k);
    stepsize3(k) = norm(d_k); % 计算更新量
    norms3(k) = norm(beta3(:, k+1)); % 计算当前解的二范数

    % 当前解与最优解的距离
    beta_distances3(k) = norm(beta3(:, k+1) - beta_opt);

    % 记录累计的系统时间
    elapsed_time3(k) = sum(elapsed_time3(1:k-1)) + toc(iteration_start_time);

    % 收敛检查
    if norm(d_k) < error_threshold
        fprintf('迭代提前收敛，步长小于阈值。\n');
        break;
    end

    % 检查运行时间是否超过最大时间
    if toc(global_start_time) > max_runtime
        fprintf('运行时间已达 %d 秒，提前退出。\n', max_runtime);
        break;
    end
end

% 记录结果
final_solution = beta3(:, k+1);
fprintf('最终解: %s\n', mat2str(final_solution, 4));
fprintf('当前解与最优解的距离: %f\n', beta_distances3(k));
fprintf('总采样次数: %d\n', total_samples3);
fprintf('实际运行时间: %.2f 秒\n', toc(global_start_time));

% ----------------绘图----------------
figure;

% 绘制当前解与最优解的距离随时间变化
subplot(2, 1, 1);
semilogy(1:k, beta_distances3(1:k), '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('与最优解的距离 (对数尺度)');
title('当前解与最优解的距离随迭代次数变化');

% 绘制误差随系统时间变化
subplot(2, 1, 2);
error = beta_distances3(1:k); % 误差就是当前解与最优解的距离
semilogy(elapsed_time3(1:k), error, '-o', 'LineWidth', 1.5); % 使用 elapsed_time 作为横坐标
xlabel('系统时间（秒）');
ylabel('误差 (对数尺度)');
title('误差随系统时间变化');

% ----------------子函数定义----------------

% 生成样本函数
function [xi, yi] = generate_samples(N, beta_true)
    dim = length(beta_true);
    xi = 2 * rand(N, dim) - 1; 
    noise = randn(N, 1); % 随机噪声
    yi = xi * beta_true + noise; % y_i = beta_true' * x_i + 噪声
end

% 定义目标函数 g(beta)
function g_k_func = compute_g(beta, xi, yi, alpha, lambda)
    % 计算残差
    residual = yi - xi * beta;
    % 计算L1损失
    li_loss = abs(residual);
    % 总目标函数值
    g_k_func = (1 + alpha * norm(beta, 1))*lambda + mean(li_loss); % L1 损失取均值
end

% 定义 h = max(1, alpha * |beta|) 的次梯度
function h_grad = h_subgradient(beta, alpha, lambda)
    h_grad = zeros(size(beta));
    for i = 1:length(beta)
        if abs(beta(i)) > 1 / alpha
            h_grad(i) = lambda * alpha * sign(beta(i)); % |β| > 1/α 时，次梯度为 α * sign(β)
        else
            h_grad(i) = 0; % |β| ≤ 1/α 时，次梯度为 0
        end
    end
end

% 求解子问题
function beta_next = solve_subproblem(beta_prev, g_k_func, h_grad, mu)
    % 定义目标函数
    fun = @(z) g_k_func(z) - (z - beta_prev)' * h_grad + 0.5 * mu * norm(z - beta_prev)^2;

    % 设置优化选项
    options = optimoptions('fminunc', 'Display', 'off'); % 使用梯度下降方法

    % 使用 fminunc 进行无约束优化
    beta_next = fminunc(fun, beta_prev, options);
end
