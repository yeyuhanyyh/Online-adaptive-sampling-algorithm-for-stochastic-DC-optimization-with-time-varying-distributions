% 参数设置
clear;
rng(40); % 固定随机种子以便复现
max_iter = 1e6; % 最大迭代次数
N = 10; % 每步的样本数量
dim = 50; % x 的维度
beta_opt = zeros(dim, 1); % 先初始化为全零
beta_opt([1, 2]) = [5, -5]; % 只在特定位置赋值
beta0 = zeros(dim, 1); % 初始化点
mu = 1; % 二次近端项系数
error_threshold = 1e-10; % 收敛阈值
lambda = 0.01;
C_g = 1;
alpha = 1; % 与目标函数相关的alpha
alpha_g = 0.45; % 更新sample size的alpha_g
delta_t_max = 5000; % 分布扰动的最大值
max_runtime = 20; % 最大运行时间（秒）

%---------------算法1--------------

% 存储结果
beta1 = zeros(dim, max_iter);
beta1(:, 1) = beta0;
stepsize1 = zeros(1, max_iter);
norms1 = zeros(1, max_iter); % 存储步长二范数
total_samples1 = 0; % 总采样数
N_history1 = zeros(1, max_iter); % 存储每次更新的 N
beta_distances1 = zeros(1, max_iter); % 当前解与最优解的距离
elapsed_time1 = zeros(1, max_iter); % 累计的系统时间

% 记录总运行开始时间
global_start_time = tic;

% 运行自适应样本大小近似点梯度法
for k = 1:max_iter
    % 启动计时器
    iteration_start_time = tic;

    % 分布扰动 delta_t 随时间衰减
    delta_t = delta_t_max * k^(-2); % 偏移随迭代次数的二次衰减

    % 奇数次迭代时增加delta_t，偶数次时减少delta_t
    if mod(k, 2) == 1
        delta_t = delta_t; % 奇数次，保留delta_t
    else
        delta_t = -delta_t; % 偶数次，取相反数
    end

    % 生成样本
    [xi, yi] = generate_samples(N, beta_opt + delta_t);

    % 更新总采样次数
    total_samples1 = total_samples1 + N;

    % 构建目标函数 g(beta)
    g_k_func = @(beta) compute_g(beta, xi, yi, alpha, N, lambda);

    % 计算 h 的次梯度
    h_grad = h_subgradient(beta1(:, k), alpha, lambda);

    % 求解优化子问题
    beta1(:, k+1) = solve_subproblem(beta1(:, k), g_k_func, h_grad, mu);

    % 更新
    d_k = beta1(:, k+1) - beta1(:, k);
    stepsize1(k) = norm(d_k); % 计算更新量
    norms1(k) = norm(beta1(:, k+1)); % 计算当前解的二范数

    % 当前解与最优解的距离
    beta_distances1(k) = norm(beta1(:, k+1) - beta_opt);

    % 自适应样本大小更新
    if k > 1
        N = update_sample_size(d_k, k, mu, C_g, alpha_g);
        N_history1(k) = N;
    end

    % 记录累计的系统时间
    elapsed_time1(k) = sum(elapsed_time1(1:k-1)) + toc(iteration_start_time);

    % 收敛检查
    if norm(d_k) < error_threshold
        break;
    end

    % 检查是否超过最大运行时间
    if toc(global_start_time) > max_runtime
        fprintf('运行达到最大时间限制：%d秒\n', max_runtime);
        break;
    end
end

% 记录结果
final_solution = beta1(:, k+1);
fprintf('最终解: %s\n', mat2str(final_solution, 4));
fprintf('当前解与最优解的距离: %f\n', beta_distances1(k));
fprintf('总采样次数: %d\n', total_samples1);

% ----------------绘图----------------
figure;

% 绘制样本数量随时间变化
subplot(3, 1, 1);
plot(1:k, N_history1(1:k), '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('样本数量');
title('样本数量随时间变化');

% 绘制当前解与最优解的距离随时间变化
subplot(3, 1, 2);
plot(1:max_iter, beta_distances1, '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('与最优解的距离');
title('当前解与最优解的距离随时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

% 绘制误差随系统时间变化
subplot(3, 1, 3);
error = beta_distances1; % 误差就是当前解与最优解的距离
plot(elapsed_time1, error, '-o', 'LineWidth', 1.5); % 使用 elapsed_time 作为横坐标
xlabel('系统时间（秒）');
ylabel('误差');
title('误差随系统时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

%--------------算法2-----------------

% 存储结果
beta2 = zeros(dim, max_iter);
beta2(:, 1) = beta0;
stepsize2 = zeros(1, max_iter);
norms2 = zeros(1, max_iter); % 存储步长二范数
total_samples2 = 0; % 总采样数
N_history2 = zeros(1, max_iter); % 存储每次更新的 N
beta_distances2 = zeros(1, max_iter); % 当前解与最优解的距离
elapsed_time2 = zeros(1, max_iter); % 累计的系统时间

% 记录总运行开始时间
global_start_time = tic;

% 运行自适应样本大小近似点梯度法
for k = 1:max_iter
    % 启动计时器
    iteration_start_time = tic;

    % 分布扰动 delta_t 随时间衰减
    delta_t = delta_t_max * k^(-2); % 偏移随迭代次数的二次衰减

    % 奇数次迭代时增加delta_t，偶数次时减少delta_t
    if mod(k, 2) == 1
        delta_t = delta_t; % 奇数次，保留delta_t
    else
        delta_t = -delta_t; % 偶数次，取相反数
    end

    % 生成样本
    [xi, yi] = generate_samples(N, beta_opt + delta_t);

    % 更新总采样次数
    total_samples2 = total_samples2 + N;

    % 构建目标函数 g(beta)
    g_k_func = @(beta) compute_g(beta, xi, yi, alpha, N, lambda);

    % 计算 h 的次梯度
    h_grad = h_subgradient(beta2(:, k), alpha, lambda);

    % 求解优化子问题
    beta2(:, k+1) = solve_subproblem(beta2(:, k), g_k_func, h_grad, mu);

    % 更新
    d_k = beta2(:, k+1) - beta2(:, k);
    stepsize2(k) = norm(d_k); % 计算更新量
    norms2(k) = norm(beta2(:, k+1)); % 计算当前解的二范数

    % 当前解与最优解的距离
    beta_distances2(k) = norm(beta2(:, k+1) - beta_opt);

    % 自适应样本大小更新
    if k > 1
        N = update_sample_size_fixed(d_k, k, mu, C_g, alpha_g);
        N_history2(k) = N;
    end

    % 记录累计的系统时间
    elapsed_time2(k) = sum(elapsed_time2(1:k-1)) + toc(iteration_start_time);

    % 收敛检查
    if norm(d_k) < error_threshold
        break;
    end

    % 检查是否超过最大运行时间
    if toc(global_start_time) > max_runtime
        fprintf('运行达到最大时间限制：%d秒\n', max_runtime);
        break;
    end
end

% 记录结果
final_solution = beta2(:, k+1);
fprintf('最终解: %s\n', mat2str(final_solution, 4));
fprintf('当前解与最优解的距离: %f\n', beta_distances2(k));
fprintf('总采样次数: %d\n', total_samples2);

% ----------------绘图----------------
figure;

% 绘制样本数量随时间变化
subplot(3, 1, 1);
plot(1:k, N_history2(1:k), '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('样本数量');
title('样本数量随时间变化');

% 绘制当前解与最优解的距离随时间变化
subplot(3, 1, 2);
plot(1:max_iter, beta_distances2, '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('与最优解的距离');
title('当前解与最优解的距离随时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

% 绘制误差随系统时间变化
subplot(3, 1, 3);
error = beta_distances2; % 误差就是当前解与最优解的距离
plot(elapsed_time2, error, '-o', 'LineWidth', 1.5); % 使用 elapsed_time 作为横坐标
xlabel('系统时间（秒）');
ylabel('误差');
title('误差随系统时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

%-----------------算法3-------------------
% 存储结果
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
    g_k_func = @(beta) compute_g3(beta, xi_all, yi_all, alpha, lambda);

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


%--------------算法4-----------------

% 存储结果
beta4 = zeros(dim, max_iter);
beta4(:, 1) = beta0;
stepsize4 = zeros(1, max_iter);
norms4 = zeros(1, max_iter); % 存储步长二范数
total_samples4 = 0; % 总采样数
N_history4 = zeros(1, max_iter); % 存储每次更新的 N
beta_distances4 = zeros(1, max_iter); % 当前解与最优解的距离
elapsed_time4 = zeros(1, max_iter); % 累计的系统时间

% 记录总运行开始时间
global_start_time = tic;

% 运行自适应样本大小近似点梯度法
for k = 1:max_iter
    % 启动计时器
    iteration_start_time = tic;

    % 分布扰动 delta_t 随时间衰减
    delta_t = delta_t_max * k^(-2); % 偏移随迭代次数的二次衰减

    % 奇数次迭代时增加delta_t，偶数次时减少delta_t
    if mod(k, 2) == 1
        delta_t = delta_t; % 奇数次，保留delta_t
    else
        delta_t = -delta_t; % 偶数次，取相反数
    end

    % 生成样本
    [xi, yi] = generate_samples(N, beta_opt + delta_t);

    % 更新总采样次数
    total_samples4 = total_samples4 + N;

    % 构建目标函数 g(beta)
    g_k_func = @(beta) compute_g(beta, xi, yi, alpha, N, lambda);

    % 计算 h 的次梯度
    h_grad = h_subgradient(beta4(:, k), alpha, lambda);

    % 求解优化子问题
    beta4(:, k+1) = solve_subproblem(beta4(:, k), g_k_func, h_grad, mu);

    % 更新
    d_k = beta4(:, k+1) - beta4(:, k);
    stepsize4(k) = norm(d_k); % 计算更新量
    norms4(k) = norm(beta4(:, k+1)); % 计算当前解的二范数

    % 当前解与最优解的距离
    beta_distances4(k) = norm(beta4(:, k+1) - beta_opt);

    % 自适应样本大小更新
    if k > 1
        N = 100;
        N_history4(k) = N;
    end

    % 记录累计的系统时间
    elapsed_time4(k) = sum(elapsed_time4(1:k-1)) + toc(iteration_start_time);

    % 收敛检查
    if norm(d_k) < error_threshold
        break;
    end

    % 检查是否超过最大运行时间
    if toc(global_start_time) > max_runtime
        fprintf('运行达到最大时间限制：%d秒\n', max_runtime);
        break;
    end
end

% 记录结果
final_solution = beta4(:, k+1);
fprintf('最终解: %s\n', mat2str(final_solution, 4));
fprintf('当前解与最优解的距离: %f\n', beta_distances4(k));
fprintf('总采样次数: %d\n', total_samples4);

% ----------------绘图----------------
figure;

% 绘制样本数量随时间变化
subplot(3, 1, 1);
plot(1:k, N_history4(1:k), '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('样本数量');
title('样本数量随时间变化');

% 绘制当前解与最优解的距离随时间变化
subplot(3, 1, 2);
plot(1:max_iter, beta_distances4, '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('与最优解的距离');
title('当前解与最优解的距离随时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

% 绘制误差随系统时间变化
subplot(3, 1, 3);
error = beta_distances4; % 误差就是当前解与最优解的距离
plot(elapsed_time4, error, '-o', 'LineWidth', 1.5); % 使用 elapsed_time 作为横坐标
xlabel('系统时间（秒）');
ylabel('误差');
title('误差随系统时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

%--------------算法5-----------------

% 存储结果
beta5 = zeros(dim, max_iter);
beta5(:, 1) = beta0;
stepsize5 = zeros(1, max_iter);
norms5 = zeros(1, max_iter); % 存储步长二范数
total_samples5 = 0; % 总采样数
N_history5 = zeros(1, max_iter); % 存储每次更新的 N
beta_distances5 = zeros(1, max_iter); % 当前解与最优解的距离
elapsed_time5 = zeros(1, max_iter); % 累计的系统时间

% 记录总运行开始时间
global_start_time = tic;

% 运行自适应样本大小近似点梯度法
for k = 1:max_iter
    % 启动计时器
    iteration_start_time = tic;

    % 分布扰动 delta_t 随时间衰减
    delta_t = delta_t_max * k^(-2); % 偏移随迭代次数的二次衰减

    % 奇数次迭代时增加delta_t，偶数次时减少delta_t
    if mod(k, 2) == 1
        delta_t = delta_t; % 奇数次，保留delta_t
    else
        delta_t = -delta_t; % 偶数次，取相反数
    end

    % 生成样本
    [xi, yi] = generate_samples(N, beta_opt + delta_t);

    % 更新总采样次数
    total_samples5 = total_samples5 + N;

    % 构建目标函数 g(beta)
    g_k_func = @(beta) compute_g(beta, xi, yi, alpha, N, lambda);

    % 计算 h 的次梯度
    h_grad = h_subgradient(beta5(:, k), alpha, lambda);

    % 求解优化子问题
    beta5(:, k+1) = solve_subproblem(beta5(:, k), g_k_func, h_grad, mu);

    % 更新
    d_k = beta5(:, k+1) - beta5(:, k);
    stepsize5(k) = norm(d_k); % 计算更新量
    norms5(k) = norm(beta5(:, k+1)); % 计算当前解的二范数

    % 当前解与最优解的距离
    beta_distances5(k) = norm(beta5(:, k+1) - beta_opt);

    % 自适应样本大小更新
    if k > 1
        N = 1000;
        N_history5(k) = N;
    end

    % 记录累计的系统时间
    elapsed_time5(k) = sum(elapsed_time5(1:k-1)) + toc(iteration_start_time);

    % 收敛检查
    if norm(d_k) < error_threshold
        break;
    end

    % 检查是否超过最大运行时间
    if toc(global_start_time) > max_runtime
        fprintf('运行达到最大时间限制：%d秒\n', max_runtime);
        break;
    end
end

% 记录结果
final_solution = beta5(:, k+1);
fprintf('最终解: %s\n', mat2str(final_solution, 4));
fprintf('当前解与最优解的距离: %f\n', beta_distances5(k));
fprintf('总采样次数: %d\n', total_samples5);

% ----------------绘图----------------
figure;

% 绘制样本数量随时间变化
subplot(3, 1, 1);
plot(1:k, N_history5(1:k), '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('样本数量');
title('样本数量随时间变化');

% 绘制当前解与最优解的距离随时间变化
subplot(3, 1, 2);
plot(1:max_iter, beta_distances5, '-o', 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('与最优解的距离');
title('当前解与最优解的距离随时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

% 绘制误差随系统时间变化
subplot(3, 1, 3);
error = beta_distances5; % 误差就是当前解与最优解的距离
plot(elapsed_time5, error, '-o', 'LineWidth', 1.5); % 使用 elapsed_time 作为横坐标
xlabel('系统时间（秒）');
ylabel('误差');
title('误差随系统时间变化');
set(gca, 'YScale', 'log'); % 设置 Y 轴为对数刻度

% ----------------子函数定义----------------

% 生成样本函数
function [xi, yi] = generate_samples(N, beta_true)
    dim = length(beta_true);
    xi = 2 * rand(N, dim) - 1; 
    noise = randn(N, 1); % 随机噪声
    yi = xi * beta_true + noise; % y_i = beta_true' * x_i + 噪声
end

% 定义目标函数 g(beta)
function g_k_func = compute_g(beta, xi, yi, alpha, N, lambda)
    % 计算残差
    residual = yi - xi * beta;
    % 计算L1损失
    li_loss = abs(residual);
    % 总目标函数值
    g_k_func = (1 + alpha * norm(beta, 1)) * lambda + mean(li_loss); % L1 损失取均值
end

% 定义目标函数 g(beta)
function g_k_func = compute_g3(beta, xi, yi, alpha, lambda)
    % 计算残差
    residual = yi - xi * beta;
    % 计算L1损失
    li_loss = abs(residual);
    % 总目标函数值
    g_k_func = (1 + alpha * norm(beta, 1))*lambda + mean(li_loss); % L1 损失取均值
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

% 自适应样本大小更新
function N = update_sample_size(d_k, k, mu, C_g, alpha_g)
    % 根据更新公式调整样本数
    grad_norm_sq = (mu / 2) * norm(d_k)^2;
    N = min(ceil(k^2.1), ceil((C_g / (mu * grad_norm_sq))^(1 / alpha_g)));
end

% 自适应样本大小更新
function N = update_sample_size_fixed(d_k, k, mu, C_g, alpha_g)
    % 根据更新公式调整样本数
    grad_norm_sq = (mu / 2) * norm(d_k)^2;
    N = ceil(k^2.1);
end