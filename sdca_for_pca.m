% 参数设置
clear
L = 1; % 可以根据需要调整的参数
max_iter = 1000; % 最大迭代次数
N_g = 10; % g的样本数量
N_h = 10; % h的样本数量
x0 = [0,0]; % 初始化点
mu = 10; % 二次近端项序列
error_threshold = 1e-3; % 收敛阈值
C_g = sqrt(2) * 8 * L * (L + 1) * (2 * L + 1)/100;
C_h = (L + 1)^2;
alpha_g = 0.45; % 更新为0.45
alpha_h = 1;    % 更新为1

% 存储结果
x = zeros(2, max_iter);
x(:, 1) = x0;
stepsize = zeros(1, max_iter);
norms = zeros(1, max_iter); % 存储二范数
total_samples_g = 0; % 初始化g的总采样次数
total_samples_h = 0; % 初始化h的总采样次数
N0 = zeros(2, max_iter); % 存储自适应采样规模初始猜测
N_g_history = zeros(1, max_iter); % 存储每次更新的 N_g
N_h_history = zeros(1, max_iter); % 存储每次更新的 N_h

% 运行自适应样本大小 SDCA
for k = 1:max_iter
    % 生成单位球上的均匀分布样本
    xi_g = generate_samples(N_g); % g的样本
    zeta_h = generate_samples(N_h); % h的样本

    % 更新总采样次数
    total_samples_g = total_samples_g + N_g;
    total_samples_h = total_samples_h + N_h;

    % 构建局部近似模型
    g_k_func = compute_g(x(:, k), xi_g, L); % 自定义函数计算g_k
    y_k = compute_subgradient(x(:, k), zeta_h, L); % 自定义函数计算y_k
    
    % 求解优化子问题
    x(:, k+1) = solve_subproblem(x(:, k), g_k_func, y_k, mu); % 自定义函数求解子问题

    % 更新
    d_k = x(:, k+1) - x(:, k); 
    stepsize(k) = norm(d_k); % 计算更新量
    norms(k) = norm(x(:, k+1)); % 计算当前解的二范数
    
    % 收敛检查
    if total_samples_g + total_samples_h > 100000
        break;
    end
    
    % 自适应样本大小更新
    if k > 1
        [N0(:,k), N_g, N_h] = update_sample_size(d_k, k, mu, C_g, C_h, alpha_g, alpha_h, L);
    end
    
    % 存储每次更新的 N_g 和 N_h
    N_g_history(k) = N_g;
    N_h_history(k) = N_h;
end

% 记录结果
final_solution = x(:, k);
fprintf('最终解: [%f, %f]\n', final_solution(1), final_solution(2));
fprintf('最终下降步长: %f\n', stepsize(k));
fprintf('g的总采样次数: %d\n', total_samples_g); % 输出g的总采样次数
fprintf('h的总采样次数: %d\n', total_samples_h); % 输出h的总采样次数

% 可视化结果
figure;
semilogy(stepsize(1:k));
xlabel('迭代次数');
ylabel('下降步长');
title('Adapt Sample Size SDCA 收敛性');
grid on;

% 绘制迭代路径
figure;
plot(x(1, 1:k), x(2, 1:k), '-o', 'LineWidth', 2);
hold on;

% 绘制单位球
theta = linspace(0, 2*pi, 100);
unit_circle_x = cos(theta);
unit_circle_y = sin(theta);
plot(unit_circle_x, unit_circle_y, 'r--', 'LineWidth', 1.5); % 单位球边界

% 标记初始点和最终点
plot(x(1, 1), x(2, 1), 'ro', 'MarkerSize', 8); % 初始点
plot(x(1, k), x(2, k), 'go', 'MarkerSize', 8); % 最终点
xlabel('x_1');
ylabel('x_2');
title('迭代路径及单位球');
axis equal; % 设置坐标轴相等
grid on;
hold off;

% 绘制解的二范数
figure;
plot(1:k, norms(1:k), '-o', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('二范数 ||x_k||');
title('解的二范数图像');
grid on;

% 绘制每次更新的 N_g 和 N_h
figure;
subplot(2, 1, 1);
plot(1:k, N_g_history(1:k), '-o', 'LineWidth', 2);
hold on;
k_values = 1:k;
k_1_2 = k_values.^2.2;
plot(k_values, k_1_2, 'm--', 'LineWidth', 1.5); % k^2.2 曲线
xlabel('迭代次数');
ylabel('g的样本数量 N_g');
title('每次更新的样本数量 N_g');
grid on;
hold off;

subplot(2, 1, 2);
plot(1:k, N_h_history(1:k), '-o', 'LineWidth', 2);
hold on;
k_values = 1:k;
k_1_2 = k_values.^2;
plot(k_values, k_1_2, 'm--', 'LineWidth', 1.5); % k^1.6 曲线
xlabel('迭代次数');
ylabel('h的样本数量 N_h');
title('每次更新的样本数量 N_h');
grid on;
hold off;

% 自定义函数示例
function samples = generate_samples(N)
    % 生成N个单位球上的样本
    samples = zeros(N, 2);
    for i = 1:N
        samples(i, :) = generate_uniform_on_upper_half_circle; % 生成均匀分布的单位向量
    end
end

function samples = generate_uniform_on_upper_half_circle
    % 生成N个上半平面单位球上的均匀分布样本
    theta = rand(1, 1) * pi; % 生成0到π之间的均匀角度
    samples = [cos(theta); sin(theta)]'; % 计算对应的坐标
end

function g_k_func = compute_g(x, samples, L)
    % 返回一个关于x的函数句柄
    g_k_func = @(z) (L/2) * norm(z)^2 - (1/2) * mean(arrayfun(@(i) (z' * samples(i, :)'), 1:size(samples, 1)));
end

function y_k = compute_subgradient(x, samples, L)
    % 计算子梯度
    grad_h = (L * x) + mean(samples, 1)'; % h的梯度
    y_k = grad_h; % 取决于具体情况，这里假设y_k等于grad_h
end

function x_new = solve_subproblem(x, g_k_func, y_k, mu)
    % 求解优化子问题
    fun = @(z) g_k_func(z) - (z - x)' * y_k + 0.5 * mu * norm(z - x)^2;

    % 设置约束条件
    options = optimoptions('fmincon', 'Display', 'off');
    x0 = x; % 从当前点开始

    % 定义非线性约束
    nonlin_con = @(z) deal(z(1)^2 + z(2)^2 - 1, []); % 不等式约束，等式约束为空

    % 使用 fmincon 进行优化
    x_new = fmincon(fun, x0, [], [], [], [], [], [], nonlin_con, options);
end

function [N0, N_g, N_h] = update_sample_size(d_k, k, mu, C_g, C_h, alpha_g, alpha_h, L)
    % 目标函数：最小化 N_g + N_h
    objective = @(N) N(1) + N(2); % N(1) = N_g, N(2) = N_h

    % 计算 d_k 的范数
    norm_d_k = norm(d_k);
    
    % 约束条件
    constraints = @(N) deal([], ...
        (0.5 * mu * norm_d_k^2) - (C_g / (mu * N(1)^alpha_g)) - (C_h / (2 * (2 * L - 1) * N(2)^alpha_h))); 

    % 初始猜测
    N0(1) = ((C_g / mu) / (0.25 * mu * norm_d_k^2))^(1 / alpha_g);
    N0(2) = ((C_h / (2 * (2 * L - 1))) / (0.25 * mu * norm_d_k^2))^(1 / alpha_h);

    % 设置选项
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

    % 使用 fmincon 求解优化问题
    [N_opt, ~] = fmincon(objective, N0, [], [], [], [], [1e-6, 1e-6], [], constraints, options); % 设置下界为小正数

    % 返回更新后的样本大小
    N_g = ceil(N_opt(1)); % 向上取整以确保样本数为整数
    N_h = ceil(N_opt(2));

    % 限制 N_g 和 N_h 的最大值
    if N_g > k^2.2&&N_h > k^2
        N_g = ceil(k^2.2);
        N_h = ceil(k^2);
    end
    if N_g > 20000
        N_g=20000;
    end
    if N_h > 10000
        N_h=10000;
    end    
end