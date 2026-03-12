figure;

% 绘制三种算法在迭代次数上的比较
subplot(2, 1, 1);
semilogy(1:max_iter, beta_distances1, '-o', 'LineWidth', 1.5, 'DisplayName', '算法1');
hold on;
semilogy(1:max_iter, beta_distances2, '-o', 'LineWidth', 1.5, 'DisplayName', '算法2');
semilogy(1:max_iter, beta_distances3, '-o', 'LineWidth', 1.5, 'DisplayName', '算法3');
hold off;

xlabel('迭代次数');
ylabel('与最优解的距离 (对数尺度)');
title('三种算法的距离随迭代次数变化');
legend('show', 'Location', 'best');
grid on;

% 绘制三种算法在运行时间上的比较
subplot(2, 1, 2);
semilogy(elapsed_time1, beta_distances1, '-o', 'LineWidth', 1.5, 'DisplayName', '算法1');
hold on;
%semilogy(elapsed_time2, beta_distances2, '-o', 'LineWidth', 1.5, 'DisplayName', '算法2');
%semilogy(elapsed_time3, beta_distances3, '-o', 'LineWidth', 1.5, 'DisplayName', '算法3');
hold off;

xlabel('系统时间（秒）');
ylabel('与最优解的距离 (对数尺度)');
title('三种算法的距离随运行时间变化');
legend('show', 'Location', 'best');
grid on;
