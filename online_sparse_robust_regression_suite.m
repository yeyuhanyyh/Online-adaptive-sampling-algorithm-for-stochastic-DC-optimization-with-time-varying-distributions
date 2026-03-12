function summary = online_sparse_robust_regression_suite(which_set, results_dir)
%ONLINE_SPARSE_ROBUST_REGRESSION_SUITE
% Reproduce the online sparse robust regression experiments from
% "An Online Adaptive Sampling Algorithm for Stochastic Difference-of-convex
% Optimization with Time-varying Distributions".
%
% Usage:
%   summary = online_sparse_robust_regression_suite('main', results_dir)
%   summary = online_sparse_robust_regression_suite('appendix', results_dir)
%
% 'main' reproduces the two experiments shown in Figures 1-2 of the paper.
% 'appendix' reproduces the three extra experiments shown in Figures 3-4.
%
% The code compares five methods:
%   1) Adaptive ospDCA (ours)
%   2) ospDCA with predetermined sample size t^2.1
%   3) S(p)DCA with one new sample per iteration and cumulative SAA
%   4) ospDCA with fixed sample size 100
%   5) ospDCA with fixed sample size 1000

    if nargin < 1 || isempty(which_set)
        which_set = 'main';
    end
    if nargin < 2 || isempty(results_dir)
        results_dir = fullfile(pwd, 'results');
    end

    if isstring(which_set)
        which_set = char(which_set);
    end
    which_set = lower(which_set);

    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    assert(exist('fminunc', 'file') == 2, ...
        'Optimization Toolbox is required because this code uses fminunc.');

    switch which_set
        case 'main'
            experiments = build_main_experiments();
            behavior_file = fullfile(results_dir, 'main_figure1_behavior.png');
            sample_file = fullfile(results_dir, 'main_figure2_sample_size.png');
            summary_file = fullfile(results_dir, 'main_summary.mat');
        case 'appendix'
            experiments = build_appendix_experiments();
            behavior_file = fullfile(results_dir, 'appendix_figure3_behavior.png');
            sample_file = fullfile(results_dir, 'appendix_figure4_sample_size.png');
            summary_file = fullfile(results_dir, 'appendix_summary.mat');
        otherwise
            error('Unknown experiment set ''%s''. Use ''main'' or ''appendix''.', which_set);
    end

    summary = struct();
    summary.which_set = char(which_set);
    summary.created_at = datestr(now, 30);

    % -------- FIX FOR MATLAB STRUCT ASSIGNMENT ERROR --------
    % Do not preallocate with repmat(struct(), ...), because that creates
    % a struct array with zero fields. Later indexed assignment with a
    % non-empty struct then fails with:
    % "Subscripted assignment between dissimilar structures."
    fprintf('\n=== Running %s ===\n', experiments(1).title_text);
    first_result = run_single_experiment(experiments(1));

    summary.experiments = repmat(first_result, 1, numel(experiments));
    summary.experiments(1) = first_result;

    for i = 2:numel(experiments)
        fprintf('\n=== Running %s ===\n', experiments(i).title_text);
        summary.experiments(i) = run_single_experiment(experiments(i));
    end
    % -------------------------------------------------------

    plot_behavior_figure(summary.experiments, behavior_file);
    plot_sample_size_figure(summary.experiments, sample_file);
    save(summary_file, 'summary', '-v7.3');

    fprintf('\nSaved behavior figure: %s\n', behavior_file);
    fprintf('Saved sample-size figure: %s\n', sample_file);
    fprintf('Saved summary mat-file: %s\n', summary_file);
end

function experiments = build_main_experiments()
    base = default_config();

    exp1 = base;
    exp1.name = 'main_a_p50';
    exp1.label = '(a)';
    exp1.p = 50;
    exp1.beta_opt = make_sparse_beta(exp1.p, [10; -15]);
    exp1.delta_scale = 100;
    exp1.runtime_limit = 5;
    exp1.seed = 40;
    exp1.title_text = '(a) p = 50, beta_opt = [10, -15, 0, ..., 0]';

    exp2 = base;
    exp2.name = 'main_b_p200';
    exp2.label = '(b)';
    exp2.p = 200;
    exp2.beta_opt = make_sparse_beta(exp2.p, [10; -15]);
    exp2.delta_scale = 100;
    exp2.runtime_limit = 20;
    exp2.seed = 40;
    exp2.title_text = '(b) p = 200, beta_opt = [10, -15, 0, ..., 0]';

    experiments = [exp1, exp2];
end

function experiments = build_appendix_experiments()
    base = default_config();

    exp1 = base;
    exp1.name = 'appendix_c_p100';
    exp1.label = '(c)';
    exp1.p = 100;
    exp1.beta_opt = make_sparse_beta(exp1.p, [5; -5]);
    exp1.delta_scale = 100;
    exp1.runtime_limit = 8;
    exp1.seed = 40;
    exp1.title_text = '(c) p = 100, beta_opt = [5, -5, 0, ..., 0]';

    exp2 = base;
    exp2.name = 'appendix_d_p200';
    exp2.label = '(d)';
    exp2.p = 200;
    exp2.beta_opt = make_sparse_beta(exp2.p, [5; -5]);
    exp2.delta_scale = 100;
    exp2.runtime_limit = 20;
    exp2.seed = 40;
    exp2.title_text = '(d) p = 200, beta_opt = [5, -5, 0, ..., 0]';

    exp3 = base;
    exp3.name = 'appendix_e_p50_large_shift';
    exp3.label = '(e)';
    exp3.p = 50;
    exp3.beta_opt = make_sparse_beta(exp3.p, [5; -5]);
    exp3.delta_scale = 5000;
    exp3.runtime_limit = 5;
    exp3.seed = 40;
    exp3.title_text = '(e) p = 50, beta_opt = [5, -5, 0, ..., 0], delta_t = (-1)^t 5000 t^{-2} 1_p';

    experiments = [exp1, exp2, exp3];
end

function cfg = default_config()
    cfg = struct();
    cfg.alpha = 1;
    cfg.lambda = 0.01;
    cfg.mu = 1;
    cfg.C_g = 1;
    cfg.alpha_g = 0.45;
    cfg.sample_exponent = 2.1;
    cfg.initial_sample_size = 10;
    cfg.max_iter = 1e6;
    cfg.step_tolerance = 1e-10;
    cfg.fixed_sample_sizes = [100, 1000];
    cfg.optim_options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'Display', 'off', ...
        'MaxIterations', 400, ...
        'MaxFunctionEvaluations', 5e4, ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-12);
end

function beta = make_sparse_beta(p, nz_values)
    beta = zeros(p, 1);
    beta(1:numel(nz_values)) = nz_values(:);
end

function experiment = run_single_experiment(cfg)
    cfg_to_save = cfg;
    if isfield(cfg_to_save, 'optim_options')
        cfg_to_save = rmfield(cfg_to_save, 'optim_options');
    end

    experiment = struct();
    experiment.config = cfg_to_save;
    experiment.adaptive = run_ospdca(cfg, 'adaptive', cfg.seed + 101);
    experiment.nonadaptive = run_ospdca(cfg, 'polynomial', cfg.seed + 202);
    experiment.spdca = run_spdca(cfg, cfg.seed + 303);

    cfg_fixed = cfg;
    cfg_fixed.fixed_sample_size = cfg.fixed_sample_sizes(1);
    experiment.fixed100 = run_ospdca(cfg_fixed, 'fixed', cfg.seed + 404);

    cfg_fixed.fixed_sample_size = cfg.fixed_sample_sizes(2);
    experiment.fixed1000 = run_ospdca(cfg_fixed, 'fixed', cfg.seed + 505);

    fprintf('  Adaptive ospDCA:  final distance = %.4e, iterations = %d, time = %.2fs\n', ...
        experiment.adaptive.final_distance, experiment.adaptive.n_iter, experiment.adaptive.elapsed(end));
    fprintf('  ospDCA (t^2.1):   final distance = %.4e, iterations = %d, time = %.2fs\n', ...
        experiment.nonadaptive.final_distance, experiment.nonadaptive.n_iter, experiment.nonadaptive.elapsed(end));
    fprintf('  S(p)DCA:          final distance = %.4e, iterations = %d, time = %.2fs\n', ...
        experiment.spdca.final_distance, experiment.spdca.n_iter, experiment.spdca.elapsed(end));
    fprintf('  ospDCA (100):     final distance = %.4e, iterations = %d, time = %.2fs\n', ...
        experiment.fixed100.final_distance, experiment.fixed100.n_iter, experiment.fixed100.elapsed(end));
    fprintf('  ospDCA (1000):    final distance = %.4e, iterations = %d, time = %.2fs\n', ...
        experiment.fixed1000.final_distance, experiment.fixed1000.n_iter, experiment.fixed1000.elapsed(end));
end

function result = run_ospdca(cfg, mode, seed)
    rng(seed, 'twister');

    beta = zeros(cfg.p, 1);
    distances = nan(cfg.max_iter, 1);
    elapsed = nan(cfg.max_iter, 1);
    sample_size = nan(cfg.max_iter, 1);
    total_samples = 0;
    next_adaptive_sample_size = cfg.initial_sample_size;
    wall_clock = tic;

    for k = 1:cfg.max_iter
        switch lower(mode)
            case 'adaptive'
                N = next_adaptive_sample_size;
            case 'polynomial'
                N = max(1, ceil(k^cfg.sample_exponent));
            case 'fixed'
                N = cfg.fixed_sample_size;
            otherwise
                error('Unknown mode "%s".', mode);
        end

        beta_true_t = current_beta_truth(cfg, k);
        [X, y] = generate_batch(N, beta_true_t);
        total_samples = total_samples + N;

        gfun = @(b) sample_average_g(b, X, y, cfg.alpha, cfg.lambda);
        hgrad = h_subgradient(beta, cfg.alpha, cfg.lambda);
        beta_next = solve_subproblem(beta, gfun, hgrad, cfg.mu, cfg.optim_options);

        step = beta_next - beta;
        beta = beta_next;

        distances(k) = norm(beta - cfg.beta_opt);
        elapsed(k) = toc(wall_clock);
        sample_size(k) = N;

        if strcmpi(mode, 'adaptive')
            next_adaptive_sample_size = adaptive_sample_size(step, k, cfg);
        end

        if norm(step) < cfg.step_tolerance || elapsed(k) >= cfg.runtime_limit
            break;
        end
    end

    result = pack_result(mode, beta, distances, elapsed, sample_size, total_samples, seed, k);
end

function result = run_spdca(cfg, seed)
    rng(seed, 'twister');

    beta = zeros(cfg.p, 1);
    distances = nan(cfg.max_iter, 1);
    elapsed = nan(cfg.max_iter, 1);
    sample_size = nan(cfg.max_iter, 1);
    total_samples = 0;
    X_all = zeros(0, cfg.p);
    y_all = zeros(0, 1);
    wall_clock = tic;

    for k = 1:cfg.max_iter
        beta_true_t = current_beta_truth(cfg, k);
        [X_new, y_new] = generate_batch(1, beta_true_t);
        X_all = [X_all; X_new]; %#ok<AGROW>
        y_all = [y_all; y_new]; %#ok<AGROW>
        total_samples = total_samples + 1;

        gfun = @(b) sample_average_g(b, X_all, y_all, cfg.alpha, cfg.lambda);
        hgrad = h_subgradient(beta, cfg.alpha, cfg.lambda);
        beta_next = solve_subproblem(beta, gfun, hgrad, cfg.mu, cfg.optim_options);

        step = beta_next - beta;
        beta = beta_next;

        distances(k) = norm(beta - cfg.beta_opt);
        elapsed(k) = toc(wall_clock);
        sample_size(k) = 1;

        if norm(step) < cfg.step_tolerance || elapsed(k) >= cfg.runtime_limit
            break;
        end
    end

    result = pack_result('spdca', beta, distances, elapsed, sample_size, total_samples, seed, k);
end

function result = pack_result(name, beta, distances, elapsed, sample_size, total_samples, seed, k)
    result = struct();
    result.name = name;
    result.seed = seed;
    result.n_iter = k;
    result.iterations = (1:k)';
    result.distance = distances(1:k);
    result.elapsed = elapsed(1:k);
    result.sample_size = sample_size(1:k);
    result.total_samples = total_samples;
    result.final_beta = beta;
    result.final_distance = distances(k);
end

function beta_true_t = current_beta_truth(cfg, t)
    shift_scalar = ((-1)^t) * cfg.delta_scale * (t^(-2));
    beta_true_t = cfg.beta_opt + shift_scalar * ones(cfg.p, 1);
end

function [X, y] = generate_batch(N, beta_true)
    p = numel(beta_true);
    X = 2 * rand(N, p) - 1;
    y = X * beta_true + randn(N, 1);
end

function value = sample_average_g(beta, X, y, alpha, lambda)
    residual = y - X * beta;
    value = mean(abs(residual)) + lambda * sum(1 + alpha * abs(beta));
end

function grad = h_subgradient(beta, alpha, lambda)
    grad = zeros(size(beta));
    active = abs(beta) > (1 / alpha);
    grad(active) = lambda * alpha * sign(beta(active));
end

function beta_next = solve_subproblem(beta_prev, gfun, hgrad, mu, optim_options)
    objective = @(z) gfun(z) - (z - beta_prev)' * hgrad + 0.5 * mu * norm(z - beta_prev)^2;
    beta_next = fminunc(objective, beta_prev, optim_options);
end

function N_next = adaptive_sample_size(step, k, cfg)
    step_energy = max((cfg.mu / 2) * norm(step)^2, eps);
    adaptive_term = ceil((cfg.C_g / (cfg.mu * step_energy))^(1 / cfg.alpha_g));
    polynomial_cap = max(1, ceil((k + 1)^cfg.sample_exponent));
    N_next = max(1, min(polynomial_cap, adaptive_term));
end

function plot_behavior_figure(experiments, save_path)
    n_exp = numel(experiments);
    fig = figure('Color', 'w', 'Position', [100, 100, 1200, 420 * n_exp]);

    for i = 1:n_exp
        exp_cfg = experiments(i).config;

        subplot(n_exp, 2, 2 * i - 1);
        hold on;
        semilogy(experiments(i).adaptive.iterations, experiments(i).adaptive.distance, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Adaptive ospDCA (ours)');
        semilogy(experiments(i).nonadaptive.iterations, experiments(i).nonadaptive.distance, 'g-', 'LineWidth', 1.5, 'DisplayName', 'ospDCA');
        semilogy(experiments(i).spdca.iterations, experiments(i).spdca.distance, 'r-', 'LineWidth', 1.5, 'DisplayName', 'S(p)DCA');
        semilogy(experiments(i).fixed100.iterations, experiments(i).fixed100.distance, 'c-', 'LineWidth', 1.5, 'DisplayName', 'ospDCA with SAA 100');
        semilogy(experiments(i).fixed1000.iterations, experiments(i).fixed1000.distance, 'm-', 'LineWidth', 1.5, 'DisplayName', 'ospDCA with SAA 1000');
        hold off;
        grid on;
        xlabel('Iterations');
        ylabel('Distance to Optimal Solution');
        title(sprintf('Performance Over Iterations | %s, p = %d', exp_cfg.label, exp_cfg.p));
        legend('Location', 'best');

        subplot(n_exp, 2, 2 * i);
        hold on;
        semilogy(experiments(i).adaptive.elapsed, experiments(i).adaptive.distance, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Adaptive ospDCA (ours)');
        semilogy(experiments(i).nonadaptive.elapsed, experiments(i).nonadaptive.distance, 'g-', 'LineWidth', 1.5, 'DisplayName', 'ospDCA');
        semilogy(experiments(i).spdca.elapsed, experiments(i).spdca.distance, 'r-', 'LineWidth', 1.5, 'DisplayName', 'S(p)DCA');
        semilogy(experiments(i).fixed100.elapsed, experiments(i).fixed100.distance, 'c-', 'LineWidth', 1.5, 'DisplayName', 'ospDCA with SAA 100');
        semilogy(experiments(i).fixed1000.elapsed, experiments(i).fixed1000.distance, 'm-', 'LineWidth', 1.5, 'DisplayName', 'ospDCA with SAA 1000');
        hold off;
        grid on;
        xlabel('Time (seconds)');
        ylabel('Distance to Optimal Solution');
        title(sprintf('Performance Over Time | %s, p = %d', exp_cfg.label, exp_cfg.p));
        legend('Location', 'best');
    end

    save_figure(fig, save_path);
    close(fig);
end

function plot_sample_size_figure(experiments, save_path)
    n_exp = numel(experiments);
    fig = figure('Color', 'w', 'Position', [100, 100, 480 * n_exp, 500]);

    for i = 1:n_exp
        exp_cfg = experiments(i).config;
        subplot(1, n_exp, i);
        hold on;
        scatter(experiments(i).nonadaptive.iterations, experiments(i).nonadaptive.sample_size, 18, 'g', 'filled', 'DisplayName', 'ospDCA');
        scatter(experiments(i).adaptive.iterations, experiments(i).adaptive.sample_size, 18, 'b', 'filled', 'DisplayName', 'Adaptive ospDCA (ours)');
        hold off;
        grid on;
        xlabel('Iterations');
        ylabel('Sample Size');
        title(sprintf('Sample Size Over Iterations | %s, p = %d', exp_cfg.label, exp_cfg.p));
        legend('Location', 'northwest');
    end

    save_figure(fig, save_path);
    close(fig);
end

function save_figure(fig, save_path)
    [folder, ~, ~] = fileparts(save_path);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end

    try
        exportgraphics(fig, save_path, 'Resolution', 300);
    catch
        print(fig, save_path, '-dpng', '-r300');
    end
end