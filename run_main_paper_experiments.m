clear; clc;
results_dir = fullfile(pwd, 'results');
summary = online_sparse_robust_regression_suite('main', results_dir); %#ok<NASGU>
fprintf('\nSaved main-paper figures and summary under:\n  %s\n', results_dir);