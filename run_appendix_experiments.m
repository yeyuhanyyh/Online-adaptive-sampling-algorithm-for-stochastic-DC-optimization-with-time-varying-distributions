clear; clc;
results_dir = fullfile(pwd, 'results');
summary = online_sparse_robust_regression_suite('appendix', results_dir); %#ok<NASGU>
fprintf('\nSaved appendix figures and summary under:\n  %s\n', results_dir);