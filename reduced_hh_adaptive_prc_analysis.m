function results = reduced_hh_adaptive_prc_analysis()
% reduced_hh_adaptive_prc_analysis
% Simulate the reduced Hodgkin-Huxley oscillator, compute a numerical
% ground-truth PRC from direct perturbation experiments, and fit it with
% an adaptive Fourier PRC estimator.

    cfg = default_config();
    results = run_analysis(cfg);
end


function cfg = default_config()
    cfg.Ib = 10.0;
    cfg.C = 1.0;
    cfg.gNa = 120.0;
    cfg.gK = 36.0;
    cfg.gL = 0.3;
    cfg.VNa = 115.0;
    cfg.VK = -12.0;
    cfg.VL = 10.6;

    cfg.dt = 0.01;
    cfg.transient_time = 1500.0;
    cfg.max_cycle_time = 200.0;
    cfg.threshold = 20.0;
    cfg.stim_amp = 0.05;

    cfg.n_phase_samples = 80;
    cfg.n_adaptive_samples = 1200;
    cfg.learning_mode = "adaptive";
    cfg.fixed_alpha = 0.01;
    cfg.order_list = 3:12;
    cfg.fit_order_list = [3 5 8 12];
    cfg.n_repeats_per_order = 24;
    cfg.summary_sample = 200;
    cfg.use_parallel = true;

    cfg.rng_seed = 7;
    cfg.output_prefix = 'reduced_hh';
end


function results = run_analysis(cfg)
    rng(cfg.rng_seed);

    orbit = compute_limit_cycle(cfg);
    phase_grid = ((0:cfg.n_phase_samples-1) + 0.5) ./ cfg.n_phase_samples;
    true_prc = compute_true_prc(cfg, orbit, phase_grid);
    use_parallel = cfg.use_parallel && can_use_parallel_pool();

    order_summary = struct([]);
    representative_models = cell(0, 1);

    for idx = 1:numel(cfg.order_list)
        M = cfg.order_list(idx);
        summary_mse_runs = zeros(cfg.n_repeats_per_order, 1);
        rep_models = cell(cfg.n_repeats_per_order, 1);
        rep_seeds = cfg.rng_seed + 10000 * idx + (1:cfg.n_repeats_per_order);

        if use_parallel
            parfor rep = 1:cfg.n_repeats_per_order
                model = run_adaptive_fit(cfg, orbit, phase_grid, true_prc, M, rep_seeds(rep));
                summary_idx = min(cfg.summary_sample, numel(model.mse_history));
                rep_models{rep} = model;
                summary_mse_runs(rep) = model.mse_history(summary_idx);
            end
        else
            for rep = 1:cfg.n_repeats_per_order
                model = run_adaptive_fit(cfg, orbit, phase_grid, true_prc, M, rep_seeds(rep));
                summary_idx = min(cfg.summary_sample, numel(model.mse_history));
                rep_models{rep} = model;
                summary_mse_runs(rep) = model.mse_history(summary_idx);
            end
        end

        order_summary(idx).order = M;
        order_summary(idx).summary_sample = cfg.summary_sample;
        order_summary(idx).summary_mse_runs = summary_mse_runs;
        order_summary(idx).mean_summary_mse = mean(summary_mse_runs);
        order_summary(idx).median_summary_mse = median(summary_mse_runs);
        order_summary(idx).q25_summary_mse = prctile(summary_mse_runs, 25);
        order_summary(idx).q75_summary_mse = prctile(summary_mse_runs, 75);

        if ismember(M, cfg.fit_order_list)
            [~, best_idx] = min(summary_mse_runs);
            representative_models{end + 1} = rep_models{best_idx};
        end
    end

    make_prc_fit_figure(cfg, phase_grid, true_prc, representative_models);
    make_order_summary_figure(cfg, order_summary);

    results.cfg = cfg;
    results.orbit = orbit;
    results.phase_grid = phase_grid;
    results.true_prc = true_prc;
    results.representative_models = representative_models;
    results.order_summary = order_summary;

    save([cfg.output_prefix '_adaptive_prc_results.mat'], 'results');
end


function orbit = compute_limit_cycle(cfg)
    tspan = 0:cfg.dt:cfg.transient_time;
    x0 = [0.0; 0.317];
    [~, X] = ode45(@(t, x) reduced_hh_rhs([], x, cfg, 0.0), tspan, x0);

    V = X(:, 1);
    crossings = find(V(1:end-1) < cfg.threshold & V(2:end) >= cfg.threshold);
    if numel(crossings) < 3
        error('Unable to find enough spikes to define the reduced-HH limit cycle.');
    end

    last_cross = crossings(end);
    prev_cross = crossings(end - 1);
    T = (last_cross - prev_cross) * cfg.dt;
    if T <= 0
        error('Estimated period is non-positive.');
    end

    idx = prev_cross:(last_cross-1);
    orbit.time = (0:numel(idx)-1)' * cfg.dt;
    orbit.state = X(idx, :);
    orbit.period = T;
    orbit.phase = orbit.time / T;
end


function prc = compute_true_prc(cfg, orbit, phase_grid)
    prc = zeros(size(phase_grid));
    for k = 1:numel(phase_grid)
        state = state_on_orbit(orbit, phase_grid(k));
        remaining_time = (1.0 - phase_grid(k)) * orbit.period;
        perturbed_state = state;
        perturbed_state(1) = perturbed_state(1) + cfg.stim_amp;
        t_cross = time_to_next_spike(cfg, perturbed_state);
        prc(k) = (remaining_time - t_cross) / orbit.period;
    end
end


function model = run_adaptive_fit(cfg, orbit, phase_grid, true_prc, M, rng_seed)
    rng(rng_seed);
    n_coef = 2 * M + 1;
    coefs = zeros(n_coef, 1);
    mse_history = zeros(cfg.n_adaptive_samples, 1);
    sampled_phase = zeros(cfg.n_adaptive_samples, 1);
    sampled_advance = zeros(cfg.n_adaptive_samples, 1);

    for k = 1:cfg.n_adaptive_samples
        phi = rand();
        state = state_on_orbit(orbit, phi);
        remaining_time = (1.0 - phi) * orbit.period;
        perturbed_state = state;
        perturbed_state(1) = perturbed_state(1) + cfg.stim_amp;
        t_cross = time_to_next_spike(cfg, perturbed_state);
        observed = (remaining_time - t_cross) / orbit.period;

        [prediction, basis] = fourier_prc(phi, coefs, M);
        err = prediction - observed;
        if cfg.learning_mode == "adaptive"
            alpha_base = k^(-0.68);
        else
            alpha_base = cfg.fixed_alpha;
        end

        for j = 1:n_coef
            harmonic_order = max(1, ceil((j - 1) / 2));
            alpha = alpha_base / harmonic_order;
            coefs(j) = coefs(j) - alpha * err * basis(j);
        end

        est = evaluate_fourier_series(phase_grid, coefs, M);
        mse_history(k) = mean((est - true_prc).^2);
        sampled_phase(k) = phi;
        sampled_advance(k) = observed;
    end

    model.order = M;
    model.coefs = coefs;
    model.phase_samples = sampled_phase;
    model.advance_samples = sampled_advance;
    model.estimate = evaluate_fourier_series(phase_grid, coefs, M);
    model.mse_history = mse_history;
    model.final_mse = mse_history(end);
end


function tf = can_use_parallel_pool()
    tf = false;
    if ~license('test', 'Distrib_Computing_Toolbox')
        return;
    end

    try
        pool = gcp('nocreate');
        if isempty(pool)
            parpool('threads');
        end
        tf = true;
    catch
        tf = false;
    end
end


function make_prc_fit_figure(cfg, phase_grid, true_prc, model_results)
    fig = figure('Color', 'w', 'Position', [100 100 1100 450]);
    subplot(1, 2, 1);
    hold on;
    plot(phase_grid, true_prc, 'k--', 'LineWidth', 2, 'DisplayName', 'Numerical PRC');
    colors = lines(numel(model_results));
    for idx = 1:numel(model_results)
        model = model_results{idx};
        plot(phase_grid, model.estimate, 'Color', colors(idx, :), ...
            'LineWidth', 2, 'DisplayName', sprintf('Adaptive fit, M=%d', model.order));
    end
    xlabel('Phase at stimulation (\phi)');
    ylabel('Phase advance (\Delta\phi)');
    title('Reduced-HH PRC and adaptive Fourier fits');
    legend('Location', 'best');
    grid on;
    box off;

    subplot(1, 2, 2);
    hold on;
    for idx = 1:numel(model_results)
        model = model_results{idx};
        semilogy(1:cfg.n_adaptive_samples, model.mse_history, ...
            'Color', colors(idx, :), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('M=%d', model.order));
    end
    xlabel('Stimulus number');
    ylabel('PRC estimation MSE');
    title('Adaptive convergence by Fourier order');
    legend('Location', 'best');
    grid on;
    box off;
    exportgraphics(fig, [cfg.output_prefix '_adaptive_prc_fit.png'], 'Resolution', 300);
end


function make_order_summary_figure(cfg, order_summary)
    fig = figure('Color', 'w', 'Position', [200 200 640 440]);
    orders = [order_summary.order];
    median_summary_mse = [order_summary.median_summary_mse];
    q25_summary_mse = [order_summary.q25_summary_mse];
    q75_summary_mse = [order_summary.q75_summary_mse];
    lower_err = median_summary_mse - q25_summary_mse;
    upper_err = q75_summary_mse - median_summary_mse;

    hold on;
    band_handle = errorbar(orders, median_summary_mse, lower_err, upper_err, 'LineStyle', 'none', ...
        'Color', [0.55 0.7 0.95], 'LineWidth', 1.8, 'CapSize', 10, ...
        'DisplayName', '25th-75th percentile');
    point_handle = semilogy(orders, median_summary_mse, 'o', 'LineWidth', 2.0, 'MarkerSize', 8, ...
        'Color', [0 0.35 0.75], 'MarkerFaceColor', [0.2 0.5 0.9], ...
        'DisplayName', 'Median MSE');
    set(gca, 'YScale', 'log');
    xlim([min(orders) - 0.4, max(orders) + 0.4]);
    xlabel('Fourier order (M)');
    ylabel(sprintf('PRC estimation MSE at sample %d', cfg.summary_sample));
    title(sprintf('Reduced-HH model-order adequacy (%d runs/order)', cfg.n_repeats_per_order));
    legend([band_handle, point_handle], {'25th-75th percentile', 'Median MSE'}, 'Location', 'best');
    grid on;
    box off;
    exportgraphics(fig, [cfg.output_prefix '_order_summary.png'], 'Resolution', 300);
end


function state = state_on_orbit(orbit, phi)
    phi = mod(phi, 1.0);
    V = interp1(orbit.phase, orbit.state(:, 1), phi, 'linear', 'extrap');
    n = interp1(orbit.phase, orbit.state(:, 2), phi, 'linear', 'extrap');
    state = [V; n];
end


function [value, basis] = fourier_prc(phi, coefs, M)
    basis = zeros(2 * M + 1, 1);
    basis(1) = 1.0;
    value = coefs(1);
    for k = 1:M
        angle = 2 * pi * k * phi;
        basis(2 * k) = cos(angle);
        basis(2 * k + 1) = sin(angle);
        value = value + coefs(2 * k) * basis(2 * k) + coefs(2 * k + 1) * basis(2 * k + 1);
    end
end


function values = evaluate_fourier_series(phi, coefs, M)
    values = zeros(size(phi));
    for idx = 1:numel(phi)
        values(idx) = fourier_prc(phi(idx), coefs, M);
    end
end


function t_cross = time_to_next_spike(cfg, x0)
    opts = odeset('Events', @(t, x) spike_event(t, x, cfg.threshold), ...
                  'RelTol', 1e-7, 'AbsTol', 1e-9);
    [~, ~, te] = ode45(@(t, x) reduced_hh_rhs(t, x, cfg, 0.0), [cfg.dt cfg.max_cycle_time], x0, opts);
    if isempty(te)
        error('No spike detected after perturbation. Increase max_cycle_time or adjust threshold.');
    end
    t_cross = te(1);
end


function dx = reduced_hh_rhs(~, x, cfg, u)
    V = x(1);
    n = x(2);
    m_inf = alpha_m(V) ./ (alpha_m(V) + beta_m(V));
    h = 0.8 - n;
    INa = cfg.gNa * (m_inf^3) * h * (V - cfg.VNa);
    IK = cfg.gK * (n^4) * (V - cfg.VK);
    IL = cfg.gL * (V - cfg.VL);
    dV = (cfg.Ib - INa - IK - IL + u) / cfg.C;
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n;
    dx = [dV; dn];
end


function val = alpha_m(V)
    val = 0.1 * safe_rate((25 - V) / 10, 25 - V);
end


function val = beta_m(V)
    val = 4.0 * exp(-V / 18);
end


function val = alpha_n(V)
    val = 0.01 * safe_rate((10 - V) / 10, 10 - V);
end


function val = beta_n(V)
    val = 0.125 * exp(-V / 80);
end


function val = safe_rate(z, numerator)
    if abs(z) < 1e-8
        val = 10.0;
    else
        val = numerator / (exp(z) - 1.0);
    end
end


function [value, isterminal, direction] = spike_event(~, x, threshold)
    value = x(1) - threshold;
    isterminal = 1;
    direction = 1;
end
