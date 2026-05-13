import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import argparse
import pickle
import os
from scipy.interpolate import interp1d

# --- Set plotting style for publication-quality figures ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2
})


# ==========================================
# Core Classes (Refactored from previous steps)
# ==========================================
def reconstruct_prc(phi, coefs):
    """
    Standalone function to calculate the PRC value from Fourier coefficients.
    Works for any model order M = (len(coefs) - 1) / 2.
    """
    M = (len(coefs) - 1) // 2
    # Start with the DC offset (a0)
    d_phi = coefs[0]

    for k in range(1, M + 1):
        angle = 2 * np.pi * k * phi
        # a_k * cos + b_k * sin
        d_phi += coefs[2 * k - 1] * np.cos(angle) + coefs[2 * k] * np.sin(angle)
    return d_phi

class PhaseAnalyzer:
    """Handles weighted Fourier analysis for phase estimation."""

    @staticmethod
    def estimate_phase(t, data, weights, omega):
        theta = 2.0 * np.pi * omega * t
        a_sin = np.sum(weights * data * np.sin(theta))
        a_cos = np.sum(weights * data * np.cos(theta))
        est_phase_rad = -np.arctan2(-a_sin, a_cos)
        return (est_phase_rad / (2.0 * np.pi)) % 1.0


class AdaptivePRCEstimator:
    """Manages a Fourier-series based PRC model and its online adaptation."""

    def __init__(self, M=3, alpha_base=0.02):
        self.M = M
        self.alpha_base = alpha_base
        self.num_coefs = 2 * M + 1
        self.coefs = np.zeros(self.num_coefs)
        # History storage for animation
        self.coef_history = [self.coefs.copy()]
        self.NSamp = 0

    def predict(self, phi, specific_coefs=None):
        """Predicts PRC value using current or specified coefficients."""
        current_coefs = specific_coefs if specific_coefs is not None else self.coefs
        d_phi = current_coefs[0]
        grad = np.zeros(self.num_coefs)
        grad[0] = 1.0

        for k in range(1, self.M + 1):
            angle = 2 * np.pi * k * phi
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            d_phi += current_coefs[2 * k - 1] * cos_val + current_coefs[2 * k] * sin_val
            grad[2 * k - 1] = cos_val
            grad[2 * k] = sin_val
        return d_phi, grad

    def update(self, phi_at_stim, observed_advance):
        """Performs LMS update with spectral regularization."""
        pred_advance, grad = self.predict(phi_at_stim)
        error = pred_advance - observed_advance
        self.NSamp += 1
        for j in range(self.num_coefs):
            # Spectral Regularization: alpha_j = alpha_base / harmonic_order
            # harmonic_order for a0 is 1 (effectively), then 1, 2, 2, 3, 3...
            harmonic_order = np.ceil(j / 2) if j > 0 else 1.0
            if self.alpha_base > 0:
                effective_alpha = self.alpha_base / harmonic_order
            else:
                effective_alpha = self.NSamp ** -0.68/harmonic_order

            self.coefs[j] -= effective_alpha * error * grad[j]

        self.coef_history.append(self.coefs.copy())
        return error

    def AdaptiveUpdate(self, phi_at_stim, observed_advance):
        """Performs LMS update with spectral regularization."""
        pred_advance, grad = self.predict(phi_at_stim)
        error = pred_advance - observed_advance
        self.NSamp += 1

        for j in range(self.num_coefs):
            # Spectral Regularization: alpha_j = alpha_base / harmonic_order
            # harmonic_order for a0 is 1 (effectively), then 1, 2, 2, 3, 3...
            harmonic_order = np.ceil(j / 2) if j > 0 else 1.0
            alpha_base = 1.0/self.NSamp**0.68
            effective_alpha = alpha_base / harmonic_order
            self.coefs[j] -= effective_alpha * error * grad[j]

        self.coef_history.append(self.coefs.copy())
        return error


class ThetaNeuron:
    """Simulates a noisy phase oscillator."""

    def __init__(self, omega=1.0, dt=0.01, noise_amp=0.01):
        self.omega = omega
        self.dt = dt
        self.noise_amp = noise_amp

    def ground_truth_prc(self, phi, A=0.3):
        # Using A=0.5 to keep phase advances reasonable relative to Omega=1
        return A * phi**2 * (1.0 - phi)

    def simulate_segment(self, t_start, t_end, start_phase, apply_stim=False):
        n_steps = int((t_end - t_start) / self.dt)
        time = np.linspace(t_start, t_end, n_steps, endpoint=False)
        phase = np.zeros(n_steps)
        phase[0] = start_phase

        stim_applied_at = -1

        for i in range(n_steps - 1):
            d_phi_det = self.omega * self.dt
            d_phi_noise = self.noise_amp * np.sqrt(self.dt) * np.random.randn()
            d_phi_stim = 0

            # Apply stimulus at the onset of segment.
            if apply_stim and i == 0:
                d_phi_stim =  self.ground_truth_prc(-phase[i] % 1.0)
                stim_applied_at = time[i]

            phase[i + 1] = phase[i] + d_phi_det + d_phi_stim + d_phi_noise

        amp = np.cos(2.0 * np.pi * phase)
        return time, amp, phase, stim_applied_at


# ==========================================
# Helper Functions for Simulation & Data Gen
# ==========================================

def run_long_simulation(n_stim=3000, alpha=0.05):
    """Runs a single long simulation for Figs 1, 2 and Video."""
    print(f"Running long simulation ({n_stim} pulses)...")
    neuron = ThetaNeuron(noise_amp=0.01)
    estimator = AdaptivePRCEstimator(M=3, alpha_base=alpha)
    analyzer = PhaseAnalyzer()
    lag = 0

    # Window parameters
    T_period = 1.0 / neuron.omega
    win_len_samples = int(np.ceil(3.0 * T_period / neuron.dt))
    t_win = np.arange(0, win_len_samples * neuron.dt, neuron.dt)
    weights = np.ones(win_len_samples) #np.exp(-np.linspace(0, 3.0, win_len_samples))

    # Data storage
    full_time, full_amp, full_phase = [], [], []
    stim_times = []
    measured_phi, measured_adv = [], []

    curr_time = 0
    curr_phase = 0

    for i in range(n_stim):
        # 1. Simulate pre-stimulus baseline + window
        r = np.random.rand() #create some jitter in the stimulation time
        t_seg, amp_seg, phi_seg, _ = neuron.simulate_segment(curr_time, curr_time + (4+r) * T_period, curr_phase,
                                                             apply_stim=False)
        full_time.extend(t_seg)
        full_amp.extend(amp_seg)
        full_phase.extend(phi_seg)
        curr_time = t_seg[-1] + neuron.dt
        curr_phase = phi_seg[-1]

        # Estimate Phase Before
        data_pre = np.array(full_amp[-win_len_samples:])
        phi_pre = analyzer.estimate_phase(-t_win[::-1], data_pre, weights[::-1], neuron.omega)

        # 2. Simulate stimulus event
        # Simulate long enough for window after
        t_seg_s, amp_seg_s, phi_seg_s, stim_t = neuron.simulate_segment(curr_time, curr_time + 4 * T_period, curr_phase,
                                                                        apply_stim=True)
        full_time.extend(t_seg_s)
        full_amp.extend(amp_seg_s)
        full_phase.extend(phi_seg_s)
        stim_times.append(stim_t)

        # Estimate Phase After
        data_post = np.array(amp_seg_s[lag:lag+win_len_samples])

        phi_post = analyzer.estimate_phase(t_win+lag*neuron.dt, data_post, weights, neuron.omega)

        #plt.ion()
        #plt.plot(-t_win[::-1], data_pre*weights[::-1],'.')
        #plt.plot(t_win+lag*neuron.dt, data_post*weights,'.')
        #plt.show()

        # Calculate & Wrap Observed Advance based on standard definition: Phi_new - Phi_old_expected
        # A positive PRC means the phase advanced (period shortened).
        # The phase measured *after* should be *larger* than the phase measured *before* relative to natural progression.
        # Wait, the standard definition of PRC is often (NewPhase - OldPhase).
        # If phi_pre is 0.9, and phi_post is 0.1 (it advanced past 0),
        # observed_adv = 0.1 - 0.9 = -0.8. Wrapped to [-0.5, 0.5] gives 0.2. Correct.
        observed_adv = (phi_post - phi_pre)
        # Wrap to [-0.5, 0.5] for correct representation of advances/delays
        observed_adv = (observed_adv + 0.5) % 1.0 - 0.5

        # Update Estimator
        estimator.update(phi_pre, observed_adv)
        measured_phi.append(phi_pre)
        measured_adv.append(observed_adv)
        curr_time = full_time[-1] + neuron.dt
        curr_phase = full_phase[-1]

    inst_freq = np.diff(full_phase) / neuron.dt
    inst_freq = np.insert(inst_freq, 0, neuron.omega)

    return {
        'time': np.array(full_time), 'amp': np.array(full_amp),
        'phase': np.array(full_phase), 'freq': inst_freq,
        'stim_times': np.array(stim_times),
        'measured_phi': np.array(measured_phi), 'measured_adv': np.array(measured_adv),
        'coef_history': estimator.coef_history, 'final_coefs': estimator.coefs,
        'true_prc_func': neuron.ground_truth_prc, 'estimator_class': estimator
    }


def run_parameter_sweep(NStimSteps=5, NAlphaSteps = 40):
    """Runs the nested loops for Figure 3 (takes time!)."""
    print("Running parameter sweep for Figure 3 (this may take a minute)...")
    N_stims = np.round(np.logspace(2, 3.47, NStimSteps)).astype(int)  # e.g., 32 to 1000
    Alphas = np.logspace(-3, -.5, NAlphaSteps)  # e.g., 0.001 to 0.3
    N_rep = 10

    neuron = ThetaNeuron()
    phi_grid = np.linspace(0, 1, 200)
    true_prc = neuron.ground_truth_prc(phi_grid)

    error_matrix = np.zeros((len(N_stims), len(Alphas), N_rep))

    for r in range(N_rep):
        print(f"  Sweep Repetition {r + 1}/{N_rep}")
        # Run one very long simulation to gather enough measured points
        long_run = run_long_simulation(n_stim=N_stims[-1], alpha=0.05)  # Alpha doesn't matter here, just need raw data
        m_phi = long_run['measured_phi']
        m_adv = long_run['measured_adv']

        for i, n_stim in enumerate(N_stims):
            # Use subset of data
            subset_phi = m_phi[:n_stim]
            subset_adv = m_adv[:n_stim]

            for j, alpha in enumerate(Alphas):
                # Retrain a fresh estimator on this subset with specific alpha
                sweeper_est = AdaptivePRCEstimator(M=3, alpha_base=alpha)
                for k in range(n_stim):
                    sweeper_est.update(subset_phi[k], subset_adv[k])

                # Calculate MSE against ground truth
                est_prc = np.array([sweeper_est.predict(p)[0] for p in phi_grid])
                mse = np.mean((true_prc + est_prc) ** 2)
                #plt.ion()
                #plt.plot(true_prc)
                #plt.plot(est_prc)
                #plt.title('MSE: {:.3g}, Alpha: {:.3g}, NStim: {}'.format(mse, alpha, n_stim))
                #plt.show()
                error_matrix[i, j, r] = mse

    return {
        'N_stims': N_stims, 'Alphas': Alphas,
        'mean_error': np.mean(error_matrix, axis=2),
        'std_error': np.std(error_matrix, axis=2)
    }


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to '{filename}'.")


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded data from '{filename}'.")
    return data


def get_simulation_data(source, cache_file, n_stim, alpha, save_cache=False):
    if source == 'load':
        return load_pickle(cache_file)

    sim_data = run_long_simulation(n_stim=n_stim, alpha=alpha)
    if save_cache:
        save_pickle(sim_data, cache_file)
    return sim_data


def get_sweep_data(source, cache_file, n_stim_steps, n_alpha_steps, save_cache=False):
    if source == 'load':
        return load_pickle(cache_file)

    sweep_data = run_parameter_sweep(n_stim_steps, n_alpha_steps)
    if save_cache:
        save_pickle(sweep_data, cache_file)
    return sweep_data


# ==========================================
# Requested Plotting & Video Functions
# ==========================================

def generate_Fig1(sim_data):
    """Generates static Figures 1, 2, and 3 using standalone reconstruction."""
    print("Generating Figure 1")
    phi_grid = np.linspace(0, 1, 200)

    # --- Figure 1: Raw Data Traces (same as before) ---
    fig1 = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, figure=fig1, height_ratios=[1, 1, 1], hspace=0.1)
    ax1, ax2, ax3 = fig1.add_subplot(gs[0]), fig1.add_subplot(gs[1]), fig1.add_subplot(gs[2])

    t_idx = (sim_data['time'] > 10) & (sim_data['time'] < 30)
    time_window = sim_data['time'][t_idx]
    amp_window = sim_data['amp'][t_idx]
    phase_window = sim_data['phase'][t_idx] % 1.0
    freq_window = sim_data['freq'][t_idx]

    stim_mask = (sim_data['stim_times'] > time_window[0]) & (sim_data['stim_times'] < time_window[-1])
    stim_times_window = sim_data['stim_times'][stim_mask]

    ax1.plot(time_window, amp_window, 'k')
    ax2.plot(time_window, phase_window, 'b')
    ax3.plot(time_window, freq_window, 'g')

    if stim_times_window.size > 0:
        stim_amp = np.interp(stim_times_window, time_window, amp_window)
        stim_phase = np.interp(stim_times_window, time_window, phase_window)
        stim_freq = np.interp(stim_times_window, time_window, freq_window)
        marker_style = dict(color='red', s=30, zorder=5)
        ax1.scatter(stim_times_window, stim_amp, **marker_style)
        ax2.scatter(stim_times_window, stim_phase, **marker_style)
        ax3.scatter(stim_times_window, stim_freq, **marker_style)
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Phase (cycles)')
    ax3.set_ylabel(r'Instantaneous Frequency ($\omega$)')
    ax3.set_xlabel('Time (cycles)')
    ax1.tick_params(axis='x', which='both', labelbottom=False)
    ax2.tick_params(axis='x', which='both', labelbottom=False)
    fig1.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.11)
    fig1.show()


def generate_Fig2(sim_data):

    print("Generating Figure 2")
    phi_grid = np.linspace(0, 1, 200)

    # --- Figure 2: Final PRC Estimate (FIXED) ---
    fig2, ax_prc = plt.subplots(figsize=(8, 6))
    true_prc = sim_data['true_prc_func'](phi_grid)

    # 1. Ground Truth
    ax_prc.plot(phi_grid, true_prc, 'k--', linewidth=2, label='Ground Truth PRC')

    # 2. Measured Data Points
    ax_prc.scatter(sim_data['measured_phi'], -sim_data['measured_adv'],
                   color='gray', alpha=0.3, s=10, label='Measured Phase Shifts (Noisy)')

    # 3. Final Estimated Fourier Fit (STANDALONE RECONSTRUCTION)
    final_coefs = sim_data['final_coefs']
    est_prc = np.array([reconstruct_prc(p, final_coefs) for p in phi_grid])
    ax_prc.plot(phi_grid, -est_prc, 'r-', linewidth=3, label='Final Adaptive Estimate (M=3)')

    ax_prc.axhline(0, color='k', linestyle=':', linewidth=1)
    ax_prc.set_xlabel(r'Phase at Stimulation ($\phi$)')
    ax_prc.set_ylabel(r'Phase Advance ($\Delta\phi$)')
    ax_prc.set_title('Figure 2: Final PRC Estimate vs Ground Truth')
    ax_prc.legend()
    ax_prc.grid(True, alpha=0.3)
    fig2.show()


def generate_Fig3(sweep_data):
    print("Generating Figure 3")
    fig3, ax_sweep = plt.subplots(figsize=(9, 7))

    # 1. Setup Color Normalization
    norm = mcolors.LogNorm(vmin=np.min(sweep_data['mean_error']),
                           vmax=np.max(sweep_data['mean_error']))

    im = ax_sweep.imshow(sweep_data['mean_error'], aspect='auto', origin='lower',
                         norm=norm, cmap='viridis')

    # 2. Map Alpha Values to Heatmap Indices
    # We create a function: log10(alpha) -> pixel_index
    log_alphas = np.log10(sweep_data['Alphas'])
    pixel_indices = np.arange(len(sweep_data['Alphas']))
    # interp1d allows us to find the exact (even fractional) index for any alpha
    alpha_to_idx = interp1d(log_alphas, pixel_indices, fill_value="extrapolate")

    # 3. Define Major Ticks (1e-3, 1e-2, 1e-1)
    major_targets = np.array([1e-3, 1e-2, 1e-1])
    # Filter targets to stay within the range of your actual sweep
    major_targets = major_targets[(major_targets >= np.min(sweep_data['Alphas'])) &
                                  (major_targets <= np.max(sweep_data['Alphas']))]

    major_indices = alpha_to_idx(np.log10(major_targets))
    ax_sweep.set_xticks(major_indices)
    ax_sweep.set_xticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in major_targets])

    # 4. Add Logarithmic Minor Ticks (the "rungs" between powers of 10)
    minor_ticks = []
    for p in range(-4, 0):  # Range of powers
        for m in range(2, 10):  # Multipliers 2-9
            val = m * 10 ** p
            if np.min(sweep_data['Alphas']) <= val <= np.max(sweep_data['Alphas']):
                minor_ticks.append(alpha_to_idx(np.log10(val)))

    ax_sweep.set_xticks(minor_ticks, minor=True)
    # Make minor ticks visible
    ax_sweep.tick_params(axis='x', which='minor', length=4, color='white')

    # 5. Handle Y-axis (N_stims)
    ax_sweep.set_yticks(np.arange(len(sweep_data['N_stims'])))
    ax_sweep.set_yticklabels([f"{n}" for n in sweep_data['N_stims']])

    pixel_indices_y = np.arange(len(sweep_data['N_stims']))
    y_values = sweep_data['N_stims']
    nstim_to_idx = interp1d(y_values, pixel_indices_y, fill_value="extrapolate")
    #NS = np.array([100.0, 1000.0, 3000.0])
    #M = np.array([0.05, 0.008, 0.004])
    #dot_x = alpha_to_idx(np.log10(M))
    #dot_y = nstim_to_idx(NS)
    #ax_sweep.plot(dot_x, dot_y, 'ro')
    NS_full = np.logspace(2, 3.47, 100)
    a = 1.0/NS_full**.68
    dot_x = alpha_to_idx(np.log10(a))
    dot_y = nstim_to_idx(NS_full)
    ax_sweep.plot(dot_x, dot_y, 'k')

    # Labels and Colorbar
    ax_sweep.set_xlabel(r'Learning Rate Base ($\alpha_{base}$)')
    ax_sweep.set_ylabel(r'Number of Stimuli ($N_{stim}$)')
    ax_sweep.set_title('Convergence Error (MSE) Trade-off')

    cbar = fig3.colorbar(im, ax=ax_sweep)
    cbar.set_label('Mean Squared Error (log scale)')

    plt.tight_layout()
    plt.show()

def generate_Fig4(sweep_data):
    print("Generating Figure 4")

    # --- Figure 4: Parameter Sweep (LINE PLOT VERSION with SHADED ERRORS) ---
    fig4, ax_lines = plt.subplots(figsize=(8, 6))

    # Extract arrays once for clarity
    means = sweep_data['mean_error']
    # If std_error doesn't exist, create a 2D zero array of the same shape as means
    stds = sweep_data.get('std_error', np.zeros_like(means))

    for i, n_stim in enumerate(sweep_data['N_stims']):
        mean_err_row = means[i, :]
        std_err_row = stds[i, :]

        line, = ax_lines.loglog(sweep_data['Alphas'], mean_err_row, '-o', label=f'N={n_stim}')

        # Shaded area: mean +/- std
        ax_lines.fill_between(sweep_data['Alphas'],
                              mean_err_row - std_err_row,
                              mean_err_row + std_err_row,
                              alpha=0.2, color=line.get_color())

    ax_lines.set_xlabel(r'Learning Rate ($\alpha$)')
    ax_lines.set_ylabel('Mean Squared Error')
    ax_lines.set_title('Convergence Error vs Learning Rate')
    ax_lines.legend(title="Stimuli Count")
    ax_lines.grid(True, which="both", ls="-", alpha=0.2)
    fig4.show()

def create_prc_adaptation_video(sim_data, filename='prc_adaptation_demo.mp4'):
    """
    Creates an MP4 video of the PRC estimate evolving over time.
    Uses standalone reconstruct_prc to avoid object-call errors.
    """
    print(f"Generating video '{filename}'... (This takes a moment)")

    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    phi_grid = np.linspace(0, 1, 200)

    # Static Elements: Ground Truth PRC
    true_prc = sim_data['true_prc_func'](phi_grid)
    ax.plot(phi_grid, true_prc, 'k--', linewidth=2, label='Ground Truth PRC')
    ax.axhline(0, color='k', linestyle=':', linewidth=1)

    # 2. Set dynamic limits based on data
    ax.set_xlim(0, 1)
    y_min = -.1 #np.min(sim_data['measured_adv']) - 0.4
    y_max = .1 # np.max(sim_data['measured_adv']) + 0.4
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(r'Phase at Stimulation ($\Phi$)')
    ax.set_ylabel(r'Phase Advance ($\Delta\Phi$)')
    ax.grid(True, alpha=0.3)

    # 3. Dynamic Elements (Initialized empty)
    scatter_pts = ax.scatter([], [], color='gray', alpha=0.5, s=15, label='Measured Data')
    line_est, = ax.plot([], [], 'r-', linewidth=3, label='Adaptive Estimate')
    title_text = ax.set_title('')
    ax.legend(loc='upper right')

    n_frames = len(sim_data['measured_phi'])

    def animate(i):
        # Update Scatter plot with all points up to stimulus i
        current_phi = sim_data['measured_phi'][:i + 1]
        current_adv = -sim_data['measured_adv'][:i + 1]
        scatter_pts.set_offsets(np.c_[current_phi, current_adv])

        # Update Estimate Line using historical coefficients stored at index i
        # We use the standalone function here
        historical_coefs = sim_data['coef_history'][i]
        est_vals = [-reconstruct_prc(p, historical_coefs) for p in phi_grid]
        line_est.set_data(phi_grid, est_vals)

        # Update Title
        title_text.set_text(f'Real-time PRC Adaptation: Stimulus {i + 1}/{n_frames}')
        return scatter_pts, line_est, title_text

    # 4. Create Animation
    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)

    # 5. Save using FFMpeg
    try:
        writervideo = animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264'])
        ani.save(filename, writer=writervideo)
        print(f"✅ Video saved successfully to {filename}")
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        print("Note: Ensure FFMPEG is installed and in your system PATH.")
    finally:
        plt.close(fig)


def generate_error_over_time_figure(sim_data, sim_data2):
    """
    Plots the Mean Squared Error (MSE) of the PRC estimate
    as a function of time (or stimulus count).
    """
    print("Calculating error convergence history...")

    # 1. Setup Phase Grid for MSE calculation
    phi_grid = np.linspace(0, 1, 200)
    true_prc = sim_data['true_prc_func'](phi_grid)

    coef_history = sim_data['coef_history']  # Shape: (n_stim, 2M+1)
    stim_times = sim_data['stim_times']

    mse_history = []

    # 2. Calculate MSE for every step of the simulation
    for i in range(len(coef_history)-1):
        current_coefs = coef_history[i]
        # Reconstruct the estimate at this specific moment in time
        est_vals = np.array([reconstruct_prc(p, current_coefs) for p in phi_grid])

        # Calculate MSE between truth and estimate
        mse = np.mean((true_prc + est_vals) ** 2)
        mse_history.append(mse)

    # 3. Create the Figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use semilogy because error often drops by several orders of magnitude
    S = np.linspace(0, len(mse_history)-1, len(mse_history))
    ax.semilogy(S, mse_history, color='b', lw=1.5, label='Fixed Alpha')


    coef_history = sim_data2['coef_history']  # Shape: (n_stim, 2M+1)
    stim_times = sim_data2['stim_times']

    mse_history = []

    # 2. Calculate MSE for every step of the simulation
    for i in range(len(coef_history)-1):
        current_coefs = coef_history[i]
        # Reconstruct the estimate at this specific moment in time
        est_vals = np.array([reconstruct_prc(p, current_coefs) for p in phi_grid])

        # Calculate MSE between truth and estimate
        mse = np.mean((true_prc + est_vals) ** 2)
        mse_history.append(mse)


    # Use semilogy because error often drops by several orders of magnitude
    ax.semilogy(S, mse_history, color='r', lw=1.5, label='Adaptive Alpha')

    # Optional: Add a moving average to smooth out the noise from stochastic updates
    #window_size = 50
    #if len(mse_history) > window_size:
    #    smoothed = np.convolve(mse_history, np.ones(window_size) / window_size, mode='valid')
        #ax.semilogy(stim_times[window_size - 1:], smoothed, color='red', lw=2, label='Trend (Moving Avg)')

    # 4. Formatting for Publication
    ax.set_xlabel('Stimulus Number', fontsize=12)
    ax.set_ylabel('Estimation MSE (log scale)', fontsize=12)
    ax.set_title('PRC Learning Convergence: Fixed vs Adaptive Learning Rate', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    plt.tight_layout()
    plt.show()
    return fig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or load Adaptive PRC simulations and generate figures."
    )
    parser.add_argument('--fixed-sim-source', choices=['run', 'load'], default='run',
                        help="Whether to run or load the fixed-alpha simulation.")
    parser.add_argument('--adaptive-sim-source', choices=['run', 'load'], default='run',
                        help="Whether to run or load the adaptive-alpha simulation.")
    parser.add_argument('--sweep-source', choices=['run', 'load'], default='run',
                        help="Whether to run or load the parameter sweep.")
    parser.add_argument('--fixed-sim-cache', default='long_sim_fixed_alpha.pkl',
                        help="Pickle file for the fixed-alpha simulation.")
    parser.add_argument('--adaptive-sim-cache', default='long_sim_adaptive_alpha.pkl',
                        help="Pickle file for the adaptive-alpha simulation.")
    parser.add_argument('--sweep-cache', default='sweep_data.pkl',
                        help="Pickle file for the parameter sweep.")
    parser.add_argument('--save-sim-cache', action='store_true',
                        help="Save simulation results to pickle after running them.")
    parser.add_argument('--save-sweep-cache', action='store_true',
                        help="Save sweep results to pickle after running them.")
    parser.add_argument('--n-stim', type=int, default=3000,
                        help="Number of stimuli for long simulations.")
    parser.add_argument('--fixed-alpha', type=float, default=0.002,
                        help="Fixed learning rate value.")
    parser.add_argument('--adaptive-alpha', type=float, default=-1.0,
                        help="Adaptive-alpha flag/value used by the script.")
    parser.add_argument('--sweep-nstim-steps', type=int, default=5,
                        help="Number of stimulus-count values in the sweep.")
    parser.add_argument('--sweep-nalpha-steps', type=int, default=40,
                        help="Number of learning-rate values in the sweep.")
    parser.add_argument('--skip-fig1', action='store_true', help="Skip Figure 1.")
    parser.add_argument('--skip-fig2', action='store_true', help="Skip Figure 2.")
    parser.add_argument('--skip-convergence', action='store_true',
                        help="Skip the convergence-over-time figure.")
    parser.add_argument('--skip-heatmap', action='store_true', help="Skip the heatmap figure.")
    parser.add_argument('--skip-lines', action='store_true', help="Skip the line-sweep figure.")
    parser.add_argument('--make-video', action='store_true', help="Generate the adaptation video.")
    return parser.parse_args()


def main():
    args = parse_args()

    need_fixed = not args.skip_convergence
    need_adaptive = not args.skip_fig1 or not args.skip_fig2 or not args.skip_convergence or args.make_video
    need_sweep = not args.skip_heatmap or not args.skip_lines

    fixed_sim_data = None
    adaptive_sim_data = None
    sweep_data = None

    if need_fixed:
        fixed_sim_data = get_simulation_data(
            source=args.fixed_sim_source,
            cache_file=args.fixed_sim_cache,
            n_stim=args.n_stim,
            alpha=args.fixed_alpha,
            save_cache=args.save_sim_cache
        )

    if need_adaptive:
        adaptive_sim_data = get_simulation_data(
            source=args.adaptive_sim_source,
            cache_file=args.adaptive_sim_cache,
            n_stim=args.n_stim,
            alpha=args.adaptive_alpha,
            save_cache=args.save_sim_cache
        )

    if need_sweep:
        sweep_data = get_sweep_data(
            source=args.sweep_source,
            cache_file=args.sweep_cache,
            n_stim_steps=args.sweep_nstim_steps,
            n_alpha_steps=args.sweep_nalpha_steps,
            save_cache=args.save_sweep_cache
        )

    if not args.skip_fig1 and adaptive_sim_data is not None:
        generate_Fig1(adaptive_sim_data)

    if not args.skip_fig2 and adaptive_sim_data is not None:
        generate_Fig2(adaptive_sim_data)

    if not args.skip_convergence and fixed_sim_data is not None and adaptive_sim_data is not None:
        generate_error_over_time_figure(fixed_sim_data, adaptive_sim_data)

    if not args.skip_heatmap and sweep_data is not None:
        generate_Fig3(sweep_data)

    if not args.skip_lines and sweep_data is not None:
        generate_Fig4(sweep_data)

    if args.make_video and adaptive_sim_data is not None:
        create_prc_adaptation_video(
            adaptive_sim_data,
            filename='prc_AdaptivePRC_AdpativeLearning.mp4'
        )

    plt.show(block=True)


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    main()


