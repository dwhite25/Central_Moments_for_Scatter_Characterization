'''
Configuration and generation tools for simulated scatterer databases.

DB_Config defines the interrogating waveform, scatterer type, noise level, analysis features, 
and spectral-estimation settings used to construct the simulation databases.

Database generation is performed in chunks so that long Google Colab runs can be resumed after interruption. 
Each output row combines the source scatterer parameters with the requested time-domain, frequency-domain, 
and statistical features produced by spectral_estimation.py.
'''

import time
import numpy as np
import pandas as pd
import base_waves as bwl
import spectral_estimation as pfl
from pathlib import Path

class DB_Config:
    '''
    configure and generate simulated return-signal databases
    - DB_Config object specifies:
        - the interrogating waveform type and bandwidth
        - the scatter-distribution model
        - the sampled time/frequency grid
        - the requested analysis features
        - the noise level
        - the polynomial/spectral estimator settings
    - typical workflow:
        1. make_interrogating_wave()
        2. make_feature_columns()
        3. add_noise_vals()
        4. set_testing_vars()
        5. optionally configure cumulants, diagnostics, or weighting
        6. test_one_chunk()
        7. make_database()
    '''
    def __init__(self, base_type='discrete', scatterer='delta', SampleRate=100,
                 power=4, segments=5, real=False, bw=1, f_low=0):
        self.base_type      = base_type     # 'discrete', 'continuous', 'super'
        self.scatterer      = scatterer     # 'delta', 'gmm'
        self.SampleRate     = SampleRate
        self.power          = power
        self.segments       = segments
        self.real           = real
        self.bw             = bw
        self.f_low          = f_low
        self.f_high         = self.f_low + bw
        self.cumulants      = False
        self.diagnostics    = False
        self.use_weights    = False
        self.alpha          = 0.6
        self.q              = 4

    def make_interrogating_wave(self, SignalLen=20, num_freqs=6, freqs=[0], coefs=[1]):
        '''
        construct the interrogating waveform and its sampled spectral grid
            - base_type='continuous': uses all FFT-compatible frequencies spanning the requested bandwidth
            - base_type='discrete': uses `num_freqs` equally spaced components across the bandwidth
            - base_type='super': uses explicitly supplied frequencies and spectral coefficients
        - frequencies supplied are in cycles per unit time; BaseFunction.Spectral receives the corresponding 
          angular frequencies
        '''
        self.SignalLen      = SignalLen
        if self.base_type   == 'continuous':
            self.num_freqs  = int(round(self.bw * self.SignalLen)) + 1
            self.freqs      = np.linspace(self.f_low, self.f_high, self.num_freqs)
            self.G_coefs    = np.ones(self.num_freqs)
        elif self.base_type == 'discrete':
            self.num_freqs  = num_freqs
            self.freqs      = np.linspace(self.f_low, self.f_high, self.num_freqs)
            self.G_coefs    = np.ones(self.num_freqs)
        elif self.base_type == 'super':
            self.num_freqs  = len(freqs)
            self.G_coefs    = np.asarray(coefs, dtype=complex)
            self.freqs      = np.asarray(freqs, dtype=float)
        else:
            print('invalid base type')
            return
        # time samples are centered about zero so that the scatter distributions
        # and interrogating waveform share the same sampled coordinate system
        self.t              = np.round(np.linspace(-SignalLen/2, SignalLen/2, SignalLen*self.SampleRate,
                                                   endpoint=False), 3)
        self.sr             = bwl.Spectral(freqs=self.freqs*2*np.pi, t=self.t, coefs=self.G_coefs)
        self.samples        = len(self.t)
        self.omegas         = 2 * np.pi * np.asarray(self.freqs)
        self.G_fft_coefs, self.G_fft_freqs, self.G_fft_bins = pfl.extract_return_coefs(self.sr.signal, self.t,
                                                                                       self.freqs, real=self.real)

    def edit_scatterer_type(self, scatterer):
        if scatterer in ['delta', 'gmm']:
            self.scatterer  = scatterer
        else:
            print('invalid scatterer type')
            return

    def edit_base_type(self, base_type):
        if base_type in ['discrete', 'continuous', 'super']:
            self.base_type  = base_type
        else:
            print('invalid base type')
            return

    def set_added_columns(self, corrs_truth=True):
        '''
        select which feature families are generated for each interrogating-wave type
            - discrete spectral waves track the relevant discrete frequencies plus spectral moment features
            - continuous spectral waves retain spectral-moment features 
                - all frequencies within signal bandlimits are relevant and do not need to be uniquely tracked
            - superoscillatory waves use local time-domain polynomial features
        '''
        if self.base_type == 'discrete':
            self.add_polys      = False
            self.add_freqs      = True
            self.add_spectrals  = True
            self.add_corrs      = corrs_truth
        elif self.base_type == 'continuous':
            self.add_polys      = False
            self.add_freqs      = False
            self.add_spectrals  = True
            self.add_corrs      = corrs_truth
        elif self.base_type == 'super':
            self.add_polys      = True
            self.add_freqs      = False
            self.add_spectrals  = False
            self.add_corrs      = corrs_truth
        else:
            print('invalid base type')
            return
        return

    def make_feature_columns(self, corrs=True):
        '''
        construct the output-column names for the selected feature configuration
            - column order must exactly match the order in which 'add_metrics()' appends features to each row
        '''
        cols    = []
        self.set_added_columns(corrs_truth=corrs)
        # polynomial fits of segments of the time domain
        if self.add_polys:
            for seg in range(self.segments):
                for pwr in range(self.power + 1):
                    cols.append(f"segment{seg}_{pwr}_real")
            for seg in range(self.segments):
                for pwr in range(self.power + 1):
                    cols.append(f"segment{seg}_{pwr}_imag")
        # real and imaginary portions of the relevant frequencies in the return signal
        if self.add_freqs:
            for freq in range(self.num_freqs):
                cols.append(f"freq{freq}_real")
            for freq in range(self.num_freqs):
                cols.append(f"freq{freq}_imag")
        # moments as predicted by polynomial fit of spectral domain
        if self.add_spectrals:
            cols += ["spectral_mean_pred", "spectral_var_pred", "spectral_std_pred",
                    "spectral_skew_pred", "spectral_kurt_pred"]
            if self.diagnostics:
                cols += ["min_freq_coef", "max_phase_jump", "fit_valid", "fit_valid_code"]
        # add return signal correlations with interrogating wave
        if self.add_corrs:
            cols += ["0_lag_corr_base", "best_corr_base", "best_lag_base"]
            # add gmm return signal correlations with delta return signal correlations
            if self.scatterer == 'gmm':
                cols += ["0_lag_corr_delta", "best_corr_delta", "best_lag_delta"]
        self.cols = cols

    def add_noise_vals(self, snr):
        ''' configure noiseless ('NA') or noisy simulation at the supplied SNR '''
        self.add_noise  = False if (snr == 'NA') else True
        self.snr        = None if (snr == 'NA') else float(snr)
        self.snr_label  = snr

    def set_testing_vars(self, max_fit_order=4, norm='intercept'):
        ''' set the spectral Taylor-fit order and normalization method '''
        self.max_fit_order  = max_fit_order
        self.norm           = norm

    def set_cumulants(self, cumulants):
        self.cumulants  = cumulants

    def set_diagnostics(self, diagnostics):
        self.diagnostics    = diagnostics

    def set_weights(self, use_weights=False, alpha=0.6, q=4):
        ''' configure optional low-frequency weighting of the spectral fit '''
        self.use_weights    = use_weights
        self.alpha          = alpha
        self.q              = q

    def make_database(self, root, chunk_size=1000):
        '''
        generate the configured simulation database and save it incrementally
            - each scatterer is converted into a simulated return waveform and analyzed using 
              the feature configuration defined by this DB_Config object
            - results are written in chunks to reduce memory usage and to support 
              long-running Colab calculations
                - if the output CSV already exists, processing resumes from the number of rows already present
                - resuming assumes that the existing output file was generated with the same DB_Config settings
        '''
        root            = Path(root)
        input_file      = root / f'{self.scatterer}_database.csv'
        db              = pd.read_csv(input_file)
        # get initial header row column names
        if self.scatterer == 'delta':
            moment_cols = db.columns.tolist()
        elif self.scatterer == 'gmm':
            moment_cols = ['mu1', 'mu2', 'sigma', 'w1', 'mean', 'variance', 'std', 'skew',
                           'kurtosis', 'inv_kurt', 'hyperskewness', 'hyperkurtosis']
        all_cols        = moment_cols + self.cols
        # create file name and make sure output directory exists
        cumulants_str   = '_cumulants' if self.cumulants else ''
        output_name     = (f'{self.scatterer}_{self.base_type}_len{self.SignalLen}_'
                           f'snr{self.snr_label}_order{self.max_fit_order}{cumulants_str}.csv')
        print(output_name)
        output_dir      = root / 'data' / 'returns'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file     = output_dir / output_name
        # create a new database file if none exists, or pick up where left off otherwise
        if output_file.exists():
            processed_df    = pd.read_csv(output_file)
            start_idx       = len(processed_df)
            print(f'  Resuming from record {start_idx}.')
        else:
            start_idx       = 0
            print('  Starting processing from the beginning.')
        # process in chunks to deal with running out of time connected to the runtime
        st  = time.time()
        for start in range(start_idx, len(db), chunk_size):
            end         = min(start + chunk_size, len(db))
            chunk       = db.iloc[start:end].copy()
            print(f'    processing records {start} to {end - 1}')
            start_time  = time.time()
            chunk_df    = chunk.apply(lambda row: pfl.add_metrics(row, self.t, self.freqs,
                                                                  self.sr, self.scatterer, power=self.power,
                                                                  real=self.real, snr=self.snr,
                                                                  add_noise=self.add_noise,
                                                                  segments=self.segments,
                                                                  SampleRate=self.SampleRate,
                                                                  add_spectrals=self.add_spectrals,
                                                                  add_polys=self.add_polys,
                                                                  add_freqs=self.add_freqs,
                                                                  add_corrs=self.add_corrs,
                                                                  max_fit_order=self.max_fit_order,
                                                                  norm=self.norm, from_cumulants=self.cumulants,
                                                                  diagnostics=self.diagnostics,
                                                                  use_weights=self.use_weights,
                                                                  alpha=self.alpha, q=self.q), axis=1).tolist()
            # check to make sure all rows are as long as the header so we don't get row mismatches
            for i, row_out in enumerate(chunk_df):
                if len(row_out) != len(all_cols):
                    raise ValueError(
                        f"Column mismatch at chunk row {i}: "
                        f"got {len(row_out)} values, expected {len(all_cols)} columns.\n"
                        f"scatterer={self.scatterer}, base_type={self.base_type}, "
                        f"snr={self.snr_label}, SignalLen={self.SignalLen}")
            chunk_df    = pd.DataFrame(chunk_df, columns=all_cols)
            # save chunk to .csv
            if output_file.exists():
                chunk_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                chunk_df.to_csv(output_file, mode='w', header=True, index=False)
            end_time    = time.time()
            print('      Time taken:', np.round((end_time - start_time)/60, 3), 'minutes.')
        et = time.time()
        print('  Processing complete! Time taken:', np.round((et - st)/60, 3), 'minutes.')

    def test_one_chunk(self, root, chunk_size=5, seed=12345, return_source=False, plot=False):
        '''
        test the configured pipeline on a small random subset before a full run
        - 'seed' controls selection of the test database rows; noise is still generated randomly
        '''
        root        = Path(root)
        input_file  = root / f'{self.scatterer}_database.csv'
        db          = pd.read_csv(input_file)
        chunk       = db.sample(n=chunk_size, random_state=seed).copy()
        chunk       = chunk.reset_index(drop=True)
        chunk_df    = chunk.apply(lambda row: pfl.add_metrics(row, self.t, self.freqs,
                                                              self.sr, self.scatterer, power=self.power,
                                                              real=self.real, snr=self.snr,
                                                              add_noise=self.add_noise,
                                                              segments=self.segments,
                                                              SampleRate=self.SampleRate,
                                                              add_spectrals=self.add_spectrals,
                                                              add_polys=self.add_polys,
                                                              add_freqs=self.add_freqs,
                                                              add_corrs=self.add_corrs,
                                                              max_fit_order=self.max_fit_order,
                                                              norm=self.norm, from_cumulants=self.cumulants,
                                                              diagnostics=self.diagnostics,
                                                              use_weights=self.use_weights,
                                                              alpha=self.alpha, q=self.q, plot=plot),
                                  axis=1).tolist()
        # get initial header row column names
        moment_cols = db.columns.tolist() if self.scatterer == 'delta' else [
                            'mu1', 'mu2', 'sigma', 'w1', 'mean', 'variance', 'std', 'skew',
                            'kurtosis', 'inv_kurt', 'hyperskewness', 'hyperkurtosis']
        all_cols    = moment_cols + self.cols
        # Validate every row before creating DataFrame
        expected_len = len(all_cols)
        for i, row_out in enumerate(chunk_df):
            actual_len = len(row_out)
            if actual_len != expected_len:
                raise ValueError(
                    f"Row output length does not match column count.\n"
                    f"Test row: {i}\n"
                    f"Actual output length: {actual_len}\n"
                    f"Expected column length: {expected_len}\n"
                    f"scatterer={self.scatterer}, base_type={self.base_type}, "
                    f"SignalLen={self.SignalLen}, snr={self.snr_label}\n"
                    f"add_spectrals={self.add_spectrals}, "
                    f"add_polys={self.add_polys}, "
                    f"add_freqs={self.add_freqs}, "
                    f"add_corrs={self.add_corrs}")
        chunk_out = pd.DataFrame(chunk_df, columns=all_cols)
        print("Test chunk passed.")
        print(f"Rows tested: {len(chunk_out)}")
        print(f"Columns produced: {len(chunk_out.columns)}")
        print(f"scatterer={self.scatterer}, base_type={self.base_type}, "
            f"SignalLen={self.SignalLen}, snr={self.snr_label}")
        if return_source:
            return chunk, chunk_out
        return chunk_out