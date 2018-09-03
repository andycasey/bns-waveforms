import numpy as np
import scipy.optimize as op

import stan_utils as stan
from utils import strain_fitting_function_corrected


class BNSWaveformEmulator(object):

    def __init__(self, eos_parameter_name="Kappa_calc", **kwargs):
        self.eos_parameter_name = eos_parameter_name
        return None


    @property
    def hierarchical_parameter_names(self):
        return ("C", )


    def fit(self, waveform_parameters, frequencies, amplitudes, **kwargs):
        r"""
        Fit the model given the parameters of the waveforms, the common
        frequencies that those waveforms are calculated on, and the amplitudes
        of the Fourier spectrum for those waveforms.

        :param waveform_parameters:
            A table of parameters for each of the numerical relativity waveforms
            that has [n_waveforms, ] rows and may contain the following
            properties (as columns):

            The compactness `C`,
            the calculated :math:`\kappa` value `Kappa_calc`,
            the mass of the primary `M1`,
            the frequency of the high-frequency peak `f2`,
            the dimensionless tidal deformability `Lambda`

            ..math:
                \Lambda = \left(\frac{\lambda}{M_{1}^{5}}\right)^{1/5}

            where

            ..math:
                \lambda = \frac{2}{3}\overbar{kappa}_{2}^{T}\overbar{R}^5

        :param frequencies:
            The common frequencies (in kHz) that each numerical relativity
            waveform is calculated on. This should be an array of shape
            [n_frequencies, ].

        :param amplitudes:
            The amplitudes of the Fourier spectrum for the numerical relativity
            waveforms. This should be an array of shape
            [n_waveforms, n_frequencies].
        """

        self._frequencies = frequencies
        self._amplitudes = amplitudes
        self._waveform_parameters = waveform_parameters

        # Hierarchically fit the waveform parameters.
        p_opt_hpm, data_dict = self._hierarchically_fit_waveform_parameters(
            waveform_parameters)

        # Frequency-shift waveforms so that they are aligned at mean f2.
        shifted_amplitudes, f2_coeff, f2_mean = self._frequency_shift_waveforms(
            waveform_parameters, frequencies, amplitudes)

        # Fit the waveforms with a linear model of the whitened labels, and the
        # mass and dimensionless tidal deformability.
        labels = np.vstack([
            waveform_parameters["M1"],
            waveform_parameters[self.eos_parameter_name],
            data_dict["y"].T
        ]).T

        mu, sigma = (np.mean(labels, axis=0), np.std(labels, axis=0))
        whitened_labels = (labels - mu)/sigma

        model = stan.load_stan_model("waveform.stan")

        N, P = whitened_labels.shape
        F = frequencies.size
        data_dict = dict(F=frequencies.size, N=N, P=P, y=shifted_amplitudes,
                         whitened_labels=whitened_labels)

        # TODO: move default op  kwds to somewhere else.
        kwds = dict(iter=100000, tol_param=1e-12, tol_obj=1e-12, tol_grad=1e-12,
                    tol_rel_grad=1e8)
        kwds.update(kwargs)

        p_opt_waveform = model.optimizing(data=data_dict, **kwds)

        # Save attributes so that we can make predictions.
        self._p_opt_waveform = p_opt_waveform
        self._p_opt_hpm = p_opt_hpm
        self._p_opt_f2 = (f2_mean, f2_coeff)
        self._p_opt_label_whiten = (mu, sigma)

        return self


    def estimate_waveform_parameters(self, M, eos_parameter):
        r"""
        Estimate the equation of state parameters given the mass of the primary
        :math:`M_1` and the dimensionless tidal deformability :math:`\Lambda`.

        :param M:
            A list-like object with the mass(es) of the primary.

        :param eos_parameter:
            A list-like object containing the values of the EOS-like parameter
            specified by the `eos__parameter_name` attribute of the model.
            This should be the same length as `M`.

        :returns:
            The estimated compactness, given the mass and equation of state
            parameter.
        """

        M, eos_parameter = (np.atleast_1d(M), np.atleast_1d(eos_parameter))

        P = self._p_opt_hpm["a"].shape[1]
        design_matrix = np.vstack([eos_parameter, np.ones(eos_parameter.size)])
        alpha = (self._p_opt_hpm["a"].T @ design_matrix)
        beta = (self._p_opt_hpm["b"].T @ design_matrix)
        return np.array([alpha[j] * M**beta[j] for j in range(P)]).T


    def estimate_frequency_shifts(self, eos_parameter):
        f2_mean, f2_coeff = self._p_opt_f2

        eos_parameter = np.atleast_1d(eos_parameter)
        frequency_shifts = np.zeros(eos_parameter.size, dtype=float)

        for i, value in enumerate(eos_parameter):
            amplitude_spectrum = self._amplitudes[self._waveform_parameters[self.eos_parameter_name] == value]
            frequency_shifts[i] = self._frequencies[np.argmax(amplitude_spectrum)] - f2_mean

        return frequency_shifts
        

    def predict(self, M, eos_parameter, frequency_shifts=None):
        r"""
        Predict a post-merger gravitational wave signal given the mass and a
        parameter describing the equation of state.

        :param M:
            The mass. This can be a single value or an array-like.

        :param eos_parameter:
            A value that describes the equation of state. This parameter can be
            described by the `eos_parameter_name` attribute of the model.

        :param frequency_shifts: [optional]
            The frequency shift to apply to the predicted amplitude spectrum.
            If `None` is supplied then the frequency will be estimated given
            the equation of state parameter.
        """

        # Calculate other properties given M, Lambda.
        waveform_parameters = self.estimate_waveform_parameters(M, eos_parameter)

        # Include mass and tidal deformability.
        parameters = np.vstack([M, eos_parameter, waveform_parameters.T]).T

        # Whiten the parameters.
        mu, sigma = self._p_opt_label_whiten
        whitened_parameters = (parameters - mu)/sigma

        # Generate the aligned waveform.
        N, _ = whitened_parameters.shape
        design_matrix = np.vstack([np.ones((1, N)), whitened_parameters.T])
        aligned_waveforms = (self._p_opt_waveform["theta"] @ design_matrix).T

        # Estimate the frequency shift needed for this waveform.
        if frequency_shifts is None:
            frequency_shifts = self.estimate_frequency_shifts(eos_parameter)
        else:
            frequency_shifts = np.atleast_1d(frequency_shifts)

        # Shift the waveform.
        waveforms = np.zeros((N, self._frequencies.size), dtype=float)
        for i, (fs, aligned_waveform) \
        in enumerate(zip(frequency_shifts, aligned_waveforms)):
            waveforms[i] = np.interp(self._frequencies,
                                     self._frequencies + fs,
                                     aligned_waveform)

        return waveforms


    def test(self, amplitudes, psd, initial=None, supply_s2_for_ff=False):
        r"""
        Estimate the waveform parameters, given an amplitude spectrum.
        """

        # TODO: Don't assume that frequencies is the same as common frequencies
        
        if initial is None:
            initial = [
                np.mean(self._waveform_parameters["M1"]),
                np.mean(self._waveform_parameters[self.eos_parameter_name]),
                self._frequencies[np.argmax(amplitudes)] - self._p_opt_f2[0],
            ]

        # TODO: A better optimisation for this.
        def cost(params):
            M, eos_parameter, frequency_shift = params
            prediction = self.predict(M, eos_parameter, [frequency_shift]) 
            ff = strain_fitting_function_corrected(
                amplitudes, prediction,
                self._frequencies, psd,
                self._p_opt_waveform["sigma"]**2 if supply_s2_for_ff else 0)

            return 1 - ff

        p_opt = op.fmin(cost, initial, disp=False, xtol=1e-8, ftol=1e-8,
                        maxiter=100000, maxfun=100000)
        # Offset the f2 shfit:
        p_opt[-1] += self._p_opt_f2[0]
        return p_opt


    def _frequency_shift_waveforms(self, waveform_parameters, frequencies,
        amplitudes):

        central_idx = int(np.median(np.argmax(amplitudes, axis=1)))
        shifted_amplitudes = np.zeros_like(amplitudes)
        N, F = shifted_amplitudes.shape
        for i in range(N):
            idx = np.argmax(amplitudes[i])

            shifted_amplitudes[i] = np.interp(np.arange(F),
                                              np.arange(F) - idx + central_idx,
                                              amplitudes[i])

        return (shifted_amplitudes, 0, self._frequencies[central_idx])



    def _hierarchically_fit_waveform_parameters(self, waveform_parameters, **kwargs):
        r"""
        Fit a hierarchical model to the waveform parameters.

        :param waveform_parameters:
            A table of parameters for each of the numerical relativity waveforms
            that has [n_waveforms, ] rows and contains at least the following
            properties (as columns):

            The compactness `C`,
            the calculated :math:`\kappa` value `Kappa_calc`,
            the mass of the primary `M1`,
            the frequency of the high-frequency peak `f2`,
            the dimensionless tidal deformability `Lambda`

            ..math:
                \Lambda = \left(\frac{\lambda}{M_{1}^{5}}\right)^{1/5}

            where

            ..math:
                \lambda = \frac{2}{3}\overbar{kappa}_{2}^{T}\overbar{R}^5
        """


        y = np.array([waveform_parameters[pn] \
                      for pn in self.hierarchical_parameter_names]).T

        N, D = y.shape
        M1 = waveform_parameters["M1"]
        eos_param = waveform_parameters[self.eos_parameter_name]

        data_dict = dict(M1=M1, eos_param=eos_param, N=N, D=D, y=y)
        kwds = dict(iter=10000, tol_param=1e-12, tol_obj=1e-12, tol_grad=1e-12,
                    tol_rel_grad=1e8)
        kwds.update(kwargs)

        p_opt = stan.load_stan_model("hpm.stan").optimizing(data_dict, **kwds)

        return (p_opt, data_dict)
