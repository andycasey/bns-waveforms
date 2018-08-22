import numpy as np
import scipy.optimize as op

import stan_utils as stan


class BNSWaveformEmulator(object):

    def __init__(self, **kwargs):
        pass


    @property
    def parameter_names(self):
        return ("C", )


    def fit(self, waveform_parameters, frequencies, amplitudes, **kwargs):
        r"""
        Fit the model given the parameters of the waveforms, the common
        frequencies that those waveforms are calculated on, and the amplitudes
        of the Fourier spectrum for those waveforms.

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
            waveform_parameters, self.parameter_names)

        # Frequency-shift waveforms so that they are aligned at mean f2.
        shifted_amplitudes, f2_coeff, f2_mean = self._frequency_shift_waveforms(
            waveform_parameters, frequencies, amplitudes)

        # Fit the waveforms with a linear model of the whitened labels, and the
        # mass and dimensionless tidal deformability.
        labels = np.vstack([
            waveform_parameters["M1"],
            1.0/waveform_parameters["Lambda"],
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


    def estimate_waveform_parameters(self, M, Lambda):
        r"""
        Estimate the equation of state parameters given the mass of the primary
        :math:`M_1` and the dimensionless tidal deformability :math:`\Lambda`.

        :param M:
            A list-like object with the mass(es) of the primary.

        :param Lambda:
            A list-like object containing the dimensionless tidal deformability.
            This should be the same length as `M`.

        :returns:
            The estimated numerical relativity parameters. Currently hard-coded
            as:

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

        M, Lambda = (np.atleast_1d(M), np.atleast_1d(Lambda))

        P = self._p_opt_hpm["a"].shape[1]
        design_matrix = np.vstack([Lambda, np.ones(Lambda.size)])
        alpha = (self._p_opt_hpm["a"].T @ design_matrix)
        beta = (self._p_opt_hpm["b"].T @ design_matrix)
        return np.array([alpha[j] * M**beta[j] for j in range(P)]).T


    def estimate_frequency_shifts(self, Lambda):
        f2_mean, f2_coeff = self._p_opt_f2
        return np.array([
            (self._frequencies[np.argmax(self._amplitudes[self._waveform_parameters["Lambda"] == l])] - f2_mean) \
            for l in Lambda])


    def predict(self, M, Lambda, frequency_shifts=None):
        r"""
        Predict a post-merger gravitational wave signal given the mass and
        dimensionless tidal deformability.
        """

        # Calculate other properties given M, Lambda.
        parameters = self.estimate_waveform_parameters(M, Lambda)

        # Include mass and tidal deformability.
        parameters = np.vstack([M, 1.0/Lambda, parameters.T]).T

        # Whiten the parameters.
        mu, sigma = self._p_opt_label_whiten
        whitened_parameters = (parameters - mu)/sigma

        # Generate the aligned waveform.
        N, _ = whitened_parameters.shape
        design_matrix = np.vstack([np.ones((1, N)), whitened_parameters.T])
        aligned_waveforms = (self._p_opt_waveform["theta"] @ design_matrix).T

        # Estimate the frequency shift needed for this waveform.
        if frequency_shifts is None:
            frequency_shifts = self.estimate_frequency_shifts(Lambda)

        # Shift the waveform.
        waveforms = np.zeros((N, self._frequencies.size), dtype=float)
        for i, (fs, aligned_waveform) \
        in enumerate(zip(frequency_shifts, aligned_waveforms)):
            waveforms[i] = np.interp(self._frequencies,
                                     self._frequencies + fs,
                                     aligned_waveform)

        return waveforms



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



    def _hierarchically_fit_waveform_parameters(self, waveform_parameters,
        parameter_names, **kwargs):
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


        y = np.array([waveform_parameters[pn] for pn in parameter_names]).T

        N, D = y.shape
        M1, Lambda = (waveform_parameters["M1"], waveform_parameters["Lambda"])

        data_dict = dict(M1=M1, Lambda=Lambda, N=N, D=D, y=y)
        kwds = dict(iter=10000, tol_param=1e-12, tol_obj=1e-12, tol_grad=1e-12,
                    tol_rel_grad=1e8)
        kwds.update(kwargs)

        p_opt = stan.load_stan_model("hpm.stan").optimizing(data_dict, **kwds)

        return (p_opt, data_dict)



if __name__ == "__main__":

    import pickle
    import matplotlib.cm
    from astropy.table import Table


    with open("input_variables.pkl", "rb") as fp:
        data = pickle.load(fp)

    waveform_parameters, label_names, waveform_names, amplitudes, \
        maximum_amplitude, waveform_ivar, scaled_asd, bns_maxima, _, __, ___, \
        psd, frequencies = data


    with open("pl_params.pkl", "rb") as fp:
        eos_descr, eos_parameters, eos_coeff = pickle.load(fp)

    waveform_parameters = Table.from_pandas(waveform_parameters)

    waveform_parameters["M1"] = 1.375 * waveform_parameters["M1"]

    # Need unscaled f2 values. Load from text.
    t2 = Table.read("10.1103_t2.txt", format="ascii")
    waveform_parameters["f2"] = 1000 * np.array(
        [t2["f_2"][t2["model"] == wn][0] for wn in waveform_names])

    waveform_parameters["f_max"] = 1000 * np.array(
        [t2["f_max"][t2["model"] == wn][0] for wn in waveform_names])

    
    # Calculate tidal Love number for equal mass binaries (eq 14 of 1604.00246)
    waveform_parameters["lambda"] = (2.0/3.0) * waveform_parameters["Meank2"] \
                                  * waveform_parameters["MeanR"]**5

    # Calculate dimensionless tidal deformability (p12 of 1604.00246).
    waveform_parameters["Lambda"] = (waveform_parameters["lambda"]/waveform_parameters["M1"]**5)**(1.0/5)



    # Fit the params.
    emulator = BNSWaveformEmulator()
    p_opt, data_dict = emulator._hierarchically_fit_waveform_parameters(
        waveform_parameters, emulator.parameter_names)


    eos_descriptions = [ea.split("-")[0] for ea in waveform_names]
    unique_eos_descriptions = list(set([ea for ea in eos_descriptions]))
    waveform_eos_idx = np.array(
        [unique_eos_descriptions.index(ea) for ea in eos_descriptions])


    x, y = (data_dict["M1"], data_dict["y"])

    N, D = data_dict["y"].shape
    E = len(unique_eos_descriptions)
    design_matrix = np.vstack([waveform_parameters["Lambda"], np.ones(N)])
    alpha = (p_opt["a"].T @ design_matrix).T
    beta = (p_opt["b"].T @ design_matrix).T

    cmap = matplotlib.cm.viridis

    for j, parameter_name in enumerate(emulator.parameter_names):

        fig, ax = plt.subplots()

        for unique_eos_idx in list(set(waveform_eos_idx)):

            color = cmap(float(unique_eos_idx)/E)

            match = waveform_eos_idx == unique_eos_idx

            yi = alpha[match, j] * x[match]**beta[match, j]

            ax.plot(x[match], yi, c=color, alpha=0.5)
            ax.scatter(x[match], y[match, j], c=color, vmin=0, vmax=E)

        #fitted_labels[:, 2 + j] = alpha * x**beta

        ax.set_title("{} (hierarchical; tidal)".format(parameter_name))


    emulator = BNSWaveformEmulator()
    emulator.fit(waveform_parameters, frequencies, amplitudes)

    predictions = emulator.predict(waveform_parameters["M1"],
                                   waveform_parameters["Lambda"])


    from utils import strain_fitting_function_corrected

    N, F = amplitudes.shape
    fitting_factors = np.zeros(N)

    for i in range(N):



        fitting_factors[i] = strain_fitting_function_corrected(
            amplitudes[i], predictions[i], frequencies, psd,
            emulator._p_opt_waveform["sigma"]**2)


    fig, ax = plt.subplots()
    ax.hist(fitting_factors)

    ax.set_xlabel("fitting factor")



    loocv_fitting_factors = np.zeros(N)
    for i in range(N):

        mask = np.ones(N, dtype=bool)
        mask[i] = False

        emulator = BNSWaveformEmulator()
        emulator.fit(waveform_parameters[mask], frequencies, amplitudes[mask])

        frequency_shift = frequencies[np.argmax(amplitudes[i])] - emulator._p_opt_f2[0]

        prediction = emulator.predict(waveform_parameters["M1"][i],
                                      waveform_parameters["Lambda"][i],
                                      frequency_shifts=[frequency_shift])

        loocv_fitting_factors[i] = strain_fitting_function_corrected(
            amplitudes[i], prediction, frequencies, psd,
            emulator._p_opt_waveform["sigma"]**2)

        print("LOOCV: {} {}".format(i, loocv_fitting_factors[i]))


        fig, ax = plt.subplots()

        ax.set_title("{0} {1} ff: {2:.2f}, loocv ff: {3:.2f}".format(i,
            waveform_names[i],
            fitting_factors[i], loocv_fitting_factors[i]))
        ax.plot(frequencies, amplitudes[i], c="k")
        ax.plot(frequencies, predictions[i], c="tab:blue")
        ax.plot(frequencies, prediction[0], c="tab:red")

        ax.axvline(waveform_parameters["f2"][i], c="#666666")

        ax.semilogx()


    fig, ax = plt.subplots()
    ax.hist(loocv_fitting_factors)

    ax.set_xlabel("loocv fitting factor")
