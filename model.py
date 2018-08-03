import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
from astropy.table import  Table


with open("input_variables.pkl", "rb") as fp:
    data = pickle.load(fp)

labels, label_names, waveform_names, waveform_amplitude, maximum_amplitude, \
    waveform_ivar, scaled_asd, bns_maxima, _, __, ___, psd, frequency = data


with open("pl_params.pkl", "rb") as fp:
    eos_descr, eos_parameters, eos_coeff = pickle.load(fp)


labels = Table.from_pandas(labels)

# Calculate tidal Love number for equal mass binaries (eq 14 of 1604.00246)
labels["lambda"] = (2.0/3.0) * labels["Meank2"] * labels["MeanR"]**5

# Calculate dimensionless tidal deformability (p12 of 1604.00246).
labels["Lambda"] = (labels["lambda"]/labels["M1"]**5)**(1.0/5)



# params = alpha*M1**beta
eos_descriptions = [ea.split("-")[0] for ea in waveform_names]
unique_eos_descriptions = list(set([ea for ea in eos_descriptions]))
waveform_eos_idx = np.array(
    [unique_eos_descriptions.index(ea) for ea in eos_descriptions])

vmin, vmax = (0, len(unique_eos_descriptions) - 1)

label_name = "C"
label_index = label_names.index(label_name)

x = 1.375 * labels["M1"]
y = labels[label_name]


fig, ax = plt.subplots()
for i, eos_description in enumerate(unique_eos_descriptions):

    eos = (waveform_eos_idx == i)
    scat = ax.scatter(x[eos], y[eos],
                      c=waveform_eos_idx[eos], vmin=vmin, vmax=vmax)

    alpha, beta = eos_coeff[i].T
    xi = np.percentile(x, np.linspace(0, 100, 100))
    yi = alpha[label_index] * xi**beta[label_index]

    ax.plot(xi, yi, c=scat.to_rgba(waveform_eos_idx[eos][0]))


import matplotlib.cm

cmap = matplotlib.cm.viridis

for ea in unique_eos_descriptions:

    fig, ax = plt.subplots()
    for i, waveform in enumerate(waveform_amplitude):
        if eos_descriptions[i] != ea: continue
        ax.plot(waveform, c=cmap((x[i] - x.min())/np.ptp(x)))

    ax.set_title(ea)



# Fit the relationship between equation of state, M1, and C

#def hierarchical_fit(masses, labels, eos_descriptions):

masses = x

# We need indices for the eos descriptions first.
eos_descriptions = np.array(eos_descriptions)
unique_eos_descriptions = np.unique(eos_descriptions)

mean_labels_per_eos = np.array([np.mean(y[ued == eos_descriptions]) \
                                for ued in unique_eos_descriptions])

eos_indices = np.argsort(mean_labels_per_eos)
eos_white_ranks = eos_indices/np.max(eos_indices)

# labels = alpha * masses ** beta
# where alpha = f(eos_white_ranks)
#  and  beta  = f(eos_white_ranks)

from stan_utils import load_stan_model, sampling_kwds

model = load_stan_model("model.stan")


N, M, D = (len(x), len(unique_eos_descriptions), 1)

for label_name in ("C", "Kappa_calc", "Mb", "Meank2", "MoI"):

    y = np.atleast_2d(labels[label_name]).reshape((-1, 1))


    p_opt = model.optimizing(data=dict(x=x, y=y, N=N, M=M, D=D,
                                       eos_index=waveform_eos_idx + 1))

    print(label_name, p_opt["eos_coeff"])

    fig, ax = plt.subplots()
    for i, eos_idx in enumerate(np.unique(waveform_eos_idx)):

        match = (waveform_eos_idx == eos_idx)
        scat = ax.scatter(x[match], y[match], c=cmap(float(i)/M), vmin=0, vmax=M)

        # predict...
        xi = np.percentile(x, np.linspace(0, 100, 100))

        eos_coeff = p_opt["eos_coeff"][i]
        alpha = np.atleast_1d(p_opt["alpha_slope"])[0] * eos_coeff \
              + np.atleast_1d(p_opt["alpha_offset"])[0]
        beta = np.atleast_1d(p_opt["beta_slope"])[0] * eos_coeff \
             + np.atleast_1d(p_opt["beta_offset"])[0]

        yi = alpha * xi**beta

        ax.plot(xi, yi, c=cmap(float(i)/M), alpha=0.5)

    ax.set_title(label_name)


model = load_stan_model("model.stan")

fit_label_names = ("C", "Kappa_calc", "Mb", "MoI")
D = len(fit_label_names)

y = np.array([labels[ln] for ln in fit_label_names]).T

data_dict = data=dict(x=x, y=y, N=N, M=M, D=D, eos_index=waveform_eos_idx + 1)
p_opt = model.optimizing(data_dict,
                         iter=10000, tol_param=1e-12, tol_obj=1e-12, 
                         tol_grad=1e-12,tol_rel_grad=1e8)

#samples = model.sampling(**sampling_kwds(data=data_dict, init=p_opt))

#plots.traceplot(samples, pars=("eos_coeff", "eos_mu", "sigma", "alpha_slope", "alpha_offset"))

L  = len(fit_label_names)
fitted_labels = np.nan * np.ones((N, 2 + L))
fitted_labels[:, 0] = p_opt["eos_coeff"][waveform_eos_idx]
fitted_labels[:, 1] = x

for j, label_name in enumerate(fit_label_names):


    fig, ax = plt.subplots()
    for i, eos_idx in enumerate(np.unique(waveform_eos_idx)):

        match = (waveform_eos_idx == eos_idx)
        scat = ax.scatter(x[match], y[match, j], c=cmap(float(i)/M), vmin=0, vmax=M)

        # predict...
        xi = np.percentile(x, np.linspace(0, 100, 100))

        eos_coeff = p_opt["eos_coeff"][i]
        alpha = p_opt["alpha_slope"][j] * eos_coeff + p_opt["alpha_offset"][j]
        beta = p_opt["beta_slope"][j] * eos_coeff + p_opt["beta_offset"][j]

        yi = alpha * xi**beta
        fitted_labels[:, 2 + j] = alpha * x**beta

        ax.plot(xi, yi, c=cmap(float(i)/M), alpha=0.5)

    ax.set_title("{} (hierarchical)".format(label_name))




from sklearn.decomposition import FactorAnalysis


central_idx = int(np.median(np.argmax(waveform_amplitude, axis=1)))

aligned = np.zeros_like(waveform_amplitude)

fig, ax = plt.subplots()
for i in range(25):

    idx = np.argmax(waveform_amplitude[i])

    aligned[i, :] = np.interp(np.arange(99),
                              np.arange(99) - idx + central_idx,
                              waveform_amplitude[i])


    ax.plot(aligned[i])


model = FactorAnalysis(n_components=5)
model.fit(aligned)

fig, ax = plt.subplots()
for i in range(5):
    ax.plot(model.components_[i])


waveform_model = load_stan_model("waveforms.stan")

mu = np.mean(fitted_labels, axis=0)
sigma = np.std(fitted_labels, axis=0)

whiten = lambda x: (x - mu)/sigma
design_matrix = lambda labels: np.hstack([1, whiten(labels)])

N, D = aligned.shape
N, L = fitted_labels.shape
data_dict  = dict(N=N, y=aligned, L=L, D=D, 
                  white_labels=whiten(fitted_labels))

# Use predicted labels
wf_opt = waveform_model.optimizing(data_dict,
                                   iter=100000, tol_param=1e-12, tol_obj=1e-12, 
                                   tol_grad=1e-12,tol_rel_grad=1e8)


from utils import strain_fitting_function_corrected


fitting_factors = np.zeros(N)

# Plot predictions for first thing.
for i in range(N):

    idx = np.argmax(waveform_amplitude[i])# + central_idx

    realigned = np.interp(np.arange(99) - idx + central_idx,
                          np.arange(99),
                          aligned[i, :])

    aligned_model = np.interp(np.arange(99) - idx + central_idx, np.arange(99),
                             wf_opt["theta"] @ design_matrix(fitted_labels[i]))

    fig, ax = plt.subplots()
    ax.plot(frequency, waveform_amplitude[i], c="k")
    ax.plot(frequency, realigned, c="b", alpha=0.5)

    start, end = idx - central_idx, -idx + central_idx
    start = max(start, 0)
    end = 100

    ax.plot(frequency[start:end], aligned_model[start:end], c="r")


    ff = strain_fitting_function_corrected(waveform_amplitude[i],
                                           aligned_model,
                                           frequency, 
                                           psd,
                                           wf_opt["sigma"]**2)

    print(i, ff)

    fitting_factors[i] = ff

