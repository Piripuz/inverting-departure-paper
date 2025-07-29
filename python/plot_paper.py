import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})
import matplotlib as mpl

import numpy as np
from jax import numpy as jnp
from jax import grad, vmap
from jaxopt import GradientDescent, Bisection

from scripts.travel_times import asymm_gaussian, asymm_gaussian_plateau
from scripts.generate_data import cost, generate_arrival
from scripts.utils import TravelTime
from scripts.find_points import find_bs, find_gs
from scripts.retrieve_data import total_liks
#%%

early_color = "green"
late_color="red"
ot_color = "blue"
tt_color = "purple"
cost_color = "orange"

f_form = lambda x, _: mpl.dates.num2date(x/24).strftime("%H:%M")
formatter = mpl.ticker.FuncFormatter(f_form)


quality = 300

#%%

tt = asymm_gaussian(sigma_l = .7)
star = 9.2
beta = .7

fig, ax = plt.subplots(figsize=(6, 4))

x = np.linspace(8, 9.5, 200)
ax.plot(x, tt(x), color=tt_color, label=r"Travel Time Function $tt_a(t_a)$")

ax.vlines(star, 0, tt(star), color=ot_color, linestyle="dashed")
ax.text(star + .05, .03, r"$t^*$", color=ot_color)

ax.hlines(tt(star), star, x[-1], color=ot_color, linestyle="dashed", linewidth=1)
ax.text(x[-1] + .01, tt(star), r"$C(t^*)$", va="center", color=ot_color)

f = lambda x: tt(x) + beta*(star - x)
ts = np.r_[8.2, 8.6, 8.85]
colors = iter(mpl.color_sequences['Dark2'])
for i, t in enumerate(ts):
    col = next(colors)
    ax.plot([t, star], [tt(t), f(t)], color=col)
    
    ax.vlines(t, 0, tt(t), color=col, linestyle="dashed", linewidth=1)
    ax.text(t + .02, -.01, fr"$t_{i}$", color=col)

    ax.hlines(f(t), star, x[-1], color=col, linestyle="dashed", linewidth=1)
    ax.text(x[-1] + .01, f(t), fr"$C(t_{i})$", va="center", color=col)
ax.axis('off')
ax.legend()
fig.savefig("../img/early_arrivals_cost.png", dpi=quality)
plt.close()

#%%

tt = lambda x: x + jnp.exp(-(x+.2)**2*4)/2 + jnp.exp(-(x-1.5)**2*5)/.9 + jnp.exp(-(x-.7)**2*7)/1.25

beta = 1.7
f = lambda x: tt(x) - beta*x

bs = jnp.r_[GradientDescent(f).run(0.).params, GradientDescent(f).run(1.25).params]
def find_end(start, tt):
    curr = start+.1
    old = start
    while tt(start) + beta*(curr - start) < tt(curr):
        old = curr
        curr += .1
    res = Bisection(lambda curr: tt(start) + beta*(curr - start) - tt(curr), old, curr).run()
    return res.params
bs_end = np.r_[find_end(bs[0], tt), find_end(bs[1], tt)]


x = np.linspace(0, 2, 200)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(x, tt(x), color=tt_color, label=r"Travel Time Function $tt_a(t_a)$")

colors = mpl.color_sequences['Dark2']

for i in range(2):
    ax.plot([bs[i], bs_end[i]], tt(np.r_[bs[i], bs_end[i]]), color=colors[i])
    ax.vlines(bs[i], .3, tt(bs[i]), linestyle="dashed", color=colors[i])
    ax.fill_between(
        [bs[i], bs_end[i]],
        2.7,
        .3,
        color=colors[i],
        alpha=.15,
        edgecolor="none",
        label=fr"Zone in which $t_e^{{opt}} = t_{i}$"
    )
    ax.text(bs[i] + .03, .4, fr"$t_{i}$", color=colors[i])
ax.axis('off')
ax.legend()
fig.savefig("../img/early_arrivals_jump.png", dpi=quality)
plt.show()

#%%

x = np.linspace(8.5, 10.5, 200)
tt = asymm_gaussian()

fig, ax = plt.subplots(figsize=(7, 3))
dtt_plot = ax.plot(mpl.dates.num2date(x/24), vmap(grad(tt))(x), alpha=.5, label=r"Derivative of the Travel Time Function $tt'(t)$")
tt_plot = ax.plot(mpl.dates.num2date(x/24), tt(x), color=tt_color, label=r"Travel Time Function $tt(t)$")

ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
ax.set_xlabel(r"$t$ (h)")
ax.set_ylabel("Travel Time")
ax.legend(handles=[*tt_plot, *dtt_plot], loc=3)

fig.savefig("../img/theo_tt.png", dpi=quality, bbox_inches="tight")
fig.show()

#%%

fig, ax = plt.subplots(figsize=(7, 4))

bs = np.r_[8.4, 9.1]
gs = np.r_[9.6, 10.2]

x = np.linspace(7.5, 11, 400)
def func(x):
    if x > bs[0] and x < bs[1]:
        return bs[0]
    elif x > gs[0] and x < gs[1]:
        return gs[1]
    return x
y = np.vectorize(func)(x)
y[np.diff(y, prepend=y[0])>.1] = np.nan
ax.plot(x, y, c=tt_color, label = r"$t^{opt}(t^*)$")

loc_width = 1.1
loc_alpha = .6
ax.hlines(
    [bs[0], gs[1]],
    [bs[1], x[0]],
    [x[-1], gs[0]],
    color=[early_color, late_color],
    linestyle="dashed",
    linewidth=loc_width,
    alpha=loc_alpha,
    zorder=1.5
)
ax.plot(
    [bs[0], gs[1]],
    [bs[0], gs[1]],
    color=ot_color,
    linestyle="dashed",
    linewidth=loc_width,
    alpha=loc_alpha,
    zorder=1.5
    )

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
alpha = .1
ax.fill_between([bs[0], bs[1]], [ymin]*2, [ymax]*2, color=early_color, alpha=alpha, label='Early Arrivals')
ax.fill_between([gs[0], gs[1]], [ymin]*2, [ymax]*2, color=late_color, alpha=alpha, label='Late Arrivals')
ax.fill_between([gs[1], x[-1]], [ymin]*2, [ymax]*2, color=ot_color, alpha=alpha, label='On-time Arrivals')
ax.fill_between([x[0], bs[0]], [ymin]*2, [ymax]*2, color=ot_color, alpha=alpha)
ax.fill_between([bs[1], gs[0]], [ymin]*2, [ymax]*2, color=ot_color, alpha=alpha)

ax.vlines([bs[0], gs[1]], ymin, [bs[0], gs[1]], [early_color, late_color], 'dashed')
ax.text(bs[0] + .05, 7.6, r"$tt_a'(t^*) = \beta$", color=early_color)
ax.text(gs[1] + .05, 8, r"$tt_a'(t^*) = -\gamma$", color=late_color)

ax.legend(loc="upper left")
ax.set_xlabel(r"$t^*$ (h)")
ax.set_ylabel(r"$t^{opt}$ (h)")

ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

fig.savefig("../img/monotone_t_a.png", dpi=quality)
# fig.show()
plt.close(fig)

#%%

x = np.linspace(-.5, 24.5, 400)
tt = TravelTime(asymm_gaussian_plateau(plateau_len=.5, sigma_r = .35))
beta = .6
gamma = .8
star = 9.4
c = lambda t: cost(tt)(t, beta, gamma, star)
y = c(x)


fig, ax = plt.subplots(figsize=(7, 3))

cost_plot = ax.plot(x, y, color=cost_color, label=r"Cost function $C(t; \beta, \gamma, t^*)$")
low_min = GradientDescent(c, acceleration=False, maxiter=4000, stepsize=.05).run(0.).params
high_min = GradientDescent(c, acceleration=False, maxiter=4000, stepsize=.05).run(24.).params
ax.scatter(
    [low_min, high_min], [c(low_min), c(high_min)],
    color=[early_color, late_color],
    s=15,
    zorder=2.1
)

ax.scatter(
    [0, 24],
    [c(0), c(24)],
    color=[early_color, late_color],
    zorder=2.1,
    marker="s"
)

square_legend = mpl.lines.Line2D(
    [0], [0],
    color="w",
    markerfacecolor="grey",
    markersize=10,
    marker="s",
    label="Optimizer Initialization Points"
)
circle_legend = mpl.lines.Line2D(
    [0], [0],
    color="w",
    markerfacecolor="grey",
    markersize=7,
    marker="o",
    label="Optimizer Convergence Points"
)
ax.legend(handles=[cost_plot[0], square_legend, circle_legend])

ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel(r"$t$ (h)")
ax.set_ylabel(r"$C(t)$")

fig.savefig("../img/optimizer_cost.png", dpi=quality, bbox_inches="tight")
plt.close()

#%% Distribution of sampled arrival times

mu_beta = .6
mu_gamma = 2.4

tt = TravelTime(asymm_gaussian_plateau(plateau_len=.1, sigma_r = .3))
ts = generate_arrival(10000, tt, mu_beta, mu_gamma, mu_t=9.5, sigma=0.1, sigma_t=1, seed=123)[3]

b_i = find_bs(mu_beta, tt)[0]
g_e = find_gs(mu_gamma, tt)[1]

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(ts, 100, label="Empirical Arrival Times Density")
h = 1600
ax.vlines(
    [b_i, g_e],
    0, h,
    color=[early_color, late_color],
    linestyle="dashed",
    zorder=.9
)

h_off = .1

ax.text(
    b_i + h_off,
    h - 40,
    r"$tt'(t) = \mu_\beta$",
    color=early_color,
    ha="left",
    va = "top"
    )
ax.text(
    g_e + h_off,
    h - 200,
    r"$tt'(t) = -\mu_\gamma$",
    color=late_color,
    ha="left",
    va = "top"
)

ax.set_xlim(6.5, 12.5)
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel("Arrival Time")
ax.set_ylabel("Number of  Samples")

ax.legend()

fig.savefig("../img/hist_means.png", dpi=quality)
plt.close()

#%%

x = np.linspace(6.5, 12.5, 400)
y = total_liks(tt, x)(mu_beta, mu_gamma, 9.5, .1, 1.)

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(ts, 100, label="Empirical Arrival Times Density")
ax.fill_between(x, y*700, color=cost_color, alpha=.35, label="Likeilhood Function")

ax.set_xlim(6.5, 12.5)
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel("Arrival Time")
ax.set_ylabel("Number of  Samples")

ax.legend()

fig.savefig("../img/hist_lik.png", dpi=quality)
plt.close()

#%%

x = np.linspace(-2.5, 2.5, 300)
y = np.exp(-x**2)

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(x, y)

k_2 = 0.70711
k_1 = -k_2

ax.axvspan(
    x[0], k_1,
    color="orange",
    ec=None,
    alpha=.3,
    label=r"$\mathcal{D}_{conv}$"
)
ax.axvspan(
    k_2, x[-1],
    color="orange",
    ec=None,
    alpha=.3
)
ax.axvspan(
    k_1, k_2,
    color="purple",
    ec=None,
    alpha=.3,
    label=r"$\mathcal{D}_{conc}$"
)

ax.vlines(
    [k_1, k_2],
    0, 1,
    color="black",
    linestyle="dashed",
    linewidth=.7,
    transform=ax.get_xaxis_transform()
    )

ax.set_xticks([k_1, k_2], [r"$k_1$", r"$k_2$"])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc=2)

fig.savefig("../img/conv_conc_gauss.png", dpi=quality)
plt.close()

#%%

tt = TravelTime(asymm_gaussian_plateau(plateau_len=.2))
beta = .8
gamma = 2.4

b_i = find_bs(beta, tt)[0]
g_e = find_gs(gamma, tt)[1]

b_i_max = find_bs(tt.maxb, tt)[0]
g_e_max = find_gs(tt.maxg, tt)[1]


x = np.linspace(7.5, 10.5, 200)

early_line_len = .4
late_line_len = .18

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, tt.f(x), color=tt_color, label=r"Travel Time function $tt(t)$")

ax.plot(
    [b_i - early_line_len, b_i + early_line_len],
    [tt.f(b_i) - beta*early_line_len, tt.f(b_i) + beta*early_line_len],
    color=early_color,
    linestyle="solid",
    linewidth=1.2
)

early_rect_dist = .6
late_rect_dist = .35
rect_len = .1

ax.plot(
    [
        b_i - early_line_len*early_rect_dist,
        b_i - early_line_len*early_rect_dist,
        b_i - early_line_len*early_rect_dist - rect_len
    ],
    [
        tt.f(b_i) - beta*early_line_len*early_rect_dist,
        tt.f(b_i) - beta*early_line_len*early_rect_dist - beta*rect_len,
        tt.f(b_i) - beta*early_line_len*early_rect_dist - beta*rect_len
    ],
    color=early_color,
    linewidth=.9,
    alpha=.7
)

ax.text(
    b_i - early_line_len*early_rect_dist + .02,
    tt.f(b_i) - beta*early_line_len*early_rect_dist - beta*rect_len - .02,
    r"slope $\beta$",
    va="top",
    ha="center",
    color=early_color
)


ax.text(
    b_i + .03,
    tt.f(b_i) - .3,
    r"$t_i^e(\beta)$",
    color=early_color
)


ax.plot(
    [g_e - late_line_len, g_e + late_line_len],
    [tt.f(g_e) + gamma*late_line_len, tt.f(g_e) - gamma*late_line_len],
    color=late_color,
    linestyle="solid",
    linewidth=1.2
)

ax.plot(
    [
        g_e - late_line_len*late_rect_dist,
        g_e - late_line_len*late_rect_dist - rect_len,
        g_e - late_line_len*late_rect_dist - rect_len
    ],
    [
        tt.f(g_e) + gamma*late_line_len*late_rect_dist,
        tt.f(g_e) + gamma*late_line_len*late_rect_dist,
        tt.f(g_e) + gamma*late_line_len*late_rect_dist + gamma*rect_len
    ],
    color=late_color,
    linewidth=.9,
    alpha=.7
)

ax.text(
    g_e - late_line_len*late_rect_dist - rect_len - .04,
    tt.f(g_e) + gamma*late_line_len*late_rect_dist - .01,
    r"slope $\gamma$",
    va="top",
    ha="center",
    color=late_color
)


ax.text(
    g_e - .02,
    tt.f(g_e) - .25,
    r"$t_f^l(\gamma)$",
    color=late_color,
    ha="right"
)

ax.set_xlim(*ax.get_xlim())

ax.axvspan(
    ax.get_xlim()[0], b_i_max,
    color="orange",
    ec=None,
    alpha=.3,
    label=r"$\mathcal{D}_{conv}$"
)
ax.axvspan(
    b_i_max, g_e_max,
    color="purple",
    ec=None,
    alpha=.3,
    label=r"$\mathcal{D}_{conc}$"
)
ax.axvspan(
    g_e_max, ax.get_xlim()[1],
    color="orange",
    ec=None,
    alpha=.3
)

ax.legend(loc=2)
ax.xaxis.set_major_formatter(formatter)

ax.set_xlabel(r"$t^*$ (h)")
ax.set_ylabel("Travel Time")

ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])


ax.set_ylim(*ax.get_ylim())

ax.vlines(
    b_i,
    ax.get_ylim()[0],
    tt.f(b_i),
    linestyle="--",
    color=early_color
)

ax.vlines(
    g_e,
    ax.get_ylim()[0],
    tt.f(g_e),
    linestyle="--",
    color=late_color
)


fig.savefig("../img/tang_cond.png", dpi=quality)
plt.close()
# fig.show()

#%%


tt = TravelTime(asymm_gaussian_plateau(plateau_len=.2))
beta = .8
gamma = 2.4

bs = find_bs(beta, tt)
gs = find_gs(gamma, tt)

b_i, g_e = bs[0], gs[1]

x = np.linspace(8, 10, 200)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, tt.f(x), color=tt_color, label=r"Travel Time function $tt(t)$")

ax.plot(
    bs,
    tt.f(bs),
    color=early_color,
    linestyle="solid",
    linewidth=1.2
)

early_rect_dist = .2
late_rect_dist = .05
rect_len = .1

ax.plot(
    [
        b_i + early_rect_dist,
        b_i + early_rect_dist + rect_len,
        b_i + early_rect_dist + rect_len
    ],
    [
        tt.f(b_i) + beta*early_rect_dist,
        tt.f(b_i) + beta*early_rect_dist,
        tt.f(b_i) + beta*early_rect_dist + beta*rect_len
    ],
    color=early_color,
    linewidth=.9,
    alpha=.7
)

ax.text(
    b_i + early_rect_dist + rect_len + .02,
    tt.f(b_i) + beta*early_rect_dist - .02,
    r"slope $\beta$",
    va="top",
    ha="center",
    color=early_color
)


ax.text(
    b_i + .03,
    tt.f(b_i) - .3,
    r"$t_i^e(\beta)$",
    color=early_color
)

ax.text(
    bs[1] - .03,
    tt.f(b_i) - .1,
    r"$t_f^e(\beta)$",
    color=early_color,
    ha="right"
)

ax.plot(
    gs, tt.f(gs),
    color=late_color,
    linestyle="solid",
    linewidth=1.2
)

ax.plot(
    [
        g_e - late_rect_dist,
        g_e - late_rect_dist - rect_len,
        g_e - late_rect_dist - rect_len
    ],
    [
        tt.f(g_e) + gamma*late_rect_dist,
        tt.f(g_e) + gamma*late_rect_dist,
        tt.f(g_e) + gamma*late_rect_dist + gamma*rect_len
    ],
    color=late_color,
    linewidth=.9,
    alpha=.7
)

ax.text(
    g_e - late_rect_dist - rect_len - .0,
    tt.f(g_e) + gamma*late_rect_dist - .01,
    r"slope $\gamma$",
    va="top",
    ha="center",
    color=late_color
)


ax.text(
    g_e - .02,
    tt.f(g_e) - .15,
    r"$t_f^l(\gamma)$",
    color=late_color,
    ha="right"
)


ax.text(
    gs[0] - .01,
    tt.f(g_e) - .08,
    r"$t_i^l(\gamma)$",
    color=late_color,
    ha="right"
)

ax.axvspan(
    *bs,
    color=early_color,
    ec=None,
    alpha=.3,
    label="Critical Early Arrival Interval"
)
ax.axvspan(
    *gs,
    color=late_color,
    ec=None,
    alpha=.3,
    label="Critical Late Arrival Interval"
)

ax.legend(loc=2)
ax.xaxis.set_major_formatter(formatter)

ax.set_xlabel(r"$t^*$ (h)")
ax.set_ylabel("Travel Time")

ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])


ax.set_ylim(*ax.get_ylim())

ax.vlines(
    bs,
    ax.get_ylim()[0],
    tt.f(bs),
    linestyle="--",
    color=early_color
)

ax.vlines(
    gs,
    ax.get_ylim()[0],
    tt.f(gs),
    linestyle="--",
    color=late_color
)


fig.savefig("../img/early_late_int.png", dpi=quality)
plt.close()
# fig.show()
