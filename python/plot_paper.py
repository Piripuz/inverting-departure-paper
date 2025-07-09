import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})
import matplotlib as mpl

import numpy as np

from scripts.travel_times import asymm_gaussian, asymm_gaussian_plateau
#%%

early_color = "green"
late_color="red"
ot_color = "blue"
tt_color = "purple"
cost_color = "orange"

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
