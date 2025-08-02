from init import *

tt = TravelTime(asymm_gaussian())
beta_0 = .5
beta_1 = .8

bs_0 = find_bs(beta_0, tt)
bs_1 = find_bs(beta_1, tt)

#%%

fig, ax = plt.subplots(figsize=(6, 4))

x = np.linspace(7.5, 10, 200)
ax.plot(x, tt.f(x), color=tt_color, label="Travel Time Function $tt(t)$")

ax.plot(
    bs_0,
    tt.f(bs_0),
    color="limegreen"
)
ax.plot(
    bs_1,
    tt.f(bs_1),
    color="darkgreen"
)
ax.axvspan(
    *bs_0,
    color="limegreen",
    alpha=.5,
    ec=None,
    label=rf"Critical Early Arrival Interval, $\beta = {beta_0}$"
)
ax.axvspan(
    *bs_1,
    color="darkgreen",
    alpha=.5,
    ec=None,
    label=rf"Critical Early Arrival Interval, $\beta = {beta_1}$"
)

ax.set_yticks([])
ax.set_xlabel("Time (h)")
ax.xaxis.set_major_formatter(formatter)
ax.legend()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.savefig("../img/decreasing_int.png", dpi=quality)
plt.close()
