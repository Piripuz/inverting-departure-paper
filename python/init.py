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

early_color = "green"
late_color="red"
ot_color = "blue"
tt_color = "purple"
cost_color = "orange"

f_form = lambda x, _: mpl.dates.num2date(x/24).strftime("%H:%M")
formatter = mpl.ticker.FuncFormatter(f_form)

quality = 300

