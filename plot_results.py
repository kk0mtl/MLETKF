import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# ==================================================
# 1) nem : number of ensemble members (x-axis : log-scale)
nem = np.array([10, 20, 30, 50, 80, 100])

t_letkf = [
    3.4027907848358154, 4.502838373184204,  5.799935817718506,
    9.733905553817749, 15.498368978500366, 19.21369743347168,
]

rmse_b_letkf = [
    2.6494549213119125, 1.7618139447931722, 1.4016066991645755,
    1.230697532025626, 1.0063118349438667, 0.5743284382106907
]

sprd_b_letkf = [
    0.8213967288826916, 0.9498832849850821, 0.9869485692188444,
    0.8975199215355074, 0.7741475635737056, 0.5200511334065346
]
# =========================================================================== #
t_getkf = [
    32.45762395858765, 68.19534158706665, 125.0853898525238,
    44.01957654953003,  304.5166506767273, 518.1329946517944,
]

rmse_b_getkf = [
    2.782490923018534, 1.614588049736404, 1.3584323322114056,  
    0.7275704986147863, 1.0386426270054578, 0.7987902252293141   
]

sprd_b_getkf = [
    0.7403311128073566, 0.8007524684445978, 0.9088387453989002,  
    0.5923087373174114, 0.8921656138904294, 0.9086143219760873 
]
# =========================================================================== #
t_mletkf = [
    35.01349925994873, 33.54001593589783, 93.82191395759583,
    38.04286431387995, 61.46668481826782, 83.69127321243286,
]

rmse_b_mletkf = [
    1.3581411469633995, 1.0362178422499715, 0.8696608508100862, 
    0.6656060962311332, 0.6219043379692353, 0.5581841580485579 
]

sprd_b_mletkf = [
    0.8695316368612664, 0.9695438089988051, 1.0083433032090214, 
    0.7500984433947948, 0.6348042286534455, 0.5580388028534159  
]
# =========================================================================== #
t_mletkf_zloc = [
    25.924729108810425, 28.706499338150024, 39.04722356796265, 
    39.26127910614014,  140.52109265327454, 86.80725312232971, 
]

rmse_b_mletkf_zloc = [
    3.7844353792370238, 3.106745388289416, 2.0316552718833334,
    0.971026217119939, 0.8183470146269569, 0.581919849000509,
]

sprd_b_mletkf_zloc = [
    0.5585474088539627, 0.5736724386797758,  0.600168925418721,
    0.5851050932941461, 0.594156653127153,  0.6094164334870952,
]
# =========================================================================== #
t_mletkf_no_loc = [
    35.01349925994873, 33.54001593589783, 93.82191395759583,
    39.99648565155684, 61.46668481826782, 83.69127321243286,
]

rmse_b_mletkf_no_loc = [
    4.804887708499121, 4.537318916032758, 4.45089057158296, 
    4.193778599086306, 3.8471249620297425, 3.1942496220596928 
]

sprd_b_mletkf_no_loc = [
    0.22223644691497824, 0.2913389228793065, 0.3278052767939679,  
    0.3720358806813944, 0.41121761902764176, 0.43080937675761277
]


colors = ["tab:pink", "tab:blue", "tab:orange", "tab:green", "tab:red"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# FIG. 8(a) : Background RMSE =========================
ax1.plot(nem, rmse_b_letkf, marker='o', label='LETKF', color=colors[0])
ax1.plot(nem, rmse_b_getkf, marker='o', label='GETKF', color=colors[1])
ax1.plot(nem, rmse_b_mletkf_no_loc, marker='o', label='MLETKF-noLoc', color=colors[2])
ax1.plot(nem, rmse_b_mletkf, marker='o', label='MLETKF', color=colors[3])
ax1.plot(nem, rmse_b_mletkf_zloc, marker='o', label='MLETKF-Z', color=colors[4])

ax1.set_ylabel("RMSE")
ax1.legend(loc="upper right",
    fontsize=8.5,      
    markerscale=0.8,    
    handlelength=1.2,    
    labelspacing=0.4,  
    borderpad=0.3,
    bbox_to_anchor=(1.0, 0.60),
    frameon=True,
    facecolor='white',
    framealpha=0.8,
    edgecolor='none')

# FIG. 8(b) : Background spread =========================
ax2.plot(nem, sprd_b_letkf, marker='o', label='LETKF', color=colors[0])
ax2.plot(nem, sprd_b_getkf, marker='o', label='GETKF', color=colors[1])
ax2.plot(nem, sprd_b_mletkf_no_loc, marker='o', label='MLETKF-noLoc', color=colors[2])
ax2.plot(nem, sprd_b_mletkf, marker='o', label='MLETKF', color=colors[3])
ax2.plot(nem, sprd_b_mletkf_zloc, marker='o', label='MLETKF-Z', color=colors[4])

ax2.set_ylabel("Spread")
ax2.set_xlabel("Ensemble size")

# x-axis
ax1.set_xscale('log')   #log scale
ax1.set_xticks(nem)
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.ticklabel_format(style='plain', axis='x')
ax1.minorticks_off()

plt.tight_layout()
plt.show()

# FIG. 10 : Time plot =========================
plt.figure(figsize=(8, 6))

plt.plot(nem, t_letkf, marker='o', label='LETKF', color=colors[0])
plt.plot(nem, t_getkf, marker='o', label='GETKF', color=colors[1])
plt.plot(nem, t_mletkf_no_loc, marker='o', label='MLETKF-noLoc', color=colors[2])
plt.plot(nem, t_mletkf, marker='o', label='MLETKF', color=colors[3])
plt.plot(nem, t_mletkf_zloc, marker='o', label='MLETKF-Z', color=colors[4])

plt.xscale('log')
plt.xticks(nem, labels=[str(n) for n in nem])
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.ticklabel_format(style='plain', axis='x')
plt.minorticks_off()

plt.xlabel("Ensemble size (k)")
plt.ylabel("Time (s)")
plt.title("DA Time by ensemble size")

plt.legend(loc="upper left",
    frameon=True,
    facecolor='white',
    framealpha=0.8,
    edgecolor='none')
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.show()


# FIG. 9 : RMSE-spread consistency plots  ==============================
filters = [
    ("LETKF", rmse_b_letkf, sprd_b_letkf),
    ("GETKF", rmse_b_getkf, sprd_b_getkf),
    ("MLETKF-noLoc", rmse_b_mletkf_no_loc, sprd_b_mletkf_no_loc),
    ("MLETKF", rmse_b_mletkf, sprd_b_mletkf),
    ("MLETKF-Z", rmse_b_mletkf_zloc, sprd_b_mletkf_zloc),
]

fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True)
axes = axes.flatten()

# plot
for i, (name, rmse, sprd) in enumerate(filters):
    ax = axes[i]

    ax.plot(nem, rmse, marker='o', label="RMSE")
    ax.plot(nem, sprd, marker='s', linestyle='--', label="Spread")

    ax.set_xscale('log')
    ax.set_xticks(nem)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.minorticks_off()

    ax.set_title(name)
    ax.grid(True, alpha=0.3)

axes[5].set_visible(False)
axes[4].legend(loc="upper right", frameon=False, bbox_to_anchor=(0.9, 0.88),)

axes[3].set_xlabel("Ensemble size (k)")
axes[4].set_xlabel("Ensemble size (k)")

axes[0].set_ylabel("RMSE, Spread")
axes[3].set_ylabel("RMSE, Spread")

plt.tight_layout()
plt.show()