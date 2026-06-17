import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ==================================================
# 1) nem : number of ensemble members (x-axis : log-scale)
nem = np.array([10, 20, 30, 50, 80, 100])

# Parse results from L96enkf.log

# DA Time(s)
t_letkf = [
    6.93950343132019,
    9.958414554595947,
    13.438926696777344,
    19.391934156417847,
    29.91693115234375,
    38.340431451797485
]

# Background RMSE
rmse_b_letkf = [
    2.6494549213119125,
    1.7618139447931722,
    1.4016066991645755,
    1.230697532025626,
    1.0063118349438667,
    0.5743284382106907
]

# Background spread
sprd_b_letkf = [
    0.8213967288826916,
    0.9498832849850821,
    0.9869485692188444,
    0.8975199215355074,
    0.7741475635737056,
    0.5200511334065346
]

# =========================================================================== #

t_mletkf_rloc = [
    35.72122931480408, 54.23299264907837, 99.77745270729065,
    46.8080153465271, 73.23809790611267, 92.18383693695068  
]

rmse_b_mletkf_rloc = [
    1.3581411469633995, 1.0362178422499715, 0.8696608508100862, 
    0.6656060962311332, 0.6219043379692353, 0.5581841580485579 
]

sprd_b_mletkf_rloc = [
    0.8695316368612664, 0.9695438089988051, 1.0083433032090214, 
    0.7500984433947948, 0.6348042286534455, 0.5580388028534159  
]

# =========================================================================== #

t_mletkf_zloc = [
   21.422680139541626, 32.32253336906433, 34.24592590332031, 
    43.97148370742798, 133.28632736206055, 141.81333565711975
]

rmse_b_mletkf_zloc = [
    3.5368345429000176, 3.0826082289034717, 2.4713741565711915, 
    1.8704394916147726, 1.0945082612201824, 0.5680810915476491  
]

sprd_b_mletkf_zloc = [
    0.5965476118198022, 0.5705154355186214, 0.5027488614865527, 
    0.5292561847277025, 0.5532651998088741, 0.5636678671145718  
]

# =========================================================================== #

t_mletkf_wo_rloc = [
    16.13016414642334, 19.336653470993042, 53.98885107040405, 
    41.02464318275452, 60.11511468887329, 47.82136297225952  
]

rmse_b_mletkf_wo_rloc = [
    4.804887708499121, 4.537318916032758, 4.45089057158296, 
    4.193778599086306, 3.8471249620297425, 3.1942496220596928 
]

sprd_b_mletkf_wo_rloc = [
    0.22223644691497824, 0.2913389228793065, 0.3278052767939679,  
    0.3720358806813944, 0.41121761902764176, 0.43080937675761277
]

# =========================================================================== #

t_getkf = [
    34.026477575302124, 64.65779781341553, 113.41520071029663, 
    104.4977958202362, 247.96972274780273, 382.874165058136    
]

rmse_b_getkf = [
    2.782490923018534, 1.614588049736404, 1.3584323322114056,  
    0.79826546176756, 1.0386426270054578, 0.7987902252293141   
]

sprd_b_getkf = [
    0.7403311128073566, 0.8007524684445978, 0.9088387453989002,  
    0.7613110756733616, 0.8921656138904294, 0.9086143219760873 
]

# =========================================================================== #

colors = ["tab:pink", "tab:blue", "tab:orange", "tab:green", "tab:red"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# FIG. 8(a) : Background RMSE =========================
ax1.plot(nem, rmse_b_letkf, marker='o', label='LETKF', color=colors[0])
ax1.plot(nem, rmse_b_getkf, marker='o', label='GETKF', color=colors[1])
ax1.plot(nem, rmse_b_mletkf_wo_rloc, marker='o', label='MLETKF-noLoc', color=colors[2])
ax1.plot(nem, rmse_b_mletkf_rloc, marker='o', label='MLETKF', color=colors[3])
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
ax2.plot(nem, sprd_b_mletkf_wo_rloc, marker='o', label='MLETKF-noLoc', color=colors[2])
ax2.plot(nem, sprd_b_mletkf_rloc, marker='o', label='MLETKF', color=colors[3])
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
plt.plot(nem, t_mletkf_wo_rloc, marker='o', label='MLETKF-noLoc', color=colors[2])
plt.plot(nem, t_mletkf_rloc, marker='o', label='MLETKF', color=colors[3])
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
    ("MLETKF-noLoc", rmse_b_mletkf_wo_rloc, sprd_b_mletkf_wo_rloc),
    ("MLETKF", rmse_b_mletkf_rloc, sprd_b_mletkf_rloc),
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
axes[4].legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.88),)

axes[3].set_xlabel("Ensemble size (k)")
axes[4].set_xlabel("Ensemble size (k)")

axes[0].set_ylabel("RMSE, Spread")
axes[3].set_ylabel("RMSE, Spread")

plt.tight_layout()
plt.show()

# FIG. 3, 4, 5 : RMSE and spread timestep plots  ==============================
def plot_overlay_rmse_b(
    npz_paths,
    labels=None,
    step_min=None,
    step_max=None,
    save=False,
    fname="overlay_rmse.png"
):
    if labels is None:
        labels = [os.path.basename(p).replace(".npz", "") for p in npz_paths]

    # filter label color
    colors = ["tab:pink", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(12, 3.2))

    for i, (path, lab) in enumerate(zip(npz_paths, labels)):
        d = np.load(path)

        step = d["step"].astype(int)
        rmse_b = d["rmse_b"].astype(float)

        mask = np.ones_like(step, dtype=bool)
        if step_min is not None:
            mask &= (step >= step_min)
        if step_max is not None:
            mask &= (step <= step_max)

        plt.plot(
            step[mask],
            rmse_b[mask],
            label=lab,
            linewidth=1.2,
            color=colors[i % len(colors)]
        )

    plt.xlabel("Time steps")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save:
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[Saved] {fname}")

    plt.show()


# =========================
# settings
# =========================
num = 3    # CASE number

paths = [
    f"./CASE{num}_LETKF_Ne50_rmse_b.npz",
    f"./CASE{num}_GETKF_Ne50_rmse_b.npz",
    f"./CASE{num}_MLETKF_noR_Ne50_rmse_b.npz",
    f"./CASE{num}_MLETKF_R_Ne50_rmse_b.npz",
    f"./CASE{num}_MLETKF_Z_Ne50_rmse_b.npz",
]

labels = [
    "LETKF",
    "GETKF",
    "MLETKF-noR",
    "MLETKF-R",
    "MLETKF-Z"
]

# =========================
# plot
# =========================
plot_overlay_rmse_b(
    npz_paths=paths,
    labels=labels,
    step_min=3000,
    step_max=4000,
    save=True,
    fname=f"CASE{num}_overlay_rmse_3000_4000.png"
)