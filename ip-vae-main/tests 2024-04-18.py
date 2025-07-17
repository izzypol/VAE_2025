# @Author: cberube
# @Date:   2021-07-26 17:07:38
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   cberube
# @Last modified time: 2021-07-26 17:07:31


import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stats

import ipvae


zdim = 1
model = ipvae.Net(zdim=zdim)
model.load_weights()

# import data
df = pd.read_csv('2024-04-18_test.csv')
columns = [f"M{i} (mV/V)" for i in range(1, 21)]  # récupérer les donneés et les mettre dans une liste
data = df[columns].values # Limiter le nombre de training data, sinon le calcul sera trop long
ip_data = torch.from_numpy(data).float()
"""x = model.decode(torch.randn(zdim))
# Add synthetic noise to it
xn = x + 4*(torch.rand(20) - 0.5)"""

# Denoise decay with a forward pass
test= 87
output = model.forward(ip_data[test])
xp = output[0]  

# Plot comparison
t = np.arange(0.4+0.04, 2, 0.08)  # the IRIS ELREC Pro windows
plt.figure()
plt.plot(t, ip_data[test].detach().numpy(), '--k', label="Ground truth")
#plt.plot(t, xn.detach().numpy(), '.k', label="Noisy input")
plt.plot(t, xp.detach().numpy(), '-C3', label="Denoised")
plt.legend()
plt.ylabel("Chargeability (mV/V)")
plt.xlabel("$t$ (s)")
plt.title("comparison 2024-04-18")
# plt.ylim([6, 29])
# plt.savefig("./figures/example-1.png", dpi=144, bbox_inches="tight")

# Plot comparison
xp = [model.forward(ip_data[test])[0] for _ in range(100)]
xp = torch.stack(xp)
# Compute statistics
xp_avg = torch.mean(xp, dim=0)
xp_std = torch.std(xp, dim=0)

plt.figure()
plt.plot(t, ip_data[test].detach().numpy(), '--k', label="Ground truth")
# plt.plot(t, xn.detach().numpy(), '.k', label="Noisy input")
plt.plot(t, xp_avg.detach().numpy(), '-C3', label="Denoised")
plt.fill_between(t,
                 (xp_avg-2*xp_std).detach().numpy(),
                 (xp_avg+2*xp_std).detach().numpy(),
                 color='C3', alpha=0.2, label=r"$2\sigma$")
plt.legend()
plt.ylabel("Chargeability (mV/V)")
plt.xlabel("$t$ (s)")
plt.title("Statistiques 2024-04-18")
# plt.ylim([6, 29])
# plt.savefig("./figures/example-2.png", dpi=144, bbox_inches="tight")

# %timeit model.forward(xn)


diff = []
for i in range(1, 560):
    xp3 = [model.forward(ip_data[i])[0] for _ in range(100)]
    xp3 = torch.stack(xp3)
    xp3_avg = torch.mean(xp3, dim=0)
    diff.append((xp3_avg.detach().numpy() - ip_data[i].detach().numpy()).mean())
diff = torch.stack(diff)
plt.figure()
plt.hist(diff, bins=50)
plt.plot(stats.mean(diff), 0, 'ro', label="Mean difference")
plt.legend()
plt.title("Histogramme 2024-04-18")
plt.show()
print(f"Mean difference: {stats.mean(diff)}")

plt.figure()
plt.plot(t, diff)
plt.title("biais 2024-04-18")
plt.show()

