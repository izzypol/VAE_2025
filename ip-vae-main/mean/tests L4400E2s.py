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
df = pd.read_csv('L4400E_2s.csv')
columns = [f"IP[{i}]" for i in range(20)]  # récupérer les donneés et les mettre dans une liste
data = df[columns].values # Limiter le nombre de training data, sinon le calcul sera trop long
ip_data = torch.from_numpy(data).float()
"""x = model.decode(torch.randn(zdim))
# Add synthetic noise to it
xn = x + 4*(torch.rand(20) - 0.5)"""

# Denoise decay with a forward pass
# test= np.random.randint(0, len(ip_data))  # Randomly select a test index
test = 541
print(f"Testing on index: {test}")
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
plt.title("comparison L4400E_2s")
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
plt.title("Statistiques L4400E_2s")
# plt.ylim([6, 29])
# plt.savefig("./figures/example-2.png", dpi=144, bbox_inches="tight")

# %timeit model.forward(xn)


diff = []
for i in range(len(ip_data)):
    xp3 = [model.forward(ip_data[i])[0] for _ in range(100)]
    xp3 = torch.stack(xp3)
    xp3_avg = torch.mean(xp3, dim=0)
    diff.append((xp3_avg.detach().numpy() - ip_data[i].detach().numpy()).mean())
plt.figure()
plt.hist(diff, bins=50)
plt.plot(stats.mean(diff), 0, 'ro', label="Overall mean difference")
plt.legend()
plt.xlabel("Mean difference (mV/V)")
plt.ylabel("Frequency")
plt.title("Histogramme L4400E_2s")
plt.show()
print(f"Mean difference: {stats.mean(diff)}")

diff2 = []
for i in range(len(ip_data)):
    xp2 = [model.forward(ip_data[i])[0] for _ in range(100)]
    xp2 = torch.stack(xp2)
    xp2_avg = torch.mean(xp2, dim=0)
    diff2.append((ip_data[i].detach().numpy() - xp2_avg.detach().numpy()))
diff2 = np.array(diff2)
diff2_avg = np.mean(diff2, axis=0)
plt.figure()
plt.plot(t, diff2_avg, 'ro')
plt.legend()
plt.xlabel("$t$ (s)")
plt.ylabel("Biais (mV/V)")
plt.title("Biais L4400E_2s")
plt.show()

diff4 = []
for i in range(len(ip_data)):
    xp4 = [model.forward(ip_data[i])[0] for _ in range(100)]
    xp4 = torch.stack(xp4)
    xp4_avg = torch.mean(xp4, dim=0)
    diff4.append(100 * (ip_data[i].detach().numpy() - xp4_avg.detach().numpy()) / ip_data[i].detach().numpy())
diff4 = np.array(diff4)
diff4_avg = np.mean(diff4, axis=0)
plt.figure()
plt.plot(t, diff4_avg, 'bo')
plt.legend()
plt.xlabel("$t$ (s)")
plt.ylabel("Biais (%)")
plt.title("Biais (%) L4400E_2s")
plt.show()
