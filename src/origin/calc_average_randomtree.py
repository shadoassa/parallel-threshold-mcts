

from dataclasses import dataclass, field
import argparse
import numpy as np
import subprocess


algorithm = ['Ratio', 'D', 'C', 'UGapE', 'LUCB', 'RR']
delta = [1,2,3,4,5,6,8,10,12,14,16,18,20]
# delta = [2,3,4,5,10,20,30]
prob_id = list(range(0,500))


def get_average(alg, d):
  tau_sum = 0
  # max_tau = 0
  for p in prob_id:
    with open(f"../exp_randomtree/result_{alg}_{p}_delta{d}.txt", 'r') as f:
      lines = f.readlines()
    # try:
    #   line = lines[8]
    # except:
    #   print(f"{alg} {d} {p}")
    line = lines[8]

    tau = float(line.split()[1])
    tau_sum += tau
    # max_tau = max(max_tau, tau)

    if int(lines[6].split()[1]) != 10:
      print(f"detect incorrect result! {alg} {p} {d}")

  # print(max_tau)
  return (tau_sum / len(prob_id))

# for d in delta:
#   print(f"delta {d}")
#   for alg in algorithm:
#     ave = get_average(alg,d)
#     print(ave)
#   print()


# d-value計算 一回動かせばあとはコメントアウト
# with open("randomtree/d-value.txt", 'w') as f:
#   for i,p in enumerate(prob_id):
#     prob = f"randomtree/prob_{p}.txt"
#     result = subprocess.run(["python3", "calc_dvalue.py", "-i", prob], capture_output=True, text=True)
#     C_star = float(result.stdout.strip())
#     f.write(f"{C_star}\n")

#     if i % 20 == 0:
#       print(f"{i}/{len(prob_id)}")


C_star = 0
with open("randomtree/d-value.txt", 'r') as f:
  lines = f.readlines()
  if len(lines) != len(prob_id):
    print("error!")
  else:
    C_star = sum([float(x) for x in lines]) / len(lines)

print(C_star)
# C_star = 1


print(f"delta_lis = {delta}")
bound = [float(d*np.log(10)*C_star) for d in delta]
print(f"bound = {bound}")

for alg in algorithm:
  # print(f"alg: {alg}")
  tmp = []
  for i,d in enumerate(delta):
    ave = get_average(alg,d)
    tmp.append(ave / bound[i])
    # print(ave)
  print(f"ave_{alg} = {tmp}")
  # print()



