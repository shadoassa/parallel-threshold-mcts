
# ランダムな木をたくさん生成するためのコマンドを作る

# number of problems(trees)
N = 500

# tree setting
d = 5
k = 5
p = 0.5
value = 0.6
theta = 0.5

outdir = "../prob_randomtree"


for i in range(N):
    probname = f"{outdir}/prob_{i}.txt"
    cmd = f"python3 gen_prob_review.py -d {d} -k {k} -p {p} --value {value} --theta {theta} --out {probname}"
    print(cmd)

# print(f"mkdir {outdir}")

