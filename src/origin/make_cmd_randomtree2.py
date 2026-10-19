


# import argparse
# DESCRIPTION = 'a'
# parser = argparse.ArgumentParser(description=DESCRIPTION)
# parser.add_argument('-d', '--depth', type=int, default=2, help='depth')
# parser.add_argument('-d', '--depth', type=int, default=2, help='depth')
# parser.add_argument('-a', '--answer', type=str, default="win", help='answer')
# parser.add_argument('-o', '--out', type=str, default="[[default]]", help='出力画像ファイル名')
# args = parser.parse_args()


exp = 2
# prob_lis = [1,2,3,4,5,6,7,8,9,10]
# prob_lis = list(range(0, 500))
prob_lis = list(range(0, 3))
# prob_lis = [11,12,13,14,15,16,17,18,19,20]

# delta_lis = [1,3,5,10,20,30] # 1e-x
# delta_lis = [1,2,3,4,5,6,8,10,12,14,16,18,20] # 1e-x
delta_lis = [4] # 1e-x

algorithm = ['Ratio', 'C', 'D', 'UGapE', 'LUCB', 'RR']
# algorithm = ['Ratio', 'C', 'D', 'P', 'UGapE', 'LUCB', 'RR']

n_itr = 10

out_prefix = "result"
out2_prefix = "result2"
enable_out2 = False
enable_out2 = True


mkdir_set = set()
for alg in algorithm:
    for delta in delta_lis:
        for p in prob_lis:
            prob = f"../prob_randomtree/prob_{p}.txt"
            out = "dummy.txt"

            cmd = f"nohup ./experiment "
            cmd += f"-p {prob} "
            cmd += f"-o {out} "
            cmd += f"-d 1e-{delta} "
            cmd += f"-a {alg} "
            cmd += f"-i {n_itr} "
            cmd += f"-e {exp} > /dev/null&"
            print(cmd)

for c in mkdir_set:
    print(c)