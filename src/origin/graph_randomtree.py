
#
# 実験結果のグラフ化
# 
#

import os
import numpy as np
import argparse
import subprocess
import matplotlib.pyplot as plt


DESCRIPTION = 'a'
parser = argparse.ArgumentParser(description=DESCRIPTION)

# delta_lis = [1, 2, 3, 4, 5, 10, 20]
# bound = [144.66696406934403, 289.33392813868807, 434.0008922080321, 578.6678562773761, 723.3348203467202, 1446.6696406934404, 2893.339281386881]
# ave_Ratio = [4.824345381751842, 4.122301894122443, 3.039745824687465, 2.982971978268297, 2.551422035946463, 1.973518707858888, 1.6510727347918066]
# ave_D = [7.9336952108142995, 6.770268570304153, 5.74780846027448, 5.479748988304003, 5.38370736546785, 4.761570303430256, 4.290894082096392]
# ave_C = [6.586923325101619, 5.1973759512890085, 4.577833906958112, 4.1069497367429895, 3.8282249410759426, 3.3020574052485254, 3.0455814348105577]
# ave_UGapE = [14.238744921837354, 10.995608501451093, 10.21627669344671, 9.82336194819716, 9.578873026849191, 9.01435962515193, 8.766886262934701]
# ave_LUCB = [11.026362585692938, 9.811969229682589, 9.454958442880502, 9.24100058779966, 9.154067264211196, 8.837311602026748, 8.707755140255584]
# ave_RR = [33.174473736101554, 27.374476443023596, 25.181803992147984, 24.33995917901595, 23.65862435849151, 22.148574836091296, 21.433897296093722]

delta_lis = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
bound = [144.66696406934403, 289.33392813868807, 434.0008922080321, 578.6678562773761, 723.3348203467202, 868.0017844160642, 1157.3357125547523, 1446.6696406934404, 1736.0035688321284, 2025.3374969708161, 2314.6714251095045, 2604.0053532481925, 2893.339281386881]
ave_Ratio = [4.824345381751842, 4.122301894122443, 3.039745824687465, 2.982971978268297, 2.551422035946463, 2.3388099384677137, 2.1375491771001274, 1.973518707858888, 1.8968634967814584, 1.804283980060364, 1.7699718221588094, 1.7211516844269814, 1.6510727347918066]
ave_D = [7.9336952108142995, 6.770268570304153, 5.74780846027448, 5.479748988304003, 5.38370736546785, 5.120902606197149, 4.921110735819064, 4.761570303430256, 4.595519700093114, 4.443556796563708, 4.327633413250478, 4.318821228985426, 4.290894082096392]
ave_C = [6.586923325101619, 5.1973759512890085, 4.577833906958112, 4.1069497367429895, 3.8282249410759426, 3.7154992741974753, 3.4241984905589926, 3.3020574052485254, 3.2412137284863527, 3.1441040367465005, 3.1139360523530955, 3.0470249955909936, 3.0455814348105577]
ave_UGapE = [14.238744921837354, 10.995608501451093, 10.21627669344671, 9.82336194819716, 9.578873026849191, 9.375801462752598, 9.178584471873759, 9.01435962515193, 9.004358793176877, 8.891525697289495, 8.833835065394926, 8.774950163407796, 8.766886262934701]
ave_LUCB = [11.026362585692938, 9.811969229682589, 9.454958442880502, 9.24100058779966, 9.154067264211196, 9.045438086607101, 8.985995582079642, 8.837311602026748, 8.842784586167209, 8.799440698972457, 8.722469971756293, 8.739998622357197, 8.707755140255584]
ave_RR = [33.174473736101554, 27.374476443023596, 25.181803992147984, 24.33995917901595, 23.65862435849151, 23.063004661294936, 22.6148937737561, 22.148574836091296, 21.770039923031145, 21.662144736676535, 21.570766571172353, 21.513043561957918, 21.433897296093722]

alg = ['Ratio', 'C', 'D', 'UGapE', 'LUCB', 'RR']
answer = 'win'

ylim = [-0.3, 34]
# ylim = [0.9, 30]
yticks = [1, 5, 10, 15, 20, 25, 30]



parser.add_argument('-o', '--out', type=str, default="[[default]]", help='出力画像ファイル名')
parser.add_argument('--legend', action='store_true', help='legend')
parser.add_argument('--ylabel', action='store_true', help='ylabel')
args = parser.parse_args()

outfile = args.out
if args.out == "[[default]]":
    outfile = "fig_randomtree.png"

args.ylabel = True


plt.rcParams["font.size"] = 18
# fig = plt.figure(figsize=(7,5))
# fig = plt.figure(figsize=(7,8))
fig = plt.figure(figsize=(7.7,6.6))
ax = fig.add_subplot(1, 1, 1) # 行数,列数,位置


if 'Ratio' in alg:
    ax.plot(delta_lis, ave_Ratio, label="RD-Tracking-TMCTS")
else:
    ax.plot([0],[1], linewidth=0)

if 'C' in alg:
    ax.plot(delta_lis, ave_C, label="C-Tracking")
else: 
    ax.plot([0],[1], linewidth=0)

if 'D' in alg:
    ax.plot(delta_lis, ave_D, label="D-Tracking")
else: 
    ax.plot([0],[1], linewidth=0)

# if 'P' in alg:
#     ax.plot(delta_lis, ave_P, label="P-Tracking")
# else:
#     ax.plot([0],[1], linewidth=0)

if 'UGapE' in alg:
    ax.plot(delta_lis, ave_UGapE, label="UGapE-MCTS")
else:
    ax.plot([0],[1], linewidth=0)

if 'LUCB' in alg:
    ax.plot(delta_lis, ave_LUCB, label="LUCB-micro")
else:
    ax.plot([0],[1], linewidth=0)

if 'RR' in alg:
    ax.plot(delta_lis, ave_RR, label="Round Robin")
else:
    ax.plot([0],[1], linewidth=0)


# if 'R' in alg:
#     (avg_ratio, stddev_ratio) = parse_result('R', bound)
#     ax.fill_between(delta_lis, avg_ratio - stddev_ratio, avg_ratio + stddev_ratio, alpha = 0.2)
#     ax.plot(delta_lis, avg_ratio, label="Randomized D-Tracking")
# else:
#     ax.fill_between([0], [1], [1], alpha = 0.0)
#     ax.plot([0],[1], linewidth=0)


ax.plot(delta_lis, [1] * len(delta_lis), label="Lower bound", linestyle='dashed', color='black')

# ax.set_xticks(delta_lis)
# ax.set_xticklabels([f"1e-{x}" for x in delta_lis])

# ax.set_yscale('log')

delta_lis = [1,5,10,20]

ax.set_xticks(delta_lis)
ax.set_xticklabels([rf"$10^{{-{x}}}$" for x in delta_lis])
# ax.set_xticks(delta_lis[::2])
# ax.set_xticklabels([f"1e-{x}" for x in delta_lis[::2]])
# ax.set_xticks(delta_lis)
# ax.set_xticklabels([f"1e-{x}" for x in delta_lis])

ax.set_yticks(yticks)
ax.set_yticklabels([rf"$\times${x}" for x in yticks])
ax.set_ylim(ylim)

ax.set_xlabel('delta')

if(args.ylabel):
    ax.set_ylabel('average stopping time / lower bound')



# y軸目盛りに1が出るようにしたい
# yticks = ax.get_yticks()
# if 1 not in yticks:
#     yticks = list(yticks) + [1]
# for i,a in enumerate(yticks):
#     if a == 0:
#         yticks = yticks[:i] + yticks[i+1:]
#         break

# ax.set_yticks(sorted(yticks))



# title = f"Ratio of average stopping time to the lower bound\n"
# title += f"Depth = {depth}, Answer = '{answer}'"
# ax.set_title(title)



if(args.legend):
    # ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    # order = [6,0,1,2,4,3,5]
    order = [5,3,4,2,1,0,6]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="center left",
        bbox_to_anchor=(1.01, 0.265),
        frameon=False,
    )

plt.savefig(outfile, pad_inches=0.05, bbox_inches='tight')
print(f"-> {outfile}")


