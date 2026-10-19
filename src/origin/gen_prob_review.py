

#
# 引数: 最大深さ D, 最大分岐数 K, 子ノード所持確率 p, 根ノードの評価値 V
# 
# 報酬空間は[0,1]とする。
# 根ノードの評価値が指定された値になるような、深さDのK分木をランダムに生成する。
# 深さが偶数の層はMAXノード、奇数の層はMINノードとする。
# 
# 具体的には、以下の手順で根から再帰的に木を生成する:
# ・根ノードの「目標評価値」に引数Vを代入し、自身をキューに入れる。
# ・キューが空になるまで以下を行う:
# 　・MAXノードの場合、[0, (自身の目標評価値)]の範囲から一様にランダムな値を抽出し、これを子ノードの目標評価値に設定、子ノードをキューに追加する。
# 　　これをK回繰り替えす。
# 　　ただし、子ノードのうち一つは必ず(子ノードの目標評価値) = (自身の目標評価値)とする。
# 　・MINノードの場合、[(自身の目標評価値), 1]の範囲から抽出し、同様に子を生成・キューに入れる。
# 　・葉ノードなら、自身の期待報酬に与えられた目標評価値を代入する。
# 
# 
# [変更]
# winの場合を考える。
# bestな行動の評価値を引数V、
# second bestな行動の評価値をthetaとする。
# 
# これで、best arm手法とthreshold手法との差を減らす
# 
# 

from dataclasses import dataclass, field

import argparse
import numpy as np
# seed = 12345
# rng = np.random.default_rng(seed)
rng = np.random.default_rng()


# 引数解析
DESCRIPTION = '問題ファイルを生成する。\n'
DESCRIPTION += '生成される木は深さDのK分木\n'
parser = argparse.ArgumentParser(description=DESCRIPTION)

parser.add_argument('-d', type=int, default=2, help='木の深さD')
parser.add_argument('-k', type=int, default=3, help='各ノードの持つ子の数K (K分木)')
parser.add_argument('-p', type=float, default=0.5, help='各ノードの持つ子の数K (K分木)')
parser.add_argument('-v', '--value', type=float, default=0.6, help='根ノードの価値V')
parser.add_argument('-t', '--theta', type=float, default=0.5, help='threshold theta')
parser.add_argument('-o', '--out', type=str, default="prob.txt", help='出力ファイル名', metavar='outfile')
parser.add_argument('--silent', action='store_true', help='標準出力をオフにするフラグ')
parser.add_argument('--non_BAI', action='store_true', help='指定した場合、second_bestをthetaと一致させない')


args = parser.parse_args()

K = args.k
D = args.d
root_value = args.value
theta = args.theta
prob = args.p

true_answer = 'win' if root_value >= 0.5 else 'lose'
# print(true_answer)



@dataclass
class expand_target:
  id: int = -1
  depth: int = -1
  value: float = -1.0
  parent: int = -1



@dataclass
class treeNode:
  id: int = -1
  depth: int = -1
  type: int = -1 # MIN = 0, MAX = 1
  value: float = 0.0
  parent: int = -1
  child: list[int] = field(default_factory=list)



# 幅優先探索で木を構築
queue = [] 

next_id = 1 # 次に作られるノードのid。root=0が作成済みなので1

tree = [] # treeNodeのベクトル


# 根ノードの実体を生成 根ノードだけ処理が特殊
node = treeNode()
node.id = 0
node.depth = 0
node.type = 1 
node.value = root_value
node.parent = -1
node.child = []
# rootだけは100%子を生成する
K_tmp = np.random.randint(2, K)
for i in range(K_tmp):
  if args.non_BAI == False:
    if i == 0: # 子のうち一つは親ノードと同じ評価値にする
      child_value = node.value
    elif i == 1: # second bestはthetaと同じにする
      child_value = theta
    else: # 残りはtheta以下にする
      child_value = rng.uniform(0, theta)
  else:
    if i == 0: # 子のうち一つは親ノードと同じ評価値にする
      child_value = node.value
    else: # 残りはnode.value以下にする
      child_value = rng.uniform(0, node.value)

  child = expand_target()
  child.id = next_id
  child.depth = node.depth + 1
  child.value = child_value
  child.parent = node.id
  next_id += 1
  node.child.append(child.id)
  queue.append(child)
tree.append(node)

# print(queue)


while True:  
  if len(queue) == 0:
    break
  
  target = queue[0]
  queue = queue[1:]

  # ノード生成
  node = treeNode()
  node.id = target.id
  node.depth = target.depth
  node.type = 1 if (node.depth%2 == 0) else 0 # 根はMAXで交互
  # node.type = 0 if (node.depth%2 == 0) else 1 # 根はMINで交互
  # node.type = 0 # 全部MIN
  node.value = target.value
  node.parent = target.parent
  node.child = []

  # 葉ノードでないなら子を生成
  if (node.depth < D) and (np.random.random() < prob):
    K_tmp = np.random.randint(2, K)
    for i in range(K_tmp):
      if i == 0: # 子のうち一つは親ノードと同じ評価値にする
        child_value = node.value
      elif i == 1 and node.depth == 0:
        pass
      elif node.type == 0: # MIN
        child_value = rng.uniform(node.value, 1)
      else: # MAX
        child_value = rng.uniform(0, node.value)

      child = expand_target()
      child.id = next_id
      child.depth = node.depth + 1
      child.value = child_value
      child.parent = node.id

      next_id += 1
      
      node.child.append(child.id)
      queue.append(child)
  else: # 葉ノードのとき
    pass
  tree.append(node)

# for t in tree:
#   print(t)


with open(args.out, 'w') as f:
  f.write("[size]\n")
  f.write(f"{next_id}\n")
  f.write("[answer]\n")
  f.write(f"{theta}\n")
  f.write(f"{true_answer}\n")
  f.write("[nodes]\n")
  for n in tree:
    tmp = f"{n.id} {n.depth} "
    # tmp += "true " if n.depth == D else "false "
    tmp += "true " if len(n.child) == 0 else "false "
    tmp += "MAX " if n.type == 1 else "MIN "
    tmp += f"{n.value} {n.parent} {len(n.child)}\n"
    for c in n.child:
      tmp += f"{c} "
    f.write(tmp + '\n')

  f.write("end\n")


# node_id depth is_leaf type value parent_id num_child
# child_id ...

if args.silent == False:
  print(f"-> {args.out}")
