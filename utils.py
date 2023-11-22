import copy
import pickle
import time
import os

from config import *

upoverhead = 0
downoverhead = 0
uptime = 0
downtime = 0


# 第index个客户端更新
def train(index):
    gradient_norm = 0
    l = 0
    gradient_no1 = 0
    for idx, (input, target) in enumerate(train_loader[index]):
        input = input.to(device)
        target = target.to(device)
        optimizer[index].zero_grad()
        output = models[index](input)  #1
        proximal_term = 0.0
        for w, w_t in zip(models[index].parameters(), models[-1].parameters()):
            proximal_term += (w- w_t).norm(2)  #1
        loss = criterion(output, target) + (MU / 2) * proximal_term
        l = loss.item()
        loss.backward()
        for name, param in models[index].named_parameters():
            # 累计历史梯度信息
            history_gradient[index][name] += param.grad  #1
        gradient_no = optimizer[index].step()
        if gradient_no >= gradient_norm:
            gradient_norm=gradient_no
        if index==9:
            gradient_no1 += gradient_no
    gradient_no1 = gradient_no1/len(train_loader[index])
    if PRINT:
        print(f"Client: {index} loss: {l}")
    # global_loss += l*(1/N_CLIENTS)
    return l, gradient_norm,gradient_no1

# 测试，默认-1代表全局模型的测试，输入客户端编号也可以测试对应客户端的精度
def pred(index=-1):

    with torch.no_grad():
        total = 0
        correct = 0
        for idx, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            output = models[index](input)  #2
            predict = output.argmax(1)
            correct += predict.eq(target).sum().item()
            total += len(input)
        print(f"Accuracy:{correct / total}")
        return correct/total


def clear_gradient():
    # 情况历史梯度信息
    for h in history_gradient:
        for k in h:
            init.zeros_(h[k])


def pushup_qcsgd():   #量化
    global upoverhead, uptime
    cur_overhead = 0
    with torch.no_grad():
        gradient = copy.deepcopy(models[-1].state_dict())
        for key in gradient:
            init.zeros_(gradient[key])
        for key in models[-1].state_dict():
            for idx, tmp in enumerate(history_gradient):
                # assert QUAN_UP, "需要量化"
                tmp = tmp[key]
                if tmp.dtype != torch.float32:
                    continue
                # 这是是逐层量化的
                if COMPENSATION:
                    tmp += compensation[idx][key]
                    pre = copy.deepcopy(tmp)
                    init.zeros_(compensation[idx][key])
                tmp, (k, b) = quantization(QUAN_BIT, tmp)  # 量化
                tmp = pickle.dumps(tmp)
                cur_overhead += len(tmp)  # 通信开销
                cur_overhead += 8
                tmp = pickle.loads(tmp)
                tmp = de_quantization(k, b, tmp)  # 去量化
                if COMPENSATION:
                    compensation[idx][key] += (pre - tmp)
                gradient[key] += tmp

        # 计算学习率
        gradient_norm = 0
        for key in gradient:
            gradient_norm += gradient[key].to(torch.float32).norm(2) ** 2
        gradient_norm = gradient_norm ** 0.5

        lr = min(GLOBALETA, N_CLIENTS * GLOBALGAMMA * GLOBALETA / gradient_norm)
        print(gradient_norm)
        print(lr)
        # 更新
        for key in models[-1].state_dict():
            models[-1].state_dict()[key] -= ((lr / N_CLIENTS) * gradient[key]).to(gradient[key].dtype)

        upoverhead += cur_overhead
        cur_time = cur_overhead / (CLIENTBANDWIDTH * 1024 * N_CLIENTS)
        uptime += cur_time
        for _ in range(N_CLIENTS):
            while random.random() < CLIENTERROR:
                # 丢包重传
                uptime += cur_time
        if TIME:
            time.sleep(cur_time)
    clear_gradient()
    return lr,gradient_norm

# def pushup_qcsgd():  #不量化
#     global upoverhead, uptime
#     cur_overhead = 0
#     with torch.no_grad():
#         gradient = copy.deepcopy(models[-1].state_dict())
#         for key in gradient:
#             init.zeros_(gradient[key])
#         for key in models[-1].state_dict():
#             for idx, tmp in enumerate(history_gradient):
#                 # assert QUAN_UP, "需要量化"
#                 tmp = tmp[key]
#                 if tmp.dtype != torch.float32:
#                     continue
#                 # 这是是逐层量化的
#                 if COMPENSATION:
#                     tmp += compensation[idx][key]
#                     pre = copy.deepcopy(tmp)
#                     init.zeros_(compensation[idx][key])
#                 # tmp, (k, b) = quantization(QUAN_BIT, tmp)  # 量化
#                 # tmp = pickle.dumps(tmp)
#                 # cur_overhead += len(tmp)  # 通信开销
#                 # cur_overhead += 8
#                 # tmp = pickle.loads(tmp)
#                 # tmp = de_quantization(k, b, tmp)  # 去量化
#                 if COMPENSATION:
#                     compensation[idx][key] += (pre - tmp)
#                 gradient[key] += tmp
#
#         # 计算学习率
#         gradient_norm = 0
#         for key in gradient:
#             gradient_norm += gradient[key].to(torch.float32).norm(2) ** 2
#         gradient_norm = gradient_norm ** 0.5
#
#         lr = min(GLOBALETA, N_CLIENTS * GLOBALGAMMA * GLOBALETA / gradient_norm)
#         print(gradient_norm)
#         print(lr)
#         # 更新
#         for key in models[-1].state_dict():
#             models[-1].state_dict()[key] -= ((lr / N_CLIENTS) * gradient[key]).to(gradient[key].dtype)
#
#         upoverhead += cur_overhead
#         cur_time = cur_overhead / (CLIENTBANDWIDTH * 1024 * N_CLIENTS)
#         uptime += cur_time
#         for _ in range(N_CLIENTS):
#             while random.random() < CLIENTERROR:
#                 # 丢包重传
#                 uptime += cur_time
#         if TIME:
#             time.sleep(cur_time)
#     clear_gradient()
#     return lr,gradient_norm


def pushup():
    # 聚合参数
    # 传参数变化量delta
    global upoverhead, uptime
    cur_overhead = 0
    with torch.no_grad():
        gradient = copy.deepcopy(models[-1].state_dict())
        for key in gradient:
            init.zeros_(gradient[key])
        # 平均参数
        for key in models[-1].state_dict():
            for idx in range(N_CLIENTS):
                # 传输参数变化量，本地多轮迭代导致取梯度需要累加比较麻烦
                delta = models[idx].state_dict()[key] - models[-1].state_dict()[key]
                if QUAN_UP:
                    # 这是是逐层量化的
                    if COMPENSATION:
                        delta += compensation[idx][key]
                        pre = copy.deepcopy(delta)
                        init.zeros_(compensation[idx][key])
                    delta, (k, b) = quantization(QUAN_BIT, delta)  # 量化
                    delta = pickle.dumps(delta)
                    cur_overhead += len(delta)  # 通信开销
                    cur_overhead += 8
                    delta = pickle.loads(delta)
                    delta = de_quantization(k, b, delta)  # 去量化
                    if COMPENSATION:
                        compensation[idx][key] += (pre - delta)
                elif SPARSE_UP:
                    # 这里是逐层稀疏化的
                    # 稀疏化的序列化存在问题
                    if COMPENSATION:
                        delta += compensation[idx][key]
                        pre = copy.deepcopy(delta)
                        init.zeros_(compensation[idx][key])
                    delta, mask = sparsification(delta, SPARSE_RATIO)
                    delta = pickle.dumps(delta)
                    mask = pickle.dumps(mask)
                    cur_overhead += len(delta)
                    cur_overhead += len(mask)
                    delta = pickle.loads(delta)
                    mask = pickle.loads(mask)
                    delta = de_sparsification(delta, mask).to(device)
                    if COMPENSATION:
                        compensation[idx][key] += (pre - delta)
                else:
                    delta = pickle.dumps(delta)
                    cur_overhead += len(delta)
                    delta = pickle.loads(delta)
                gradient[key] += (delta * (client_nums[idx] / total))
            models[-1].state_dict()[key] += gradient[key]
        upoverhead += cur_overhead
        cur_time = cur_overhead / (CLIENTBANDWIDTH * 1024 * N_CLIENTS)
        uptime += cur_time
        for _ in range(N_CLIENTS):
            while random.random() < CLIENTERROR:
                # 丢包重传
                uptime += cur_time
        if TIME:
            time.sleep(cur_time)


def pushdown():
    # 发送全局模型
    # 发布参数，传models[-1].state_dict()[key]
    global downoverhead, downtime
    cur_overhead = 0
    with torch.no_grad():
        for key in models[-1].state_dict():
            for idx in range(N_CLIENTS):
                init.zeros_(models[idx].state_dict()[key])
                tmp = models[-1].state_dict()[key]
                if QUAN_DOWN:
                    # 这里是逐层量化的
                    tmp, (k, b) = quantization(QUAN_BIT, tmp)  # 量化
                    tmp = pickle.dumps(tmp)
                    cur_overhead += len(tmp)
                    cur_overhead += 8
                    tmp = pickle.loads(tmp)
                    tmp = de_quantization(k, b, tmp)  # 去量化
                elif SPARSE_DOWN:
                    # 这里是逐层稀疏化的
                    # 下载时不建议稀疏化，因为传的是参数
                    # 稀疏化的序列化存在问题
                    tmp, mask = sparsification(tmp, SPARSE_RATIO)
                    tmp = pickle.dumps(tmp)
                    mask = pickle.dumps(mask)
                    cur_overhead += len(tmp)
                    cur_overhead += len(mask)
                    tmp = pickle.loads(tmp)
                    mask = pickle.loads(mask)
                    tmp = de_sparsification(tmp, mask).to(device)
                else:
                    tmp = pickle.dumps(tmp)
                    cur_overhead += len(tmp)
                    tmp = pickle.loads(tmp)
                models[idx].state_dict()[key] += tmp
        cur_time = cur_overhead / (SERVERBANDWIDTH * 1024)
        downoverhead += cur_overhead
        downtime += cur_time
        for _ in range(N_CLIENTS):
            while random.random() < SERVERERROR:
                downtime += cur_time / N_CLIENTS
        if TIME:
            time.sleep(cur_time)


def quantization(bit: int, x: torch.Tensor) -> (torch.Tensor, (int, int)):
    # 量化 均匀量化
    ans = copy.deepcopy(x)
    l, r = 0, (1 << bit) - 1
    mi, ma = x.min().item(), x.max().item()
    k = (r - l) / (ma - mi) if ma != mi else 1
    b = -mi * k if ma != mi else 0
    # 向下取整 floor
    # 向上取整 ceil
    # 四舍五入 round
    # 截取整数部分 trunc
    ans *= k
    ans += b
    ans = torch.trunc(ans)
    if QUAN_BIT == 8:
        ans = ans.to(torch.uint8)
    return ans, (k, b)


def de_quantization(k: int, b: int, x: torch.Tensor) -> torch.Tensor:
    # 去量化
    ans = copy.deepcopy(x)
    if QUAN_BIT == 8:
        ans = ans.to(torch.float32)
    ans -= b
    ans /= k
    return ans


def sparsification(x: torch.Tensor, s: float) -> (torch.Tensor, torch.Tensor):
    # Top-k稀疏化，掩码法
    with torch.no_grad():
        tmp = copy.deepcopy(abs(x.view(-1)))
        k = float(tmp.sort()[0][int(len(tmp) * s)])
        mask = torch.where(abs(x) > k, True, False)
        ans = torch.multiply(mask, x)
    ans = ans[ans != 0].view(-1)
    return ans, mask


def de_sparsification(x: torch.Tensor, mask: torch.Tensor):
    # 这里可能有点慢
    sz = mask.size()
    mask = mask.view(-1)
    ans = torch.zeros(mask.size())
    idx = 0
    for i, _ in enumerate(ans):
        if mask[i]:
            ans[i] = x[idx]
            idx += 1
    return ans.view(sz)


def main():
    trainloss_file = './trainloss' + args.model + '.txt'
    if (os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    pushdown()

    Noniid_norm_file = './Norm' + args.model + '.txt'
    if (os.path.isfile(Noniid_norm_file)):
        os.remove(Noniid_norm_file)
    f_norm = open(Noniid_norm_file, 'a')

    for i in range(EPOCH):
        gradient_global = 0
        G_loss = 0
        # 依次训练
        gradient_noniid = []
        for _ in range(LOCAL_E):
            for j in range(N_CLIENTS):
                g_loss, gradient_global_norm,G_Noniid=train(j)
                G_loss += g_loss
                if gradient_global_norm>=gradient_global:
                    gradient_global=gradient_global_norm
                if G_Noniid ==0:
                    continue
                else:
                    gradient_noniid.append(G_Noniid)
                    print(gradient_noniid)
                    f_norm.write(str(gradient_noniid[-1])+'\n')
        f_norm.flush()

        if args.optim == 'qcsgd':
            LR,Gradient_norm=pushup_qcsgd()
        else:
            pushup()  # 上传参数
        print(f"Epoch:{i + 1}", end="  ")
        acc = pred(-1)  # 发布全局参数
        global_loss = G_loss / (LOCAL_E*N_CLIENTS)
        f_trainloss.write(str(global_loss) + "\t" + str(acc) + "\t" + str(Gradient_norm) + "\t" + str(LR)+"\t" + str(gradient_global)+'\n')
        f_trainloss.flush()
        pushdown()
    global upoverhead, downoverhead, uptime, downtime
    print(f"Total communication overhead:{upoverhead + downoverhead} bytes")
    print(f"\tUp overhead {upoverhead} bytes")
    print(f"\tDown overhead {downoverhead} bytes")
    print(f"Total communication time:{round(uptime + downtime, 2)} seconds")  # 有点小问题，模拟的依次传输
    print(f"\tUp time {round(uptime, 2)} seconds")
    print(f"\tDown time {round(downtime, 2)} seconds")
