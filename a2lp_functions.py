import math
import torch
from tqdm import tqdm
import torch.nn.functional as F

def batchAccuracy(probs, labels):
    olabels = probs.argmax(dim=2).long()
    labels = labels.long()
    matches = labels.eq(olabels).float()
    acc_test = matches.mean(1)
    m = acc_test.mean().item()
    pm = acc_test.std().item() *1.96 / math.sqrt(probs.shape[0])
    return m, pm

def label_propagation_standard(opt, support, support_ys, query):

    # affinity matrix
    k = opt.K
    alpha = torch.Tensor([opt.alpha]).cuda()
    y_s = support_ys
    y_s = F.one_hot(y_s)
    y_q  = torch.zeros(query.shape[0], query.shape[1], y_s.size(2))

    fs, qs = F.normalize(torch.Tensor(support),p=2, dim=-1), F.normalize(torch.Tensor(query),p=2, dim=-1)
    l_all = torch.cat([y_s, y_q], dim=1).cuda()
    z_all = torch.cat([fs, qs], dim=1).cuda()
    n_label = support_ys.shape[1]

    A = torch.bmm(z_all, z_all.permute(0, 2, 1))
    A = torch.pow(A, 3)  # raise to power 3
    A = torch.clamp(A, min=0)  # non-negative
    N = A.size(1)  # N = number of examples per task
    A[:, range(N), range(N)] = 0  # zero diagonal

    # graph construction
    _, indices = torch.topk(A, k, dim=-1)
    print(A.shape, indices.shape)
    mask = torch.zeros_like(A)  # (tasks, N, N)
    mask = mask.scatter(-1, indices, 1)  # (tasks, N, N)
    A = A * mask

    W = (A + A.permute(0, 2, 1)) * 0.5  # (tasks, N, N)
    D = W.sum(-1)
    D_sqrt_inv = torch.sqrt(1.0 / (D))
    D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, N)
    D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N, 1)
    W = D1 * W * D2  # (tasks, N, N)
    # label propagation
    I = torch.eye(N)  # (N, N)
    I = I.unsqueeze(0).repeat(z_all.size(0), 1,1).cuda()  # (tasks,N, N)
    propagator = torch.inverse(I - alpha * W)  # (tasks, N, N)
    scores_all = torch.matmul(propagator, l_all)
    Z = scores_all[:, n_label:, :]
    probs_l1 = F.normalize(Z, 2, dim=2)
    probs_l1[probs_l1 < 0] = 0
    p_labels = torch.argmax(probs_l1[:,:,:opt.n_ways], 2)
    z_all = z_all
    support_features= z_all[:, :n_label, :]
    query_features = z_all[:, n_label:, :]
    return p_labels, probs_l1, Z, support_features, query_features

def ce_parallel(opt, support, support_ys, query):
    y_s = support_ys
    y_s = F.one_hot(y_s).float().cuda()
    n_tasks = support.size(0)
    counts = y_s.sum(1).view(n_tasks, -1, 1)
    support = support.cuda()
    query = query.cuda()

    weights = y_s.transpose(1, 2).matmul(support)
    weights = (weights / counts).cuda()
    weights.requires_grad_()
    optimizer = torch.optim.Adam([weights], lr=opt.lr)
    for i in tqdm(range(opt.init_ft_iter)):
        logits_s = 15 * (support.matmul(weights.transpose(1, 2)) - 1 / 2 * (weights**2).sum(2).view(n_tasks, 1, -1) - 1 / 2 * (support**2).sum(2).view(n_tasks, -1, 1))  #
        ce = - (y_s* torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
        loss = ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logits_q = 15 * (query.matmul(weights.transpose(1, 2)) - 1 / 2 * (weights**2).sum(2).view(n_tasks, 1, -1) - 1 / 2 * (query**2).sum(2).view(n_tasks, -1, 1))  #
    query_ys_pred = logits_q.argmax(2)

    return query_ys_pred, logits_q.softmax(2)#p_labels, probs_l1, Z, support_features, query_features


def prototypical(opt, support, support_ys, query):

    y_s = support_ys
    y_s = F.one_hot(y_s).float().cuda()
    print(y_s.dtype)
    n_tasks = support.size(0)
    counts = y_s.sum(1).view(n_tasks, -1, 1)
    support = support.cuda()
    query = query.cuda()

    weights = y_s.transpose(1, 2).matmul(support)
    weights = (weights / counts).cuda()
    logits_q = 15 * (query.matmul(weights.transpose(1, 2)) - 1 / 2 * (weights**2).sum(2).view(n_tasks, 1, -1) - 1 / 2 * (query**2).sum(2).view(n_tasks, -1, 1))  #
    query_ys_pred = logits_q.argmax(2)

    return query_ys_pred, logits_q.softmax(2)

def a2lp(opt, support, support_ys, query):

    # affinity matrix
    k = opt.K

    y_s = support_ys
    y_s = F.one_hot(y_s)
    y_q = torch.zeros(query.shape[0], query.shape[1], y_s.size(2))
    fs, qs = F.normalize(support,p=2, dim=-1, eps=1e-12), F.normalize(query,p=2, dim=-1, eps=1e-12)
    l_all = torch.cat([y_s, y_q], dim=1).cuda()

    z_all = torch.cat([fs, qs], dim=1).cuda()
    n_label = y_s.size(1)
    alpha = torch.Tensor([opt.alpha]).cuda()
    z_all = z_all.requires_grad_()
    optimizer = torch.optim.Adam([z_all], lr=opt.lr)
    for i in tqdm(range(opt.init_ft_iter)):

        A = torch.bmm(z_all, z_all.permute(0, 2, 1))  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        A = torch.pow(A, 3)  # power transform
        A = torch.clamp(A, min=0)  # non-negative
        N = A.size(1)  # N = n_way*n_sup+n_way*n_query
        A[:, range(N), range(N)] = 0  # zero diagonal
        # graph construction
        _, indices = torch.topk(A, k, dim=-1)  # (tasks, n_way*n_sup+n_way*n_query, topk)
        mask = torch.zeros_like(A)  # (tasks, N, N)
        mask = mask.scatter(-1, indices, 1)  # (tasks, N, N)
        A = A * mask  # (tasks, N, N)
        W = (A + A.permute(0, 2, 1)) * 0.5  # (tasks, N, N)
        D = W.sum(-1)  # (tasks, N)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-12))  # (tasks, N)
        D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N, 1)
        W = D1 * W * D2  # (tasks, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        # label propagation
        I = torch.eye(N)  # (N, N)
        I = I.unsqueeze(0).repeat(z_all.size(0), 1,1).cuda()  # (tasks, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        propagator = torch.inverse(I - alpha * W)  # (tasks, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        scores_all = torch.matmul(propagator, l_all)*15  # (tasks, n_way*n_sup+n_way*n_query, n_way)
        logits_s = scores_all[:, :n_label, :]

        ce = - (y_s.cuda() * torch.log(logits_s.softmax(2)+ 1e-12)).sum(2).mean(1).sum(0)
        loss = ce
        optimizer.zero_grad()
        loss.backward()
        #------ONLY supp change------------------
        z_all.grad.data[:, n_label:, :].fill_(0)
        optimizer.step()
    #-------------identify the labels of the neighbours of each---------------------------------------------------------
    Z = scores_all[:, n_label:, :]#
    probs_l1 = F.normalize(Z, 2, dim=-1)
    probs_l1[probs_l1 < 0] = 0
    p_labels = torch.argmax(probs_l1[:,:,:opt.n_ways], 2)[:,n_label:]
    z_all = z_all
    support_features= z_all[:, :n_label, :]
    query_features = z_all[:, n_label:, :]
    return p_labels, probs_l1, Z, support_features, query_features














