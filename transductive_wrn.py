import numpy as np
import torch
from test_arguments import parse_option
import a2lp_functions as af
import scipy as sp
from scipy.stats import t

use_gpu = torch.cuda.is_available()


def centerDatas(datas):
    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
    datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    return datas

def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def trans_a2lp(opt, X, Y, labelled_samples):
    support_features, query_features =  X[:,:labelled_samples], X[:,labelled_samples:]
    support_ys, query_ys = Y[:,:labelled_samples], Y[:,labelled_samples:]
    #------------------------------------------------------A2LP-------------------------------------------------------------
    query_ys_pred, probs, weights, support_features, query_features= af.a2lp(opt, support_features, support_ys, query_features)
    acc, std = af.batchAccuracy(probs, query_ys.cuda())
    return acc, std

def trans_lp_standard(opt, X, Y, labelled_samples):
    support_features, query_features =  X[:,:labelled_samples], X[:,labelled_samples:]
    support_ys, query_ys = Y[:,:labelled_samples], Y[:,labelled_samples:]
    #------------------------------------------------------LP-------------------------------------------------------------
    query_ys_pred, probs, weights, support_features, query_features= af.label_propagation_standard(opt, support_features, support_ys, query_features)
    acc, std = af.batchAccuracy(probs, query_ys.cuda())
    return acc, std

def trans_imprint_ce(opt, X, Y, labelled_samples):
    support_features, query_features =  X[:,:labelled_samples], X[:,labelled_samples:]
    support_ys, query_ys = Y[:,:labelled_samples], Y[:,labelled_samples:]
    #------------------------------------------------------IMPRINT+LCE-------------------------------------------------------------
    query_ys_pred, probs = af.ce_parallel(opt, support_features, support_ys, query_features)
    acc, std = af.batchAccuracy(probs, query_ys.cuda())
    return acc, std

def trans_prototypical(opt, X, Y, labelled_samples):
    support_features, query_features =  X[:,:labelled_samples], X[:,labelled_samples:]
    support_ys, query_ys = Y[:,:labelled_samples], Y[:,labelled_samples:]
    #------------------------------------------------------PROTOTYPICAL-------------------------------------------------------------
    query_ys_pred, probs = af.prototypical(opt, support_features, support_ys, query_features)
    acc, std = af.batchAccuracy(probs, query_ys.cuda())
    return acc, std

if __name__ == '__main__':
# ---- data loading
    params = parse_option()
    n_shot = params.n_shots
    n_ways = params.n_ways
    n_queries = params.n_queries
    n_runs=1000

    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    dataset = params.dataset

    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    print(ndatas.shape)
    ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shot+n_queries,5).clone().view(n_runs, n_samples)
    if params.preprocessing=='PLC':
        # Power transform
        beta = 0.5
        #------------------------------------PLC-----------------------------------------------
        nve_idx = np.where(ndatas.cpu().detach().numpy()<0)
        ndatas[nve_idx] *= -1
        ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
        ndatas[nve_idx]*=-1 # return the sign
        ndatas = scaleEachUnitaryDatas(ndatas)
        ndatas = centerDatas(ndatas)
        #------------------------------------------------------------------------------------------
    else:
        ndatas = scaleEachUnitaryDatas(ndatas)

    print(ndatas.type())
    n_nfeat = ndatas.size(2)

    n_nfeat = ndatas.size(2)

    print("size of the datas...", ndatas.size())

    if params.algorithm == 'A2LP':
        acc_mine, acc_std = trans_a2lp(params, ndatas, labels, n_lsamples)
        print('DATASET: {}, final accuracy A2LP: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset, acc_mine * 100,acc_std * 100, n_shot,n_queries))
    elif params.algorithm == 'imprint+ce':
        acc_mine, acc_std = trans_imprint_ce(params, ndatas, labels, n_lsamples)
        print('DATASET: {}, final accuracy imprint+ce: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset, acc_mine * 100, acc_std * 100, n_shot, n_queries))
    elif params.algorithm == 'prototypical':
        acc_mine, acc_std = trans_prototypical(params, ndatas, labels, n_lsamples)
        print('DATASET: {}, final accuracy prototypical: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset, acc_mine * 100,acc_std * 100,n_shot, n_queries))
    elif params.algorithm == 'LP':
        acc_mine, acc_std = trans_lp_standard(params, ndatas, labels, n_lsamples)
        print('DATASET: {}, final accuracy LP: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset, acc_mine * 100,acc_std * 100, n_shot,n_queries))
    else:
        print('Algorithm not supported!')
    

