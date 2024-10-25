import torch

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.chamfer_distance.chamfer_distance_confidence import ChamferDistanceConfidence
from extensions.earth_movers_distance.emd import EarthMoverDistance
from extensions.earth_movers_distance.emd_confidence import EarthMoverDistanceConfidence


CD = ChamferDistance()
CD_Confidence = ChamferDistanceConfidence()
EMD = EarthMoverDistanceConfidence()
EMD_Vanilla = EarthMoverDistance()

def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0






def cd_loss_L1_confidence_old(pcs1, pcs2, gamma):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 4)
        pcs2 (torch.tensor): (B, M, 3)
    """
    confidence_sig = torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]

    pcs1 = torch.cat([pcs1[:,:,:3], confidence], dim=2)

    dist1, dist2 = CD_Confidence(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    dists = (torch.mean(dist1)  + confidence_loss + torch.mean(dist2)  + confidence_loss) + confidence_loss / 2.0

    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    confidence_loss = torch.mean(torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    #confidence_loss = torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)

    



    dists = 10*dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('CD LOSSES',dists)
    print('Gamma Loss', torch.mean(confidence_loss))
    print('Distance Loss', (torch.mean(dist1) + torch.mean(dist2)) / 2.0)

    return dists






def cd_loss_L1_confidence_diff(pcs1, pcs2, gamma):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 4)
        pcs2 (torch.tensor): (B, M, 3)
    """
    confidence_sig = pcs1[:,:,3]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]

    #confidence_sig_2 = pcs1[:,:,4]


    pcs1 = torch.cat([pcs1[:,:,:3], confidence], dim=2)

    

    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    #confidence_loss = torch.mean(torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    confidence_loss = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2))
    #confidence_loss_2 = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_2),confidence_sig_2).pow(2))
    #print('Confidence Loss', confidence_loss)

    dist1, dist2 = CD_Confidence(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    print('---Dist---',torch.mean(dist1))
    print('----Confidence-----', torch.mean(confidence_loss))
    dists = torch.mean(dist2) + torch.mean(confidence_loss)# + confidence_loss_2)#(torch.mean(dist1 + confidence_loss)  +  torch.mean(dist2 + confidence_loss)) / 2.0  



    #dists = 10*dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('CD LOSSES',dists)
    print('Gamma Loss', torch.mean(confidence_loss))#, torch.mean(confidence_loss_2))
    print('Distance Loss', dists)#(torch.mean(dist1) + torch.mean(dist2)) / 2.0)

    return 10*dists




def cd_loss_L1_confidence_try(pcs1, pcs2, gamma):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 4)
        pcs2 (torch.tensor): (B, M, 3)
    """
    confidence_sig = pcs1[:,:,3]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]

    #confidence_sig_2 = pcs1[:,:,4]


    pcs1_pts = pcs1[:,:,:3]#torch.cat([pcs1[:,:,:3], confidence], dim=2)

    

    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    #confidence_loss = torch.mean(torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    confidence_loss = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2))
    #confidence_loss_2 = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_2),confidence_sig_2).pow(2))
    #print('Confidence Loss', confidence_loss)

    dist1, dist2 = CD(pcs1_pts, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    print('---Dist---',torch.mean(dist1))
    print('----Confidence-----', torch.mean(confidence_loss))
    dists = torch.mean(dist2) + torch.mean(confidence_loss)# + confidence_loss_2)#(torch.mean(dist1 + confidence_loss)  +  torch.mean(dist2 + confidence_loss)) / 2.0  



    #dists = 10*dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('CD LOSSES',dists)
    print('Gamma Loss', torch.mean(confidence_loss))#, torch.mean(confidence_loss_2))
    print('Distance Loss', dists)#(torch.mean(dist1) + torch.mean(dist2)) / 2.0)

    return 10*dists


def cd_loss_L1_confidence(pcs1, pcs2, gamma):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 4)
        pcs2 (torch.tensor): (B, M, 3)
    """
    confidence_sig_x = pcs1[:,:,3]#torch.nn.ReLU(pcs1[:,:,3])
    #confidence_x = torch.nn.ReLU(confidence_sig_x[:, :, None])
    #confidence_x = torch.nn.functional.sigmoid(confidence_sig_x[:, :, None])
    confidence_x = confidence_sig_x[:, :, None]
    #confidence_x = torch.nn.ReLU(confidence_sig_x)


    # confidence_sig_y = pcs1[:,:,4]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    # #confidence_y =  torch.nn.ReLU(confidence_sig_y[:, :, None])
    # confidence_y =  confidence_sig_y[:, :, None]

    # confidence_sig_z  = pcs1[:,:,5]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    # #confidence_z =  torch.nn.ReLU(confidence_sig_z[:, :, None])
    # confidence_z =  confidence_sig_z[:, :, None]
    # #confidence_sig_2 = pcs1[:,:,4]



    pcs1_pts = torch.cat([pcs1[:,:,:3], confidence_x], dim = 2)#, confidence_y, confidence_z], dim=2)

    

    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    #confidence_loss = torch.mean(torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    confidence_loss_x = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_x),confidence_x).pow(2))
    #confidence_loss_y = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_y),confidence_sig_y).pow(2))
    #confidence_loss_z = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_z),confidence_sig_z).pow(2))

    #confidence_loss_2 = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_2),confidence_sig_2).pow(2))
    #print('Confidence Loss', confidence_loss)
    confidence_loss = (confidence_loss_x )#+ confidence_loss_y + confidence_loss_z)
    dist1, dist2 = CD_Confidence(pcs1_pts, pcs2)
    #dist1, dist2 = CD(pcs1[:,:,:3], pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    print('---Dist---',dist2, torch.mean(dist2))
    print('----Confidence-----', torch.mean(confidence_loss))
    
    dists = (torch.mean(dist1) + torch.mean(dist2))/2.# + confidence_loss)# + confidence_loss_2)#(torch.mean(dist1 + confidence_loss)  +  torch.mean(dist2 + confidence_loss)) / 2.0  



    #dists = 10*dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('CD LOSSES',dists)
    print('Gamma Loss', torch.mean(confidence_loss))#, torch.mean(confidence_loss_2))
    print('Distance Loss', dists)#(torch.mean(dist1) + torch.mean(dist2)) / 2.0)

    return 10*dists




def cd_loss_L1_confidence_pred(pcs1, gamma):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 4)
        pcs2 (torch.tensor): (B, M, 3)
    """
    confidence_sig_x = pcs1[:,:,3]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    #confidence_x = torch.nn.ReLU(confidence_sig_x[:, :, None])
    confidence_x = confidence_sig_x[:, :, None]


    confidence_sig_y = pcs1[:,:,4]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    #confidence_y =  torch.nn.ReLU(confidence_sig_y[:, :, None])
    confidence_y =  confidence_sig_y[:, :, None]

    confidence_sig_z  = pcs1[:,:,5]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    #confidence_z =  torch.nn.ReLU(confidence_sig_z[:, :, None])
    confidence_z =  confidence_sig_z[:, :, None]


    #pcs1_pts = torch.cat([pcs1[:,:,:3], confidence_x, confidence_y, confidence_z], dim=2)

    

    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    #confidence_loss = torch.mean(torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    confidence_loss_x = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_x),confidence_sig_x).pow(2))
    confidence_loss_y = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_y),confidence_sig_y).pow(2))
    confidence_loss_z = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_z),confidence_sig_z).pow(2))

    #confidence_loss_2 = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_2),confidence_sig_2).pow(2))
    #print('Confidence Loss', confidence_loss)
    confidence_loss = (confidence_loss_x + confidence_loss_y + confidence_loss_z)/3.
    
    '''
    dist1, dist2 = CD_Confidence(pcs1_pts, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    print('---Dist---',torch.mean(dist1))
    print('----Confidence-----', torch.mean(confidence_loss))
    
    dists = torch.mean(dist2 + confidence_loss)# + confidence_loss_2)#(torch.mean(dist1 + confidence_loss)  +  torch.mean(dist2 + confidence_loss)) / 2.0  



    #dists = 10*dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('CD LOSSES',dists)
    print('Gamma Loss', torch.mean(confidence_loss))#, torch.mean(confidence_loss_2))
    print('Distance Loss', dists)#(torch.mean(dist1) + torch.mean(dist2)) / 2.0)
    '''

    return 10*torch.mean(confidence_loss)



def cd_loss_L1_confidence_diff_works(pcs1, pcs2, gamma):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 4)
        pcs2 (torch.tensor): (B, M, 3)
    """
    confidence_sig = pcs1[:,:,3]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]

    #confidence_sig_2 = pcs1[:,:,4]


    pcs1_pts = torch.cat([pcs1[:,:,:3], confidence], dim=2)

    

    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    #confidence_loss = torch.mean(torch.mean(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    confidence_loss = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2))
    #confidence_loss_2 = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig_2),confidence_sig_2).pow(2))
    #print('Confidence Loss', confidence_loss)

    dist1, dist2 = CD_Confidence(pcs1_pts, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    print('---Dist---',torch.mean(dist1))
    print('----Confidence-----', torch.mean(confidence_loss))
    dists = torch.mean(dist2 + confidence_loss)# + confidence_loss_2)#(torch.mean(dist1 + confidence_loss)  +  torch.mean(dist2 + confidence_loss)) / 2.0  



    #dists = 10*dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('CD LOSSES',dists)
    print('Gamma Loss', torch.mean(confidence_loss))#, torch.mean(confidence_loss_2))
    print('Distance Loss', dists)#(torch.mean(dist1) + torch.mean(dist2)) / 2.0)

    return 10*dists


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss_one_hot_match(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists, match = EMD(pcs1, pcs2)

    fine_size = list(pcs2.size())
    
    match = torch.nn.functional.softmax(match, dim=2)
    
    
    a = match.argmax (2)
    print('idx array',a.size(), a.type())
    match_one_hot = torch.nn.functional.one_hot(a, num_classes = fine_size[1] )
    print('one hot array', match_one_hot.size(), match_one_hot.type())
    print('gt array', pcs2.size(), pcs2.type())
    match_one_hot = torch.tensor(match_one_hot, dtype=torch.float)
    return torch.mean(dists), torch.matmul(match_one_hot, pcs2)
    
    #return torch.mean(dists), torch.matmul(match, pcs2)



def emd_loss_original(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists, match = EMD_Vanilla(pcs1, pcs2)

    #fine_size = list(pcs2.size())
    
    #match = torch.nn.functional.softmax(match, dim=2)
    
    
    #a = match.argmax (2)
    #print('idx array',a.size(), a.type())
    #match_one_hot = torch.nn.functional.one_hot(a, num_classes = fine_size[1] )
    #print('one hot array', match_one_hot.size(), match_one_hot.type())
    #print('gt array', pcs2.size(), pcs2.type())
    #match_one_hot = torch.tensor(match_one_hot, dtype=torch.float)
    print('COARSE EMD LOSS', torch.mean(dists))
    return torch.mean(dists), match #torch.matmul(match_one_hot, pcs2)



def emd_loss(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists, match = EMD(pcs1, pcs2)
    confidence = pcs1[:,:,3]
    print(confidence, confidence.size())#, pred.size())
    #confidence_raw = confidence 
    #fine_size = list(pcs2.size())
    
    #match = torch.nn.functional.softmax(match, dim=2)
    
    
    #a = match.argmax (2)
    #print('idx array',a.size(), a.type())
    #match_one_hot = torch.nn.functional.one_hot(a, num_classes = fine_size[1] )
    #print('one hot array', match_one_hot.size(), match_one_hot.type())
    #print('gt array', pcs2.size(), pcs2.type())
    #match_one_hot = torch.tensor(match_one_hot, dtype=torch.float)
    dists = dists #+ torch.mean(torch.mul((torch.subtract(torch.ones_like(confidence),confidence)).pow(2),0.01))

    return torch.mean(dists), match




def emd_confidence_loss(pcs1, pcs2, gamma):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    confidence_sig = pcs1[:,:,3]#torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]

    pcs1 = torch.cat([pcs1[:,:,:3], confidence], dim=2)
    dists, match = EMD(pcs1, pcs2)
    print(dists.size())
    
    #print(confidence.size())
    #confidence_loss = torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)) 
    #confidence_loss = torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1)
    confidence_loss = torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2))
    #confidence_loss = torch.sum(torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2),1) 
    print(confidence_loss.size(), dists.size())
    dists = torch.mean(dists)# + torch.mean(confidence_loss,1))#dists + #torch.mul(gamma,confidence_loss)#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('LOSSES',dists.size(), confidence_loss.size(), torch.mean(dists), torch.mean(confidence_loss))
    return dists, match

  


def emd_confidence_loss_pick(pcs1, pcs2, gamma):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    confidence_sig = torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]

    #pcs1 = torch.cat([pcs1[:,:,:3], confidence], dim=2)
    dists, match = EMD_Vanilla(pcs1[:,:,:3], pcs2)
    
    #print(confidence.size())
    #confidence_loss = torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig),confidence_sig).pow(2)),1))
    dists_pick = torch.mul(dists, confidence_sig)
    dists = dists_pick + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('LOSSES',torch.mean(dists), confidence_loss)
    return torch.mean(dists), match






def emd_confidence_loss_old(pcs1, pcs2, confidence, gamma):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """

    #dists, match = EMD(pcs1, pcs2)
    gt = pcs2
    pred = pcs1
    confidence = confidence[:,:,None]
    print(confidence.size(), pred.size())
    confidence_raw = confidence 
    confidence  = confidence.expand(-1, -1, 3)
    
    print(confidence.size(), pred.size())
    #print(pred_confidence.size())
    #pred_confidence = torch.add(torch.mul(confidence,pred),torch.mul(confidence,pred))# torch.matmul(torch.subtract(torch.ones_like(confidence),confidence), gt))
    pred_confidence = torch.add(torch.mul(confidence,pred), torch.mul(torch.subtract(torch.ones_like(confidence),confidence), gt)) 
    #print(confidence.size(), pred.size())
    #print(pred_confidence.size())

    dists = (gt - pred_confidence).pow(2).sum(-1).sqrt() + torch.mul(torch.subtract(torch.ones_like(confidence_raw),confidence_raw).sum(-1),gamma)
    #dists = torch.mean(torch.subtract(gt, pred).pow(2).sum(-1).sqrt()) 
    #mseloss = torch.nn.MSELoss()
    #dists = mseloss(pcs2, pcs1).sqrt()
    print('dists',dists.size())
    return torch.mean(dists)







# emd_loss_confidence = distance(GT, confidence*Pred + (1-confidence)*GT) + alpha*(|1 - confidence|) 
# pred_uncertainty = confidence*Pred + (1-confidence)*GT

def emd_loss_2(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists, match = EMD(pcs1, pcs2)
    #confidence = pcs1[:,:,3]
    #print(confidence.size())#, pred.size())
    #confidence_raw = confidence 
    #fine_size = list(pcs2.size())
    
    #match = torch.nn.functional.softmax(match, dim=2)
    
    
    #a = match.argmax (2)
    #print('idx array',a.size(), a.type())
    #match_one_hot = torch.nn.functional.one_hot(a, num_classes = fine_size[1] )
    #print('one hot array', match_one_hot.size(), match_one_hot.type())
    #print('gt array', pcs2.size(), pcs2.type())
    #match_one_hot = torch.tensor(match_one_hot, dtype=torch.float)
    dists = dists #+ torch.mean(torch.mul((torch.subtract(torch.ones_like(confidence),confidence)).pow(2),0.01))

    return torch.mean(dists), match



def emd_loss_original_2(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists, match = EMD_Vanilla(pcs1, pcs2)
    dists = dists.to('cuda:1')
    match = match.to('cuda:1')

    #fine_size = list(pcs2.size())
    
    #match = torch.nn.functional.softmax(match, dim=2)
    
    
    #a = match.argmax (2)
    #print('idx array',a.size(), a.type())
    #match_one_hot = torch.nn.functional.one_hot(a, num_classes = fine_size[1] )
    #print('one hot array', match_one_hot.size(), match_one_hot.type())
    #print('gt array', pcs2.size(), pcs2.type())
    #match_one_hot = torch.tensor(match_one_hot, dtype=torch.float)
    print('COARSE EMD LOSS', torch.mean(dists))

    return torch.mean(dists), match #torch.matmul(match_one_hot, pcs2)



def emd_confidence_loss_2(pcs1, pcs2, gamma):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """

    pcs1_conf = pcs1[:,:,3]
    pcs1_conf = pcs1_conf.to('cuda:1')
    confidence_sig = torch.nn.functional.sigmoid(pcs1_conf)
    confidence_sig = confidence_sig.to('cuda:1')
    confidence = confidence_sig[:, :, None]
    confidence = confidence.to('cuda:1')

    pcs1_xyz = pcs1[:,:,:3]
    pcs1_xyz = pcs1_xyz.to('cuda:1')
    pcs1_sig = torch.cat([pcs1_xyz, confidence], dim=2)
    pcs1_sig = pcs1_sig.to('cuda:1')
    pcs2 = pcs2.to('cuda:1')
    dists, match = EMD(pcs1_sig, pcs2)
    dists = dists.to('cuda:1')
    match = match.to('cuda:1')
    
    #print(confidence.size())
    #confidence_loss = torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))


    tensor_ones = torch.ones_like(confidence_sig)
    tensor_ones = tensor_ones.to('cuda:1')


    sub_step = torch.subtract(tensor_ones,confidence_sig)
    sub_step = sub_step.to('cuda:1')
    pow_step = sub_step.pow(2)
    pow_step = pow_step.to('cuda:1')
    mul_step = torch.mul(gamma,pow_step)
    mul_step = mul_step.to('cuda:1')
    sum_step = torch.sum(mul_step,1)
    sum_step = sum_step.to('cuda:1')
    confidence_loss = torch.mean(sum_step)
    confidence_loss = confidence_loss.to('cuda:1')



    #confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(tensor_ones,confidence_sig).pow(2)),1))

    #confidence_loss = confidence_loss.to('cuda:1') 
    dists = dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    dists = dists.to('cuda:1')
    print('LOSSES',torch.mean(dists), confidence_loss)
    return torch.mean(dists), match



def emd_confidence_loss_2_old(pcs1, pcs2, gamma):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    confidence_sig = torch.nn.functional.sigmoid(pcs1[:,:,3].to('cuda:1')).to('cuda:1')
    confidence = confidence_sig[:, :, None]
    confidence = confidence.to('cuda:1')

    pcs1 = torch.cat([pcs1[:,:,:3].to('cuda:1'), confidence], dim=2)
    dists, match = EMD(pcs1.to('cuda:1'), pcs2)
    dists = dists.to('cuda:1')
    
    #print(confidence.size())
    #confidence_loss = torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    confidence_loss = torch.mean(torch.sum(torch.mul(gamma,torch.subtract(torch.ones_like(confidence_sig).to('cuda:1'),confidence_sig).pow(2)),1))
    
    confidence_loss = confidence_loss.to('cuda:1') 
    dists = dists + confidence_loss#torch.mul(gamma,torch.mean((torch.subtract(torch.ones_like(confidence),confidence)).pow(2)))
    print('LOSSES',torch.mean(dists), confidence_loss)
    return torch.mean(dists.to('cuda:1')), match


