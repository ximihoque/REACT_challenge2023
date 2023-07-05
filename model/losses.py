import torch
import torch.nn as nn
import torch.nn.functional as F



class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, output1, output2, label):

        # Calculate the contrastive loss
        cosine_sim = self.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cosine_sim, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_sim, min=0.0), 2))
        return loss_contrastive

class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002, contrastive=None):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.kl_loss = KLLoss()
        self.kl_p = kl_p
        self.contrastive = contrastive

    def forward(self, gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, distribution, **kwargs):

        # emt_loss = self.bce(pred_emotion[:, :, :15].view(-1, 15), gt_emotion[:, :, :15].view(-1, 15)) + self.mse(pred_emotion[:, :, 15:], gt_emotion[:, :, 15:])
        # _3dmm_loss = self.mse(pred_3dmm[:,:, :52], gt_3dmm[:,:, :52]) + 10*self.mse(pred_3dmm[:,:, 52:], gt_3dmm[:,:, 52:])
        # rec_loss = emt_loss + _3dmm_loss
        rec_loss = self.mse(pred_emotion, gt_emotion) + self.mse(pred_3dmm[:,:, :52], gt_3dmm[:,:, :52]) + 10*self.mse(pred_3dmm[:,:, 52:], gt_3dmm[:,:, 52:])
        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        # loss = 0
        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss

        if self.contrastive:
            contrastive_loss_pos = self.contrastive(kwargs['spk'], kwargs['list_pos'], 1)
            contrastive_loss_neg = self.contrastive(kwargs['spk'], kwargs['list_neg'], 0)
            contrastive_loss = contrastive_loss_pos + contrastive_loss_neg
            loss += contrastive_loss

            return loss, rec_loss, kld_loss, contrastive_loss
        else:
            return loss, rec_loss, kld_loss


    def __repr__(self):
        return "VAELoss()"



def div_loss(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss




# ================================ BeLFUSION losses ====================================

def MSELoss_AE_v2(prediction, target, target_coefficients, mu, logvar, coefficients_3dmm, 
                  w_mse=1, w_kld=1, w_coeff=1, 
                  **kwargs):
    # loss for autoencoder. prediction and target have shape of [batch_size, seq_length, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, seq_length, features]"
    batch_size = prediction.shape[0]

    # join last two dimensions of prediction and target
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    coefficients_3dmm = coefficients_3dmm.reshape(coefficients_3dmm.shape[0], -1)
    target_coefficients = target_coefficients.reshape(target_coefficients.shape[0], -1)

    MSE = ((prediction - target) ** 2).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    COEFF = ((coefficients_3dmm - target_coefficients) ** 2).mean()

    loss_r = w_mse * MSE + w_kld * KLD + w_coeff * COEFF
    return {"loss": loss_r, "mse": MSE, "kld": KLD, "coeff": COEFF}


def MSELoss(prediction, target, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1)
    
    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def L1Loss(prediction, target, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(axis=-1)
    
    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def BeLFusionLoss(prediction, target, encoded_prediction, encoded_target, 
                  losses = [L1Loss, MSELoss], 
                  losses_multipliers = [1, 1], 
                  losses_decoded = [False, True], 
                  **kwargs):
    # encoded_prediction has shape of [batch_size, num_preds, features]
    # encoded_target has shape of [batch_size, num_preds, features]
    # prediction has shape of [batch_size, num_preds, seq_len, features]
    # target has shape of [batch_size, num_preds, seq_len, features]
    assert len(losses) == len(losses_multipliers), "losses and losses_multipliers must have the same length"
    assert len(losses) == len(losses_decoded), "losses and losses_decoded must have the same length"
    #assert len(encoded_prediction.shape) == 3 and len(prediction.shape) == 4, "BeLFusionLoss only works with multiple predictions"

    if len(encoded_prediction.shape) == 2 and len(prediction.shape) == 3: # --> for the test script to work, only a single pred is used
        # unsqueeze the first dimension, because we only have one prediction
        prediction = prediction.unsqueeze(1)
        target = target.unsqueeze(1)
        encoded_prediction = encoded_prediction.unsqueeze(1)
        encoded_target = encoded_target.unsqueeze(1)

    if len(encoded_prediction.shape) == 3 and len(prediction.shape) == 4:
        # join last two dimensions of prediction and target
        prediction = prediction.reshape(prediction.shape[0], prediction.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)
    else:
        raise NotImplementedError("BeLFusionLoss only works with multiple predictions")

    # compute losses
    losses_dict = {"loss": 0}
    for loss_name, w, decoded in zip(losses, losses_multipliers, losses_decoded):
        loss_final_name = loss_name + f"_{'decoded' if decoded else 'encoded'}"
        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction, target, reduction="min")
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, reduction="min")
    
        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict

