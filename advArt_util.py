import torch
from torchvision import transforms
import torch.nn.functional as F

def smoothness(patch):
    # Compute L_tv
    # tvcomp1 = torch.sum(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]+0.000001),0)
    # tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
    # tvcomp2 = torch.sum(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]+0.000001),0)
    # tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
    tvdiff1 = torch.abs(patch[:, :, 1:] - patch[:, :, :-1])
    tvcomp1 = torch.linalg.norm(tvdiff1, ord=2, dim=0)
    tvcomp1 = torch.sum(tvcomp1)
    tvdiff2 = torch.abs(patch[:, 1:, :] - patch[:, :-1, :])
    tvcomp2 = torch.linalg.norm(tvdiff2, ord=2, dim=0)
    tvcomp2 = torch.sum(tvcomp2)
    tv = tvcomp1 + tvcomp2
    return tv/torch.numel(patch)


def similiar(patch, target):
    # Compute L_sim
    sim = torch.abs(torch.flatten(patch, 1) - torch.flatten(target, 1)) 
    sim = torch.linalg.norm(sim, ord=2, dim=0)
    L_sim = torch.mean(sim)*torch.mean(sim)
    return L_sim


def detect_loss(probabilities, labels, piecewise=None):
    # print(probabilities)
    result = torch.zeros(len(probabilities))
    for i in range(len(probabilities)):
        lab = labels[i][:,0]
        # lab = lab[lab==0]
        current = probabilities[i][:lab.shape[0]]
        result[i] = torch.mean(current)
    if piecewise is not None:
        result = torch.where((result<piecewise), 0, probabilities)
    L_det = torch.mean(result)
    return L_det


def combine(img, patch_t, mask):
    # Combine the patch to the image
    patch_mask = patch_t * mask
    advs = torch.unbind(patch_mask, 1)
    for adv in advs:
        img = torch.where((adv == 0), img, adv)
    return img


def perspective(patch):
    persp = transforms.RandomPerspective(p=1)
    patch_t = persp(patch)
    return patch_t


def wrinkles(patch):
    C, H, W = patch.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1).cuda()
    yy = torch.arange(0, H).view(-1,1).repeat(1,W).cuda()
    xx = xx.view(1,H,W)
    yy = yy.view(1,H,W)
    grid = torch.cat((xx,yy),0).float().cuda()  # torch.Size([2, H, W])
    # print("grid "+str(grid.shape)+" : \n"+str(grid))
    grid = grid.view(2,-1)  # torch.Size([2, H*W])
    grid = grid.permute(1,0)  # torch.Size([H*W, 2])
    perturbed_mesh = grid
    for k in range(15):
        # Choosing one vertex randomly
        vidx = torch.randint(grid.shape[0],(1,))
        vtex = grid[vidx, :]
        # Vector between all vertices and the selected one
        xv  = perturbed_mesh - vtex
        # Random movement 
        mv = (torch.rand(1,2).cuda() - 0.5)*50
        hxv = torch.zeros(xv.size(0), xv.size(1)+1).cuda()
        hxv[:, :-1] = xv
        mv_r3 = torch.cat((mv, torch.tensor([[0]]).cuda()), 1)
        hmv = torch.tile(mv_r3, (xv.shape[0],1)).cuda()
        d = torch.cross(hxv, hmv)
        d = torch.abs(d[:, 2])
        wt = d / torch.linalg.norm(mv, ord=2)
        wt = 1 - wt**2/(W**2 + H**2)
        msmv = mv * wt.unsqueeze(1)
        perturbed_mesh = perturbed_mesh + msmv

    perturbed_mesh_2 = perturbed_mesh.permute(1,0)
    max_x = torch.max(perturbed_mesh_2[0])
    min_x = torch.min(perturbed_mesh_2[0])
    # print("max_x : "+str(max_x)+" / min_x : "+str(min_x)
    max_y = torch.max(perturbed_mesh_2[1])
    min_y = torch.min(perturbed_mesh_2[1])
    # print("max_y : "+str(max_y)+" / min_y : "+str(min_y))
    perturbed_mesh_2[0,:] = (W-1)*(perturbed_mesh_2[0,:]-min_x)/(max_x-min_x)
    perturbed_mesh_2[1,:] = (H-1)*(perturbed_mesh_2[1,:]-min_y)/(max_y-min_y)
    perturbed_mesh_2 = perturbed_mesh_2.view(-1, H, W).float()

    vgrid = perturbed_mesh_2.unsqueeze(0).cuda()
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/(W-1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/(H-1)-1.0
    vgrid = vgrid.permute(0,2,3,1).cuda()
    patch = patch.unsqueeze(0).cuda()
    patch_t = F.grid_sample(patch, vgrid, align_corners=True)  # torch.Size([1, 3, H, W])
    patch_t = patch_t.squeeze(0)
    return patch_t


def rotate(patch, targetResize):
    angle = torch.cuda.FloatTensor(1).uniform_(-10, 10).item()
    result = transforms.functional.rotate(patch, angle, expand=True)
    resize = transforms.Resize((targetResize, targetResize))
    result = resize(result)
    return result


def noise(patch):
    brightness = torch.cuda.FloatTensor(1).uniform_(-0.10, 0.10)
    contrast = torch.cuda.FloatTensor(1).uniform_(0.8, 1.2)
    noise = torch.cuda.FloatTensor(patch.size()).uniform_(-0.1, 0.1)
    result = patch * contrast + brightness + noise
    result.data = torch.clamp(result.data, min=0, max=1)
    return result


def NPS(patch, colorspace):
    # Non-printability score, colorspace should be a (3, n) tensor containing n pixels
    patch = patch.unsqueeze(-1)
    patch = patch.expand(-1, -1, -1, colorspace.size(-1))
    colorspace = colorspace.unsqueeze(1)
    colorspace = colorspace.unsqueeze(1)
    colorspace = colorspace.expand(-1, patch.size(1), patch.size(2), -1)
    diff = torch.abs(patch - colorspace)
    diff = torch.sum(diff, dim=0)
    minimum = torch.min(diff, dim=2).values
    result = torch.sum(minimum)
    return result

def blur(patch):
    filter = torch.zeros(3, 3, 3, 3).cuda()
    avg_filter = torch.ones(3,3) / 9
    for i in range(3):
        filter[i,i] = avg_filter
    result = F.conv2d(patch, filter, padding="same")
    return result
