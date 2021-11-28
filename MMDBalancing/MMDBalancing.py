import torch

class MMDBalancing():
    def __init__(self, kernel = None, k_XX = None, k_XY = None, k_YY = None,):
        self.k = kernel
        self.k_XX = k_XX
        self.k_XY = k_XY
        self.k_YY = k_YY
        
    def GD(self, source, target,
           weights,
           source_weights = None,
           target_weights = None,
           learning_rate = 0.001,
           lambda_l2 = 0.):
                
        with torch.no_grad():
            n = len(source)
            m = len(target)
            # generate kernel matrix
            if self.k_XX == None:
                self.k_XX = torch.zeros((n,n))
                self.k_XY = torch.zeros((n,m))
                self.k_YY = torch.zeros((m,m))
                if self.k == None:
                    self.k = lambda x,y: torch.exp(-torch.linalg.norm(x-y)**2/2.0)
                print("Calculating kernel matrices...")
                for i in range(n):
                    for j in range(m):
                        self.k_XY[i,j] = self.k(source[i],target[j])
                for i in range(n):
                    for j in range(n):
                        self.k_XX[i,j] = self.k(source[i],source[j])
                for i in range(m):
                    for j in range(m):
                        self.k_YY[i,j] = self.k(target[i],target[j])
                print("Kernel matrices constructed!")
            if source_weights != None:
                w = source_weights.reshape((n,))/n
            else:
                w = torch.ones(n)/n
            if target_weights != None:
                w_ring = target_weights.reshape((m,))/m
            else:
                w_ring = torch.ones(m)/m
                
        MMD = torch.matmul(torch.matmul((w*weights).T,self.k_XX),w*weights) + torch.matmul(torch.matmul(w_ring.T,self.k_YY),w_ring) - 2.0*torch.matmul(torch.matmul((w*weights).T,self.k_XY),w_ring)
        MMD_reg = MMD + lambda_l2*weights.square().mean()
        optimizer = torch.optim.Adam([weights], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        optimizer.zero_grad()
        MMD_reg.backward()
        optimizer.step() 
        return MMD

