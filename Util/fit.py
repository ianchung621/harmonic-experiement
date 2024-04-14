import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from Util.load import File
except:
    from load import File
import torch
import torch.nn as nn
torch.manual_seed(420)

class FitData:

    def __init__(self, data:File):
        self.data = data
        self.fitting_type = data.harm
        self.x = data.df['theta'].values
        self.y = data.df['fliped_amp'].values if ('fliped_amp' in data.df.columns) else data.df['amp'].values
        self.yerr = data.df['amp_err'].values
        self.model = SecondHarmonicFun() if self.fitting_type == '2nd' else FirstHarmonicFun()
        if self.fitting_type == '1st':
            self.param = [self.model.a.detach().numpy(), self.model.b.detach().numpy(), self.model.c.detach().numpy()]
        elif self.fitting_type == '2nd':
            self.param = [self.model.a.detach().numpy(), self.model.b.detach().numpy(), self.model.c.detach().numpy(), self.model.d.detach().numpy()]
    
    def __repr__(self):
        mean = self.y.mean()
        std = self.y.std()
        if self.fitting_type == '1st':
            return f'fitting type: 1st \n function:{std*self.param[0]}*sin(2*(theta - {self.param[1]})+{std*self.param[2]+mean})'
        elif self.fitting_type == '2nd':
            return f'fitting type: 2nd \n function:{std*self.param[0]}*cos(2*(theta - {self.param[1]})*cos(theta - {self.param[1]}) + {std*self.param[3]*self.param[1]}*cos(theta - {self.param[1]}) + {std*self.param[2]+mean})'
        
    def optimize(self, lr=0.01, max_iter = 1000, eval_iter = 100, threshold = 100 ,verbose = False):

        # Preprocessing
        model = self.model
        x = torch.tensor(self.x,dtype=torch.float32)
        y = torch.tensor(self.y,dtype=torch.float32)
        mean = y.mean(dim=0)
        std = y.std(dim=0)

        # Standardize the data
        y = (y - mean) / std

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = CustomLoss(threshold)

        # Training loop
        max_iter = max_iter
        for iter in range(max_iter):
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % eval_iter == 0 or iter == max_iter - 1:
                if verbose == True:
                    print(f'step {iter}, loss: {loss.item():.8f}')
        
        # post processing
        if self.fitting_type == '1st':
            self.param = [(model.a).detach().numpy(), model.b.detach().numpy(), (model.c).detach().numpy()]
            self.V_H = float(std.item()*self.param[0])
        elif self.fitting_type == '2nd':
            self.param = [(model.a).detach().numpy(), model.b.detach().numpy(), (model.c).detach().numpy(), (model.d).detach().numpy()]
            self.V_FL = float(std*self.param[0])
            self.V_SSE = float(std*self.param[0]*self.param[3])

    def predict(self,x):
        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(self.y,dtype=torch.float32)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        y = (y - mean) / std

        out = self.model(x)*std + mean

        return out.detach().numpy()
    
    def eval(self):
        return np.mean(np.sqrt((self.y - self.predict(self.x))**2))
    
    def plot_fitline(self):
        x_plot = np.linspace(0,360,360)
        plt.plot(x_plot,self.predict(x_plot),'r')

    def plot(self):
        x_plot = np.linspace(0,360,360)
        plt.plot(x_plot,self.predict(x_plot),'r',label = 'Fit')

        plt.errorbar(self.x,self.y,self.yerr,fmt='bo',label = 'Data')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp("+self.data.unit_str+")")
        plt.title("Hall voltage - angle"+f' {self.fitting_type}')

class CustomLoss(torch.nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        if threshold is not None:
            self.threshold = torch.tensor([threshold], dtype=torch.float32, requires_grad=False)
        else:
            self.threshold = None

    def forward(self, pred, targ):
        if self.threshold is not None:
            loss = torch.where(torch.abs(pred - targ) < self.threshold,
                                (pred - targ)**2,
                                self.threshold**2)
        else:
            loss = (pred - targ)**2  # Calculate squared error directly
        return torch.mean(loss)

class FirstHarmonicFun(nn.Module):
    def __init__(self):
        super().__init__()
        self._a = nn.Parameter(torch.randn(1))
        self._b = nn.Parameter(torch.randn(1))
        self._c = nn.Parameter(torch.randn(1))

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a.data = torch.tensor([value],dtype=torch.float32)
        self._a.requires_grad = False

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b.data = torch.tensor([value],dtype=torch.float32)
        self._b.requires_grad = False

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c.data = torch.tensor([value],dtype=torch.float32)
        self._c.requires_grad = False

    def forward(self, x):
        x = torch.deg2rad(x)
        return self.a * torch.sin(2*(x - self.b)) + self.c

class SecondHarmonicFun(nn.Module):
    def __init__(self):
        super().__init__()
        self._a = nn.Parameter(torch.randn(1)*0.1)
        self._b = nn.Parameter(torch.randn(1))
        self._c = nn.Parameter(torch.randn(1)*0.1)
        self._d = nn.Parameter(torch.randn(1)*0.1)
    
    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a.data = torch.tensor([value],dtype=torch.float32)
        self._a.requires_grad = False

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b.data = torch.tensor([value],dtype=torch.float32)
        self._b.requires_grad = False

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c.data = torch.tensor([value],dtype=torch.float32)
        self._c.requires_grad = False
    
    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d.data = torch.tensor([value],dtype=torch.float32)
        self._d.requires_grad = False
    
    def forward(self, x):
        x = torch.deg2rad(x)
        return torch.abs(self.a * torch.cos(2*(x - self.b))*torch.cos(x - self.b) + self.a * self.d * torch.cos(x - self.b)) + self.c


if __name__ == '__main__':
    
    field = 5
    # load data
    fn = f'DATA/NiPS3_epic/exp3_20K_{field}T_3mA_FC_CC.csv'
    data_1st = File(fn)
    # fit curve
    function_1st = FitData(data_1st)
    function_1st.optimize()
    print(function_1st)

    fig = plt.figure(figsize=(12, 6))

    # plotting
    plt.subplot(1,2,1)
    function_1st.plot()
    plt.legend()

    # load data
    fn = f'DATA/NiPS3_epic/exp2_20K_{field}T_3mA_FC_CC_2nd.csv'
    data_2nd = File(fn)
    data_2nd.flip_signal(((60,165),(240,350)),plot=False)
    # fit curve
    function_2nd = FitData(data_2nd)
    function_2nd.model.b = float(function_1st.param[1])
    function_2nd.optimize(lr = 0.01, max_iter=2000,eval_iter=2000,threshold = None, verbose=True)
    print(function_2nd)

    #plotting
    plt.subplot(1,2,2)
    function_2nd.plot()
    plt.legend()

    V_1st = function_1st.V_H
    V_2nd = function_2nd.V_FL

    H_FL = data_2nd.field*(V_2nd/data_2nd.current)/(V_1st/data_1st.current)
    Pt_width = 5e-6
    Pt_thickness = 10e-9
    current_density = 1e-3*data_2nd.current/Pt_thickness/Pt_width*1e-12 # in unit of 10^12 A/m^2
    print(f'current density: {current_density:.4f} E12 A/m²')
    print(f'Field like SOT effective field: {H_FL:.4f} T')
    print(f'Field like SOT effective field per current density: {H_FL/current_density:.4f} T/(10¹²A/m²)')
    plt.tight_layout()
    plt.savefig('harmfit.png')