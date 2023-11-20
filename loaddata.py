import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

class DATA():

    def __init__(self,fn ,expand = 1, mutiplier = 1E6, unit = 'mV'):
        self.unit_str = unit
        if unit == 'mV':
            self.unit = 1e3
        elif unit == 'uV':
            self.unit = 1e6
        self.df = pd.read_csv(fn,header=None)
        col_names=['theta','amp','amp_err','phase','phase_err']+[f'V{i}' for i in range(len(self.df.columns)-5)]
        self.df.columns = col_names
        self.df['amp'] = self.df['amp']*self.unit/expand/mutiplier
        self.df['amp_err'] = self.df['amp_err']*self.unit/expand/mutiplier
        for i in range(len(self.df.columns)-5):
            self.df[f'V{i}'] = self.df[f'V{i}']*self.unit/expand/mutiplier
        self.voltage_history = self.df[[f'V{i}' for i in range(len(self.df.columns)-5)]]
        get_current = lambda file: int(re.search(r'(\d+)mA', file).group(1)) if re.search(r'(\d+)mA', file) else None
        get_field = lambda file: int(re.search(r'(\d+)T', file).group(1)) if re.search(r'(\d+)T', file) else None
        get_temp = lambda file: int(re.search(r'(\d+)K', file).group(1)) if re.search(r'(\d+)K', file) else None
        get_expID = lambda file: int(re.search(r'exp(\d+)_', file).group(1)) if re.search(r'exp(\d+)_', file) else None
        get_FC = lambda file: re.search(r'(_FC_|_ZFC_|_FC|_ZFC)', file).group(1) if re.search(r'(_FC_|_ZFC_|_FC|_ZFC)', file) else None
        get_CC = lambda file: re.search(r'(_CC_|_ZCC_|_CC|_ZCC)', file).group(1) if re.search(r'(_CC_|_ZCC_|_CC|_ZCC)', file) else None
        get_harm = lambda file: '2nd' if re.search(r'2nd', file) else '1st'
        self.current = get_current(fn)
        self.field = get_field(fn)
        self.temp = get_temp(fn)
        self.expID = get_expID(fn)
        self.harm = get_harm(fn)
        self.FC = get_FC(fn).replace('_', '') if get_FC(fn) is not None else get_FC(fn)
        self.CC = get_CC(fn).replace('_', '') if get_CC(fn) is not None else get_CC(fn)

    def plot_V_theta(self, xlim = None, ylim = None, label = None, fmt = 'bo',title = ''):
        plt.errorbar(self.df['theta'],self.df['amp'],self.df['amp_err'],fmt=fmt,label = label)
        plt.xlabel("angle(degree)")
        plt.ylabel("amp("+self.unit_str+")")
        plt.title("Hall voltage - angle"+title)
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)
    
    def plot_V_theta_all(self, xlim = None, ylim = None,title = '',cmap = 'Blues'):
        for i in range(len(self.df.columns)-5):
            if cmap != None:
                plt.scatter(self.df['theta'],self.df[f'V{i}'],c=[i]*len(self.df),cmap=cmap,vmax=len(self.df.columns)-5,vmin=0,s=0.5, label = f'V{i}')
            else:
                plt.scatter(self.df['theta'],self.df[f'V{i}'], c='k', s=0.5, label = f'V{i}')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp("+self.unit_str+")")
        plt.title("Hall voltage - angle"+title)
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)
    

    def get_voltage_history(self,angle):
        row_idx = np.argmin(np.abs(self.df['theta']-angle))
        return self.voltage_history.iloc[row_idx].values
    
    def get_V_theta(self):
        '''
        return theta,V
        '''
        return self.df['theta'],self.df['amp']


class DATAS():
    
    def __init__(self,folder):
        self.fns = glob.glob(f'{folder}/**.csv')
        self.datas = [DATA(fn) for fn in self.fns]
    
    def select_data(self,temp = None,field = None, FC = None, CC = None, current = None, expID = None, harm = '1st'):
        fns = []
        datalist = []
        for fn,data in zip(self.fns,self.datas):
            cond_temp = (data.temp == temp) or (temp == None)
            cond_field = (data.field == field) or (field == None)
            cond_current = (data.current == current) or (current == None)
            cond_FC = (data.FC == FC) or (FC == None)
            cond_CC = (data.CC == CC) or (CC == None)
            cond_ID = (data.expID == expID) or (expID == None)
            cond_harm = (data.harm == harm)
            if cond_temp and cond_field and cond_current and cond_FC and cond_CC and cond_ID and cond_harm:
                fns.append(fn)
                datalist.append(data)
        self.fns = fns
        self.datas = datalist
    
    def plot_temp(self,T_range = (10,200), xlim = None, ylim = None, title = '', cmap = 'coolwarm'):
        theta_Vs = [data.get_V_theta() for data in self.datas]
        temps = [data.temp for data in self.datas]

        sorted_data = sorted(zip(theta_Vs, temps), key=lambda x: x[1])
        theta_Vs, temps = zip(*sorted_data)

        for theta_V, temp in zip(theta_Vs,temps):
            plt.scatter(theta_V[0],theta_V[1],c=[temp]*len(theta_V[0]),cmap=cmap,vmax=T_range[1],vmin=T_range[0],label = f'{temp} K')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp(mV)")
        plt.title("Hall voltage - angle" + title)
        plt.legend(title = 'tempurature')
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)
    
    def plot_field(self,H_range = (1,5), xlim = None, ylim = None, title = '', cmap = 'cool'):

        theta_Vs = [data.get_V_theta() for data in self.datas]
        fields = [data.field for data in self.datas]

        sorted_data = sorted(zip(theta_Vs, fields), key=lambda x: x[1])
        theta_Vs, fields = zip(*sorted_data)

        for theta_V, field in zip(theta_Vs,fields):
            plt.scatter(theta_V[0],theta_V[1],c=[field]*len(theta_V[0]),cmap=cmap,vmax=H_range[1],vmin=H_range[0],label = f'{field} T')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp(mV)")
        plt.title("Hall voltage - angle" + title)
        plt.legend(title = 'field')
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)

    
    def plot_current(self,I_range = (1,15), xlim = None, ylim = None, title = '', cmap = 'cool'):

        theta_Vs = [data.get_V_theta() for data in self.datas]
        currents = [data.current for data in self.datas]

        sorted_data = sorted(zip(theta_Vs, currents), key=lambda x: x[1])
        theta_Vs, currents = zip(*sorted_data)

        for theta_V, current in zip(theta_Vs,currents):
            plt.scatter(theta_V[0],theta_V[1],c=[current]*len(theta_V[0]),cmap=cmap,vmax=I_range[1],vmin=I_range[0],label = f'{current} mA')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp(mV)")
        plt.title("Hall voltage - angle" + title)
        plt.legend(title = 'current')
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)
    
    def plot_cond(self, xlim = None, ylim = None, title = ''):

        theta_Vs = [data.get_V_theta() for data in self.datas]
        CCs = [data.CC for data in self.datas]
        FCs = [data.FC for data in self.datas]

        for theta_V, FC, CC in zip(theta_Vs,FCs,CCs):
            plt.scatter(theta_V[0],theta_V[1],label = FC+', '+CC)
        plt.xlabel("angle(degree)")
        plt.ylabel("amp(mV)")
        plt.title("Hall voltage - angle"+title)
        plt.legend(title = 'condition')
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)

    
class FitData:

    def __init__(self, data:DATA):
        self.data = data
        self.fitting_type = data.harm
        self.x = data.df['theta'].values
        self.y = data.df['amp'].values
        self.yerr = data.df['amp_err'].values
        self.model = FirstHarmonicFun() if self.fitting_type == '1st' else SecondHarmonicFun()
        self.param = [self.model.a.detach().numpy(), self.model.b.detach().numpy(), self.model.c.detach().numpy()]
    
    def __repr__(self):
        if self.fitting_type == '1st':
            return f'fitting type: 1st \n function:{self.param[0]}*sin(2*(theta + {self.param[1]})+{self.param[2]})'
        else:
            return f'fitting type: 2nd \n function:{self.param[0]}*cos(2*(theta + {self.param[1]})*cos(theta + {self.param[1]})+{self.param[2]})'
        
    def optimize(self, lr=0.01, max_iter = 1000, eval_iter = 100, threshold = 100 ,verbose = False):

        # Preprocessing
        model = self.model
        x = torch.tensor(self.x,dtype=torch.float32)
        y = torch.tensor(self.y,dtype=torch.float32)
        mean = y.mean(dim=0)
        std = y.std(dim=0)

        # Standardize the data
        y = (y - mean) / std if self.fitting_type == '1st' else y/std

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = CustomLoss(threshold)

        # Training loop
        max_iter = max_iter
        for iter in range(max_iter):
            # Forward pass
            if self.fitting_type == '1st':
                outputs = model(x)
            else:
                outputs,_ = model(x)
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
            self.param = [(model.a*std).detach().numpy(), model.b.detach().numpy(), (model.c*std-mean).detach().numpy()]
        elif self.fitting_type == '2nd':
            self.param = [(model.a*std).detach().numpy(), model.b.detach().numpy(), (model.c*std).detach().numpy()]

    def predict(self,x,calibrated = False):
        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(self.y,dtype=torch.float32)
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        if self.fitting_type == '1st':
            out = self.model(x)*std + mean
        elif self.fitting_type == '2nd':
            if calibrated:
                _,out = self.model(x)
            else:
                out,_ = self.model(x)
            out = out*std
        return out.detach().numpy()
    
    def eval(self):
        return np.mean(np.sqrt((self.y - self.predict(self.x))**2))
    
    def plot(self):
        x_plot = np.linspace(0,360,360)
        plt.plot(x_plot,self.predict(x_plot),'r',label = 'Fit')
        self.data.plot_V_theta(label='Data',title=f' {self.fitting_type}')
    
    def plot_fitline(self):
        x_plot = np.linspace(0,360,360)
        plt.plot(x_plot,self.predict(x_plot),'r')

    def plot_2nd_calibrated(self, thres_ratio = 0.4):
        x_plot = np.linspace(0,360,360)
        plt.plot(x_plot,self.predict(x_plot,calibrated=True),'r',label = 'Fit')

        y_pred_orig = self.predict(self.x,calibrated=False)
        y_pred_cali = self.predict(self.x,calibrated=True)
        y_pred_diff = y_pred_orig - y_pred_cali
        y_cali = np.where(y_pred_diff > thres_ratio*np.max(y_pred_diff),-self.y,self.y)
        plt.errorbar(self.x,y_cali,self.yerr,fmt='bo',label = 'Data')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp("+self.data.unit_str+")")
        plt.title("Hall voltage - angle"+f' {self.fitting_type}')

class CustomLoss(torch.nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = torch.tensor([threshold],dtype=torch.float32,requires_grad=True)

    def forward(self, pred, targ):
        loss = torch.where(torch.abs(pred - targ) < self.threshold,
                           (pred - targ)**2,
                           self.threshold**2)
        return torch.mean(loss)

class FirstHarmonicFun(nn.Module):
    def __init__(self):
        super().__init__()
        self._a = nn.Parameter(torch.rand(1))
        self._b = nn.Parameter(torch.rand(1))
        self._c = nn.Parameter(torch.rand(1))

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
        self._a = nn.Parameter(torch.rand(1))
        self._b = nn.Parameter(torch.rand(1))
        self._c = nn.Parameter(torch.rand(1))
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
        return self.a * torch.abs(torch.cos(2*(x - self.b))*torch.cos(x - self.b) + self.c), self.a * torch.cos(2*(x - self.b))*torch.cos(x - self.b) + self.c

if __name__ == '__main__':
    # load data
    fn = 'DATA/NiPS3_100nm_1/exp1_953Hz_10mA_1st.csv'
    data_1st = DATA(fn)
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
    fn = 'DATA/NiPS3_100nm_1/exp8_953Hz_12mA_3T_2nd_detailscan.csv'
    data_2nd = DATA(fn,expand=100)
    # fit curve
    function_2nd = FitData(data_2nd)
    function_2nd.model.b = float(function_1st.param[1])
    print(function_1st.param[1])
    function_2nd.optimize(threshold=1)
    print(function_2nd)
    #plotting
    plt.subplot(1,2,2)
    function_2nd.plot_2nd_calibrated()
    plt.legend()

    V_1st = float(function_1st.param[0])
    V_2nd = float(function_2nd.param[0])

    H_FL = data_2nd.field*(V_2nd/data_2nd.current)/(V_1st/data_1st.current)
    Pt_width = 5e-6
    Pt_thickness = 100e-9
    current_density = 1e-3*data_2nd.current/Pt_thickness/Pt_width*1e-12 # in unit of 10^12 A/m^2
    print(f'current density: {current_density*1e12} A/m²')
    print(f'Field like SOT effective field: {H_FL:.4f} T')
    print(f'Field like SOT effective field per current density: {H_FL/current_density:.4f} T/(10¹²A/m²)')
    plt.tight_layout()
    plt.savefig('harmfit.png')


    