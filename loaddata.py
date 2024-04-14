import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import torch
import torch.nn as nn
torch.manual_seed(69)

class DATA():

    def __init__(self,fn , unit = 'mV'):

        self.current = search_num(fn, 'mA')
        self.field = search_num(fn, 'T')
        self.temp = search_num(fn, 'K')
        self.expID = int(re.search(rf'exp(\d+)_', fn).group(1)) if re.search(rf'exp(\d+)_', fn) else None
        self.harm = '2nd' if re.search(rf'2nd', fn) else '1st'
        self.FC = search_keyword(fn, 'ZFC', 'FC')
        self.CC = search_keyword(fn, 'ZCC', 'CC')
        self.unit_str = unit

        if unit == 'mV':
            toV = 1e3
        elif unit == 'uV':
            toV = 1e6
        self.df = pd.read_csv(fn,header=None)
        col_names=['theta','amp','amp_err','phase','phase_err']+[f'V{i}' for i in range(len(self.df.columns)-5)]
        self.df.columns = col_names

        expand = 100 if self.harm == '2nd' else 1
        self.df['amp'] = self.df['amp']*toV/expand
        self.df['amp_err'] = self.df['amp_err']*toV/expand
        for i in range(len(self.df.columns)-5):
            self.df[f'V{i}'] = self.df[f'V{i}']*toV/expand
        self.voltage_history = self.df[[f'V{i}' for i in range(len(self.df.columns)-5)]]

    def plot_V_theta(self, xlim = None, ylim = None, label = None, fmt = 'bo',title = ''):
        plt.errorbar(self.df['theta'],self.df['amp'],self.df['amp_err'],fmt=fmt,label = label)
        plt.xlabel("angle(degree)")
        plt.ylabel("amp("+self.unit_str+")")
        plt.title("Hall voltage - angle "+title)
        plt.xlim(xlim) if xlim != None else None
        plt.ylim(ylim) if ylim != None else None
        
    def plot_Vr_and_phase(self, ylim_vx = None, ylim_vy = None,title = '', mutiplier = 1):
        
        # Creating plot with dataset_1
        fig, ax1 = plt.subplots() 
        
        ax1.set_xlabel('angle(degree)') 
        ax1.set_ylabel('voltage (mV)',color = 'b') 
        ax1.tick_params('y',color = 'b')
        ax1.errorbar(self.df['theta'],self.df['amp'],self.df['amp_err'],fmt='bo',label = 'voltage (mV)')
        ax1.legend()
        ax1.set_ylim(ylim_vx) if ylim_vx != None else None
            
        # Adding Twin Axes to plot using dataset_2
        ax2 = ax1.twinx() 
        ax2.set_ylabel('phase (deg)', color = 'r')
        ax2.tick_params('y', color = 'r')
        ax2.errorbar(self.df['theta'],self.df['phase'],self.df['phase_err'],fmt='ro',label = 'phase (deg)')
        ax2.legend()
        plt.title("locking result"+title)
        ax1.set_ylim(ylim_vx) if ylim_vx != None else None
    
    def plot_Vx_and_Vy(self, ylim_voltage = None, ylim_phase = None,title = '', mutiplier = 1):
        
        Vr = self.df['amp']
        phase = self.df['phase']
        # Creating plot with dataset_1
        fig, ax1 = plt.subplots()
        
        ax1.set_xlabel('angle(degree)') 
        ax1.set_ylabel('V_x (mV)',color = 'b') 
        ax1.tick_params('y',color = 'b')
        ax1.errorbar(self.df['theta'],Vr*np.cos(np.deg2rad(phase)),fmt='bo',label = 'Vx (mV)')
        ax1.legend()
        ax1.set_ylim(ylim_voltage) if ylim_voltage != None else None
        
        # Adding Twin Axes to plot using dataset_2
        ax2 = ax1.twinx() 
        ax2.set_ylabel('V_y (mV)', color = 'r')
        ax2.tick_params('y', color = 'r')
        ax2.errorbar(self.df['theta'],Vr*np.sin(np.deg2rad(phase)),fmt='ro',label = 'Vy (mV)')
        ax2.legend()
        plt.title("locking result"+title)
        ax1.set_ylim(ylim_phase)if ylim_phase != None else None
    
    def plot_V_theta_all(self, xlim = None, ylim = None,title = '',cmap = 'Blues'):
        for i in range(len(self.df.columns)-5):
            if cmap != None:
                plt.scatter(self.df['theta'],self.df[f'V{i}'],c=[i]*len(self.df),cmap=cmap,vmax=len(self.df.columns)-5,vmin=0,s=0.5, label = f'V{i}')
            else:
                plt.scatter(self.df['theta'],self.df[f'V{i}'], c='k', s=0.5, label = f'V{i}')
        plt.xlabel("angle(degree)")
        plt.ylabel("amp("+self.unit_str+")")
        plt.title("Hall voltage - angle"+title)
        plt.xlim(xlim) if xlim != None else None
        plt.ylim(ylim) if ylim != None else None

    def get_voltage_history(self,angle):
        row_idx = np.argmin(np.abs(self.df['theta']-angle))
        return self.voltage_history.iloc[row_idx].values
    
    def get_V_theta(self):
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
    
    def plot_data(self, attribute, range = None, xlim=None, ylim=None, title='', cmap='coolwarm', label=''):
        if attribute not in ['temp', 'field', 'current']:
            raise ValueError("Invalid attribute. Choose from 'temp', 'field', or 'current'.")
        
        data_values = [getattr(data, attribute) for data in self.datas]
        sorted_data = sorted(zip(data_values, self.datas), key=lambda x: x[0])
        sorted_values, sorted_datas = zip(*sorted_data)

        for value, data_obj in zip(sorted_values, sorted_datas):
            plt.scatter(data_obj.get_V_theta()[0], data_obj.get_V_theta()[1],
                        c=[value]*len(data_obj.get_V_theta()[0]),
                        cmap=cmap, vmax=range[1], vmin=range[0],
                        label=f'{value} {label}')
        
        plt.xlabel("angle(degree)")
        plt.ylabel("amp(mV)")
        plt.title("Hall voltage - angle" + title)
        plt.legend(title=attribute)
        
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
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
        self._a = nn.Parameter(torch.randn(1)*0.001)
        self._b = nn.Parameter(torch.randn(1)*0.001)
        self._c = nn.Parameter(torch.randn(1)*0.001)
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


def search_num(string, keyword):
            if re.search(rf'(\d+){keyword}', string):
                return float(re.search(rf'(\d+){keyword}', string).group(1))
            elif re.search(rf'(\d+).(\d+){keyword}', string):
                a = re.search(rf'(\d+).(\d+){keyword}', string).group(1)
                b = re.search(rf'(\d+).(\d+){keyword}', string).group(2)
                return float(a+'.'+b)
            else:
                return None
        
def search_keyword(string, keyword, keyword_substring):
    if keyword in string:
        return keyword
    elif keyword_substring in string:
        return keyword_substring
    else:
        return None

if __name__ == '__main__':
    # load data
    fn = 'DATA/NiPS3_epic/exp2_20K_5T_3mA_FC_CC.csv'
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
    fn = 'DATA/NiPS3_epic/exp2_20K_5T_3mA_FC_CC_2nd.csv'
    data_2nd = DATA(fn,expand=100)
    # fit curve
    function_2nd = FitData(data_2nd)
    function_2nd.model.b = float(function_1st.param[1])

    function_2nd.optimize(lr = 0.001, max_iter=100000,eval_iter=1000,threshold = None, verbose=True)
    print(function_2nd)
    #plotting
    plt.subplot(1,2,2)
    function_2nd.plot_2nd_calibrated(thres_ratio=0.1)
    plt.legend()

    V_1st = float(function_1st.param[0])
    V_2nd = float(function_2nd.param[0])

    H_FL = data_2nd.field*(V_2nd/data_2nd.current)/(V_1st/data_1st.current)
    Pt_width = 5e-6
    Pt_thickness = 10e-9
    current_density = 1e-3*data_2nd.current/Pt_thickness/Pt_width*1e-12 # in unit of 10^12 A/m^2
    print(f'current density: {current_density*1e12} A/m²')
    print(f'Field like SOT effective field: {H_FL:.4f} T')
    print(f'Field like SOT effective field per current density: {H_FL/current_density:.4f} T/(10¹²A/m²)')
    plt.tight_layout()
    plt.savefig('harmfit.png')