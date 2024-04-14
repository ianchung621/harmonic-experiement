import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

class File():

    def __init__(self,fn , unit = 'mV'):

        self.current = search_num(fn, 'mA')
        self.field = search_num(fn, 'T')
        self.temp = search_num(fn, 'K')
        self.expID = int(re.search(r'exp(\d+)_', fn).group(1)) if re.search(r'exp(\d+)_', fn) else None
        self.harm = '2nd' if re.search(r'2nd', fn) else '1st'
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

    def get_voltage_history(self,angle,plot = True):
        row_idx = np.argmin(np.abs(self.df['theta']-angle))
        if plot==True:
            plt.plot(self.voltage_history.iloc[row_idx].values)
            plt.xlabel("angle(degree)")
            plt.ylabel("amp("+self.unit_str+")")
            plt.title(f'voltage history of {angle}')
        return self.voltage_history.iloc[row_idx].values
    
    def get_V_theta(self):
        return self.df['theta'],self.df['amp']
    
    def flip_signal(self,angle_ranges,plot = True, label = None):
        sign = np.ones(len(self.df))
        for a_range in angle_ranges:
            sign = sign*np.where(np.logical_and(self.df['theta'] > a_range[0] ,self.df['theta'] < a_range[1]) , -np.ones(len(self.df)),np.ones(len(self.df)))
        self.df['fliped_amp'] = self.df['amp']*sign
        if plot == True:
            plt.scatter(self.df['theta'] ,sign*self.df['amp'], label = label,alpha=0.5)
            plt.xlabel("angle(degree)")
            plt.ylabel("amp("+self.unit_str+")")
            plt.title(f'fliped voltage')

class Folder():
    
    def __init__(self,folder):
        self.fns = glob.glob(f'{folder}/**.csv')
        self.datas = [File(fn) for fn in self.fns]
    
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