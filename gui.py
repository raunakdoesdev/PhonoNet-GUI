from tkinter import *
import os
from matplotlib.cm import get_cmap
from tkinter import filedialog
from tkinter import Menu
import tkinter as tk
import json, torch, librosa, sys
import torch.utils.data as data
import numpy as np
import librosa.display
from models import RagaDetector

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
# from __future__ import unicode_literals
import youtube_dl

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


class SalientRagaDetector(nn.Module):
  def __init__(self, raga_detector): # Create new broken down raga detector based on existing model
    super(SalientRagaDetector, self).__init__()
    self.encoder1 = nn.Sequential(raga_detector.encoder[0:6])
    self.encoder2 = nn.Sequential(raga_detector.encoder[6:11])
    self.encoder3 = nn.Sequential(raga_detector.encoder[11:16])
    self.encoder4 = nn.Sequential(raga_detector.encoder[16:20])
    self.fc1 = raga_detector.fc1

  def forward(self, x):
    out0 = x.mean(1).unsqueeze(0)

    self.up1 =  nn.Upsample([x.shape[2], x.shape[3]], mode = 'nearest')
    x = self.encoder1(x)
    out1 = x.mean(1).unsqueeze(0)

    self.up2 =  nn.Upsample([x.shape[2], x.shape[3]], mode = 'nearest')
    x = self.encoder2(x)
    out2 = x.mean(1).unsqueeze(0)

    self.up3 = nn.Upsample([x.shape[2], x.shape[3]], mode = 'nearest')
    x = self.encoder3(x)
    out3 = x.mean(1).unsqueeze(0)

    self.up4 = nn.Upsample([x.shape[2], x.shape[3]], mode = 'nearest')
    x = self.encoder4(x)
    out4 = x.mean(1).unsqueeze(0)

    batch_size = x.shape[0]
    x = x.view(batch_size, -1)

    # Magical line that scales everything and does the mathsss
    activation_map = torch.mul(self.up1(torch.mul(self.up2(torch.mul(self.up3(out3), out2)), out1)), out0)

    return x, activation_map



ragas = ['Todi'           ,
'Puriyā dhanaśrī' ,
'Rāgēśrī'         ,
'Mārvā'           ,
'Yaman kalyāṇ'    ,
'Bhūp'            ,
'Miyān malhār'    ,
'Madhukauns'      ,
'Madhuvanti'      ,
'Basant'          ,
'Khamāj'          ,
'Kēdār'           ,
'Lalit'           ,
'Alahaiyā bilāval',
'Bhairav'         ,
'Bihāg'           ,
'Jōg'             ,
'Bilāsakhānī tōḍī',
'Mārūbihāg'       ,
'Darbāri'         ,
'Dēś'             ,
'Ahira bhairav'   ,
'Śrī'             ,
'Gauḍ malhār'     ,
'Ābhōgī'          ,
'Śuddh sāraṅg'    ,
'Bairāgi'         ,
'Mālkauns'        ,
'Bāgēśrī'         ,
'Hamsadhvāni']

import matplotlib.pyplot as plt

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.top.attributes('-type', 'dialog')
        self.l=Label(top,text="Enter URL")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class Program:
    def solve(self, file_name):
        print('Loading song')
    
        sr = 22050
        if(os.path.isfile(file_name.replace('.mp3', '.npy'))):
            vsong = np.load(file_name.replace('.mp3', '.npy')) 
        else:
            y, sr = librosa.load(file_name)
            vsong = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)
            np.save(file_name.replace('.mp3', ''), vsong)

        print('Done loading song')

        song = torch.from_numpy(vsong).unsqueeze(0).unsqueeze(0).float()
        model = RagaDetector(0, 256)
        model.load_state_dict(torch.load('aug2_epoch_665_acc_0.7888888888888889.model', map_location='cpu')['net'])
        model.eval()

        # Create vismodel on top of existing raga detector
        vis_model = SalientRagaDetector(model)
        vis_model.eval()

        out = None
        chunk_confidences = []
        clearvis = []
        with torch.no_grad():
            song_segs = song.split(1500, dim=3)
            for i in range(len(song_segs)):
                chunk = song_segs[i] # get specific chunk
                if not chunk.shape[3] == 1500:  # hardcoded 1500 size
                    padding = torch.zeros(1, 1, chunk.size()[2], 1500 - chunk.size()[3])
                    chunk = torch.cat((chunk, padding), 3)
                if(out is None):
                   out = model.fc1(model(chunk.contiguous()))
                else:
                   out += model.fc1(model(chunk.contiguous()))
                chunk_confidences.append(torch.nn.functional.softmax(out, dim=1).numpy()[0] * 100.)

                _,saliency = vis_model(chunk.contiguous())
                clearvis.append(saliency[0][0])

        print(out)
        clearvis = np.concatenate(clearvis, axis=1)
        confidences = torch.nn.functional.softmax(out, dim=1).numpy()[0]*100
        print(chunk_confidences)
        self.line1.remove()
        self.line1 = self.ax.bar(ragas, confidences)
        self.canvas.draw()

        prediction = np.argmax(confidences)

        # Define a ten second window size
        x_size = 120
        max_x_start = -1
        max_accuracy_delta = 10000

        for x_start in range(0, vsong.shape[1], x_size): # iterate through windows of x_size
          
          # Crop out segment of the chromagram
          new_vsong = np.array(vsong, copy=True)
          new_vsong[:, x_start : x_start + x_size] = 0
          
          # Run new system through network
          song_segs = torch.from_numpy(new_vsong).unsqueeze(0).unsqueeze(0).float().split(1500, dim=3)
          with torch.no_grad():
            out = model.fc1(model(song_segs[x_start//1500].contiguous()))

          confidences = torch.nn.functional.softmax(out, dim=1).numpy()[0]*100.
          
          # Compute accuracy delta
          accuracy_delta = 100. * (float(confidences[prediction]) - float(chunk_confidences[x_start//1500][prediction])) / float(chunk_confidences[x_start//1500][prediction])

          if(accuracy_delta < max_accuracy_delta):
              max_accuracy_delta = accuracy_delta
              max_x_start = x_start

          print('\nRemoving ' + str(int(x_start*0.09287981859)) + ' s - ' + str(int((x_start+x_size)*0.09287981859)) + ' s => ' + str(accuracy_delta) + u' Δ % Accuracy' ) 
          print(x_start)
          print(x_start + x_size)

        import subprocess
        subprocess.Popen([ "vlc" , file_name, "--start-time", str(max_x_start * 92.87981859 / 1000.), "--stop-time", str((max_x_start + x_size) * 92.87981859 / 1000.), '--loop'])

        plt.subplot(2, 1, 1)
        librosa.display.specshow((vsong[:, max_x_start : max_x_start + x_size]), x_axis='time', hop_length=2048, sr=sr, cmap=get_cmap('magma'), y_axis='chroma')
        plt.colorbar()
        plt.title('Original Chromagram')
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        librosa.display.specshow((clearvis[:, max_x_start : max_x_start + x_size]), x_axis='time', hop_length=2048, sr=sr, cmap=get_cmap('magma'), y_axis='chroma')
        plt.colorbar()
        plt.title('PhonoViz Generated Chromagram')
        plt.tight_layout()
        plt.show()

    
    def youtubeDialog(self):
        self.w = popupWindow(self.root)
        self.root.wait_window(self.w.top)
        url = self.w.value
        name = url.split('=')[1]

        if not os.path.exists(name + ".mp3"):
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': name + '.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        self.solve(name + ".mp3")

    def fileDialog(self):
        file_name = (filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("mp3 files","*.mp3"),("all files","*.*"))))
        self.solve(file_name)


    def __init__(self):
        self.root = Tk()
        self.root.attributes('-type', 'dialog')
        self.root.title("PhonoNet")
        self.root.geometry('1000x800')
        self.menu = Menu(self.root)
        
        self.new_item = Menu(self.menu)
        self.new_item.add_command(label='New Audio File', command=self.fileDialog)
        self.new_item.add_command(label='New Youtube Song', command=self.youtubeDialog)
        self.menu.add_cascade(label='File', menu=self.new_item)
        self.fig = Figure(figsize=(5,7,), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line1 = self.ax.bar(ragas, [0] * len(ragas))
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Prediction Confidence (%)', fontsize=14)
        self.ax.set_xlabel('\nRaga', fontsize=14)
        self.ax.set_title('Prediction Confidence vs. Raga', fontsize=14)
        for tick in self.ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(12)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.root.config(menu=self.menu)
        self.root.mainloop()

        print('Loading song')

        if(os.path.isfile(file_name.replace('.mp3', '.npy'))):
            vsong = np.load(file_name.replace('.mp3', '.npy')) 
        else:
            y, sr = librosa.load(file_name)
            vsong = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)
            np.save(file_name.replace('.mp3', ''), vsong)

        print('Done loading song')

        song = torch.from_numpy(vsong).unsqueeze(0).unsqueeze(0).float()
p = Program()
