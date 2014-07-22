## NoteTrainer - by Alan Smith ##

import sys
import random
import math
import os     #
import pyaudio
from scipy import signal
import pygame
from socket import *
from pygame.locals import *
from random import * 
import numpy
from scipy.signal import blackmanharris, fftconvolve
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find

# See http://www.swharden.com/blog/2013-05-09-realtime-fft-audio-visualization-with-python/
class SoundRecorder:
        
    def __init__(self):
        self.RATE=48000
        self.BUFFERSIZE=3072 #1024 is a good buffer size 3072 works for Pi
        self.secToRecord=.05
        self.threadsDieNow=False
        self.newAudio=False
        
    def setup(self):
        self.buffersToRecord=int(self.RATE*self.secToRecord/self.BUFFERSIZE)
        if self.buffersToRecord==0: self.buffersToRecord=1
        self.samplesToRecord=int(self.BUFFERSIZE*self.buffersToRecord)
        self.chunksToRecord=int(self.samplesToRecord/self.BUFFERSIZE)
        self.secPerPoint=1.0/self.RATE
        self.p = pyaudio.PyAudio()
        self.inStream = self.p.open(format=pyaudio.paInt16,channels=1,rate=self.RATE,input=True,frames_per_buffer=self.BUFFERSIZE)
        self.xsBuffer=numpy.arange(self.BUFFERSIZE)*self.secPerPoint
        self.xs=numpy.arange(self.chunksToRecord*self.BUFFERSIZE)*self.secPerPoint
        self.audio=numpy.empty((self.chunksToRecord*self.BUFFERSIZE),dtype=numpy.int16)               
    
    def close(self):
        self.p.close(self.inStream)
    
    def getAudio(self):
        audioString=self.inStream.read(self.BUFFERSIZE)
        self.newAudio=True
        return numpy.fromstring(audioString,dtype=numpy.int16)
        
# See https://github.com/endolith/waveform-analyzer/blob/master/frequency_estimator.py
def parabolic(f, x): 
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)
    
# See https://github.com/endolith/waveform-analyzer/blob/master/frequency_estimator.py
def freq_from_autocorr(raw_data_signal, fs):                          
    corr = fftconvolve(raw_data_signal, raw_data_signal[::-1], mode='full')
    corr = corr[len(corr)/2:]
    d = diff(corr)
    start = find(d > 0)[0]
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px    

def loudness(chunk):
    data = numpy.array(chunk, dtype=float) / 32768.0
    ms = math.sqrt(numpy.sum(data ** 2.0) / len(data))
    if ms < 10e-8: ms = 10e-8
    return 10.0 * math.log(ms, 10.0)
        


def find_nearest(array, value):
    index = (numpy.abs(array - value)).argmin()
    return array[index]

def closest_value_index(array, guessValue):
    # Find closest element in the array, value wise
    closestValue = find_nearest(array, guessValue)
    # Find indices of closestValue
    indexArray = numpy.where(array==closestValue)
    # Numpys 'where' returns a 2D array with the element index as the value
    return indexArray[0][0]

def build_default_tuner_range():
    
    return {65.41:'C2', 
            69.30:'C2#',
            73.42:'D2',  
            77.78:'E2b', 
            82.41:'E2',  
            87.31:'F2',  
            92.50:'F2#',
            98.00:'G2', 
            103.80:'G2#',
            110.00:'A2', 
            116.50:'B2b',
            123.50:'B2', 
            130.80:'C3', 
            138.60:'C3#',
            146.80:'D3',  
            155.60:'E3b', 
            164.80:'E3',  
            174.60:'F3',  
            185.00:'F3#',
            196.00:'G3',
            207.70:'G3#',
            220.00:'A3',
            233.10:'B3b',
            246.90:'B3', 
            261.60:'C4', 
            277.20:'C4#',
            293.70:'D4', 
            311.10:'E4b', 
            329.60:'E4', 
            349.20:'F4', 
            370.00:'F4#',
            392.00:'G4',
            415.30:'G4#',
            440.00:'A4',
            466.20:'B4b',
            493.90:'B4', 
            523.30:'C5', 
            554.40:'C5#',
            587.30:'D5', 
            622.30:'E5b', 
            659.30:'E5', 
            698.50:'F5', 
            740.00:'F5#',
            784.00:'G5',
            830.60:'G5#',
            880.00:'A5',
            932.30:'B5b',
            987.80:'B5', 
            1047.00:'C6',
            1109.0:'C6#',
            1175.0:'D6', 
            1245.0:'E6b', 
            1319.0:'E6', 
            1397.0:'F6', 
            1480.0:'F6#',
            1568.0:'G6',
            1661.0:'G6#',
            1760.0:'A6',
            1865.0:'B6b',
            1976.0:'B6', 
            2093.0:'C7'
            } 
            
class NoteTrainer(object):  
        
    def main(self, screen):

        screen_size = (1024,768)
        screen_color = (0, 0 ,0)
        stepsize = 5
        
        # Build frequency, noteName dictionary
        tunerNotes = build_default_tuner_range()

        # Sort the keys and turn into a numpy array for logical indexing
        frequencies = numpy.array(sorted(tunerNotes.keys()))

        top_note = len(tunerNotes)-1
        bot_note = 0
        

        top_note = 24
        bot_note = 0
        
        # Misc variables for program controls
        centrescreen = (screen_size[0]/2,screen_size[1]/2)  # Read the tuple and create new ones
        screen = pygame.display.set_mode(screen_size)
        screen.fill(screen_color)
        inputnote = 1                               # the y value on the plot
        oldposition = (0,0)                         # memory of the last position
        shownotes = True                            # note names shown or invisible
        signal_level=0                              # volume level
        fill = True                                 #
        trys = 1
        needle = False
        cls = True
        col = False
        circ = False
        line = False
        auto_scale = False
        toggle = False
        stepchange = False
        soundgate = 19                             # zero is loudest possible input level
        targetnote=0
        SR=SoundRecorder()                          # recording device (usb mic)
        
        while trys <> 0:
            stepsizecolor = (randint(25,255),randint(45,255),randint(45,255))
            #stepsizecolor = (255,255,255)
            trys += 1
            
            for n in range(0,screen_size[0],stepsize+1):
                #### Main screen trace loop ####
                
                SR.setup()
                raw_data_signal = SR.getAudio()                                         #### raw_data_signal is the input signal data 
                signal_level = round(abs(loudness(raw_data_signal)),2)                  #### find the volume from the audio sample
                
                try: 
                    inputnote = round(freq_from_autocorr(raw_data_signal,SR.RATE),2)    #### find the freq from the audio sample
                    
                except:
                    inputnote == 0
                    
                SR.close()
                
                if inputnote > frequencies[len(tunerNotes)-1]:                        #### not interested in notes above the notes list
                    continue
                    
                if inputnote < frequencies[0]:                                     #### not interested in notes below the notes list
                    continue    
                        
                if signal_level > soundgate:                                        #### basic noise gate to stop it guessing ambient noises 
                    continue
                
                
                targetnote = closest_value_index(frequencies, round(inputnote, 2))      #### find the closest note in the keyed array
                position = ((n) ,(screen_size[1]-(int(screen_size[1]/(frequencies[top_note]-frequencies[bot_note]) * (inputnote - frequencies[bot_note])))) ) 
                
                ######## set up user controls ######## 
                
                for event in pygame.event.get():
                    if event.type ==  QUIT:  # for quitting if in windowed mode
                        SR.close()
                        return
                    elif event.type == KEYDOWN:
                        # Show the lines
                        if event.key == K_l:    
                            oldposition = position
                            line = not line
                        
                        # Decrease step / stepsize
                        if event.key == K_v:    
                            stepchange = True
                            if stepsize > 1:
                                stepsize -= 5
                                
                        # Increase step / stepsize
                        if event.key == K_b:    
                            stepchange = True
                            stepsize += 5
                            
                        # Clear screen after every sample
                        if event.key == K_s:    
                            toggle = not toggle
                        
                        # Increase top_note range
                        if event.key == K_y:    
                            if top_note < len(tunerNotes)-7:
                                top_note += 6
                        # Decrease top_note range
                        if event.key == K_h:    
                            if top_note > 6 and top_note > bot_note + 6:
                                top_note -= 6
                        # Increase bot_note range
                        if event.key == K_u:    
                            if bot_note < top_note:
                                bot_note += 6
                        # Decrease bot_note range
                        if event.key == K_j:    
                            if bot_note > 6:
                                bot_note -= 6 
                        
                        if event.key == K_a:    
                            auto_scale = not auto_scale
                             
                        if event.key == K_z:    
                            needle = not needle
                            
                           
                        # Max note range
                        if event.key == K_i:    
                            bot_note = 0
                            top_note = len(tunerNotes)-1  
                                
                        # Show the note names    
                        if event.key == K_n:
                            shownotes = not shownotes
                        
                        # Clear screen after every Sweep
                        if event.key == K_m:    
                            cls = not cls      
                        
                        # Random Colours
                        if event.key == K_r:    
                            col = not col    
                        
                        # Fill Circles
                        if event.key == K_f:    
                            fill = not fill    
                        
                        # Display Circles
                        if event.key == K_c:    
                            circ = not circ   
                        
                        # Take Screen Dump
                        if event.key == K_p:    
                            pygame.image.save(screen, ("Pishot-" + str(trys) + "-" + str(n) + ".jpeg"))      
                        # Quit
                        if event.key == K_q:    
                            SR.close()
                            return
                
                ##### use the controls to make changes to the data #####
                
                if stepchange == True:                     #go to start of the loop if the step size is altered
                    stepchange = not stepchange
                    break 
                
                if auto_scale:
                    if bot_note < 55 and bot_note < top_note + 6:
                        bot_note = targetnote - 6
                    if top_note > 5 and top_note > bot_note + 6:
                        top_note = targetnote  + 6
                    auto_scale = False
                
                if col:
                    err = abs(frequencies[targetnote]-inputnote)
                    if err < 1.0:
                        stepsizecolor = (0,255,0)   
                    if err >= 1.0 and err <=2.5:
                        stepsizecolor = (255,255,255)   
                    if err > 2.5:
                        stepsizecolor = (255,0,0)   
                               
                if circ:
                    pygame.draw.circle(screen, stepsizecolor, position, 1 + abs(int(20-signal_level)*3), fill)
                
                if needle:
                    pygame.draw.line(screen, stepsizecolor,centrescreen,(centrescreen[0]+((frequencies[targetnote]-inputnote)*20),centrescreen[1]-200), 3)
                
                if n == 0 or n == screen_size[0]:
                    oldposition = position
                
                if line:
                    if inputnote < frequencies[len(tunerNotes)-1]:                     #### not interested in notes above the notes list
                        if oldposition < position:                                     #### prevent backward line draws 
                            pygame.draw.line(screen, stepsizecolor,oldposition,position, 2)
                        oldposition = position                                         #### memory of position
                
                ####### Draw Stuff on the screen #######                
                
                # write a little info/status box on top of screen
                meter = "###################################"
                font2 = pygame.font.Font(None, 18)
                err = abs(frequencies[targetnote]-inputnote)
                text2 = font2.render("Range: " + str(tunerNotes[frequencies[bot_note]]) + " - " + str(tunerNotes[frequencies[top_note]])
                   + "    Show Notes(n):" + str(shownotes) + "    Circles(c):" + str(circ) 
                   + "    Lines(l):" + str(line) + "    Stepsize(v<>b):" + str(stepsize)
                   + "    Fill Circles(f):" + str(not fill) + "    Rand Clr(r):" + str(col) + "    Clr Screen(m):" + str(cls) + "   Lev: " + meter[1:int(20-signal_level)]
                   , 1, (255,255,0))
                pygame.draw.rect(screen, (0,0,0), (0,0,screen_size[0],20))
                pygame.draw.line(screen, (200,200,200),(0,20),(1024,20), 1)
                screen.blit(text2, (5,5))
                
                # display note names if selected
                if shownotes:            
                    font = pygame.font.Font(None, abs(int(1+(20-signal_level)*7)))
                    err = abs(frequencies[targetnote]-inputnote)
                    if err < 1.5:
                        text = font.render(str(tunerNotes[frequencies[targetnote]]), 1, (0,255,0))
                    if err >= 1.5 and err <=2.5:
                        text = font.render(str(tunerNotes[frequencies[targetnote]]), 1, (255,255,255))
                    if err > 2.5:
                        text = font.render(str(tunerNotes[frequencies[targetnote]]), 1, (255,0,0))
                    screen.blit(text, (position))
                    
                # update the display    
                pygame.display.flip()
                pygame.display.update()
                
                if toggle:                            # clear screen every frame
                    screen.blit(screen, (0, 0))
                    screen.fill(screen_color)
            
            if cls:                                   # clear screen at the end of every loop run
                screen.blit(screen, (0, 0))
                screen.fill(screen_color)
            
            
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1024,768))
    NoteTrainer().main(screen)
    














           
