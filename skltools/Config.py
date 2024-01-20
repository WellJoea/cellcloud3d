#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import os
import time


from Logger import Logger

info = '''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : %s                       *
* E-mail : welljoea@gmail.com                             *
* You are using MLkit scripted by Zhou Wei.               *
* If you find some bugs, please email to me.              *
* Please let me know and acknowledge in your publication. *
* Sincerely                                               *
* Best wishes!                                            *
***********************************************************
'''%(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

class _Setlog:
    def __init__(self, outdir=None, outfile=None, clevel = 'INFO', Flevel = 'INFO', cml=False):
        self.time = time.strftime("%Y%m%d%H%M", time.localtime())
        self.info = info.strip()
        self._outdir = outdir
        self._outfile = outfile
        self._clevel = clevel
        self._Flevel = Flevel
        
        self.cml = cml
        if self.cml:
            try:
                from Arguments import Args
                self.arginfo = [ self.log.NI('**%s|%-13s: %s'%(str(i).zfill(2), k, str(getattr(self.args, k))) ) 
                             for i,k in enumerate(vars(self.args), start=1) ]
                            self.args = Args()
            except:
                pass

    @property
    def clevel(self):
        return self._clevel
    @clevel.setter
    def clevel(self, values):
        self._clevel= values

    @property
    def Flevel(self):
        return self._Flevel
    @Flevel.setter
    def Flevel(self, values):
        self._Flevel = values

    @property
    def outdir(self):
        return self._outdir
    @outdir.setter
    def outdir(self, outdir):
        self._outdir = outdir
        if self._outdir : 
            os.makedirs( os.path.dirname(self._outdir) , exist_ok=True)
            #os.chdir(self._outdir)

    @property
    def outfile(self):
        return self._outfile
    @outfile.setter
    def outfile(self, outfile):
        outfile = f'{self.outdir}/{outfile}'
        self._outfile = outfile
        
    @property
    def log(self):
        return (Logger(out=self.outfile, clevel=self.clevel, Flevel=self.Flevel))
    
    #@classmethod
    #def log(cls):
    #    return (Logger(out=cls.outfile, clevel=cls.clevel, Flevel=cls.Flevel))
    
Config=_Setlog()