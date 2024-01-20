#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from termcolor import colored, cprint
import logging
class DispatchingFormatter:
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)

class DispatchingFormatter0:
    """Dispatch formatter for logger and it's sub logger."""
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        # Search from record's logger up to it's parents:
        logger = logging.getLogger(record.name)
        while logger:
            # Check if suitable formatter for current logger exists:
            if logger.name in self._formatters:
                formatter = self._formatters[logger.name]
                break
            else:
                logger = logger.parent
        else:
            # If no formatter found, just use default:
            formatter = self._default_formatter
        return formatter.format(record)

class Logger:
    level_dict = {
        'NOTSET'  : logging.NOTSET,
        'DEBUG'   : logging.DEBUG,
        'INFO'    : logging.INFO,
        'WARNING' : logging.WARNING,
        'ERROR'   : logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        0 : logging.NOTSET,
        1 : logging.DEBUG,
        2 : logging.INFO,
        3 : logging.WARNING,
        4 : logging.ERROR,
        5 : logging.CRITICAL,
    }
    Formatters = { 
            'c' : logging.Formatter( '[%(asctime)s] [%(levelname)-4s]: %(message)s', '%Y-%m-%d %H:%M:%S'),
            'p' : logging.Formatter( '[%(levelname)-4s]: %(message)s'),
            'n' : logging.Formatter( '%(message)s' ),
    }
    ChangeFrom = DispatchingFormatter(Formatters, Formatters['n'])

    def __init__(self, out=None, filemode='w', clevel = 'INFO', Flevel = 'INFO'):
        
        logging.basicConfig(
            level    = Logger.level_dict[clevel] ,
            format   = '[%(asctime)s] [%(levelname)-4s]: %(message)s',
            datefmt  = '%Y-%m-%d %H:%M:%S',
            filename = None,
        )
        if (logging.getLogger().hasHandlers()):
            logging.getLogger().handlers.clear()

        if not out in [None, '']:
            File = logging.FileHandler(out, mode= filemode, encoding=None, delay=False)
            File.setLevel(Logger.level_dict[Flevel])
            File.setFormatter(Logger.ChangeFrom)
            logging.getLogger().addHandler(File)

        Hand = logging.StreamHandler()
        Hand.setFormatter(Logger.ChangeFrom)
        logging.getLogger().addHandler(Hand)

        self.R = logging
        self.C = logging.getLogger('c')
        self.P = logging.getLogger('p')
        self.N = logging.getLogger('n')
        self.CI = logging.getLogger('c').info
        self.PI = logging.getLogger('p').info
        self.NI = logging.getLogger('n').info
        self.CW = logging.getLogger('c').warning
        self.PW = logging.getLogger('p').warning
        self.NW = logging.getLogger('n').warning

    def ci(self, *arg, **kargs):
        self.CI(colored(*arg, **kargs))

    def pi(self, *arg, **kargs):
        self.PI(colored(*arg, **kargs))

    def ni(self, *arg, **kargs):
        self.NI(colored(*arg, **kargs))

    def cw(self, *arg, **kargs):
        self.CW(colored(*arg, **kargs))

    def pw(self, *arg, **kargs):
        self.PW(colored(*arg, **kargs))

    def nw(self, text, color=None, on_color=None, attrs=None):
        self.NW(colored(text, color=color, on_color=on_color, attrs=attrs))
