import os
import sys
import time
import types
import atexit
import shutil
import inspect
import itertools
from contextlib import contextmanager

from termcolor import colored

if os.name == 'nt':
    # pylint: disable=import-error
    import colorama
    colorama.init()


class LoggerFail(Exception):
    """Failure.  """


class Logger:
    _levels = (
        'debug', 'verbose', 'info', 'key', 'warn', 'error', 'fail', 'off')
    _colors = ('cyan', 'white', 'blue', 'green', 'magenta', 'yellow', 'red')
    _signs = ('?', '·', '-', '*', '!', '‼', 'x')
    _spinner = itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
    _spinner_done = '⣿'

    _default_width = 200

    def __init__(self):
        super().__init__()
        self.level = 'info'
        self.pause_level = 'fail'
        self.frame = False
        self.color = 'color' in os.environ.get('TERM', '')
        self._last_is_update = False
        self._last_use_spinner = True
        self._last_level = self.level
        self._once = []

    @property
    def width(self):
        try:
            return self._width
        except AttributeError:
            pass
        width, _ = shutil.get_terminal_size((self._default_width, 24))
        return width

    @width.setter
    def width(self, value):
        self._width = value

    @classmethod
    def _level_key(cls, level):
        return cls._levels[level]

    @property
    def level(self):
        return self._level_key(self._level)

    @level.setter
    def level(self, value):
        self._level = self._levels.index(value)

    def is_enabled(self, level):
        return self._level <= self._levels.index(level)

    @contextmanager
    def use_level(self, level):
        prev_level = self.level
        self.level = level
        yield
        self.level = prev_level

    @property
    def pause_level(self):
        return self._level_key(self._pause_level)

    @pause_level.setter
    def pause_level(self, value):
        self._pause_level = self._levels.index(value)
        self.debug(f'Log pause level: {value}')

    @contextmanager
    def use_pause_level(self, level):
        prev_level = self.pause_level
        self.pause_level = level
        yield
        self.pause_level = prev_level

    @contextmanager
    def demote(self):
        _key = self.key
        _info = self.info
        self.key = _info
        self.info = self.debug
        yield
        self.key = _key
        self.info = _info

    def colored(self, text, level):
        return colored(text, self._colors[self._levels.index(level)])

    def debug_colored(self, text):
        return self.colored(text, 'debug')

    def verbose_colored(self, text):
        return self.colored(text, 'verbose')

    def info_colored(self, text):
        return self.colored(text, 'info')

    def key_colored(self, text):
        return self.colored(text, 'key')

    def warn_colored(self, text):
        return self.colored(text, 'warn')

    def error_colored(self, text):
        return self.colored(text, 'error')

    def _frame_info(self):
        # facepalm
        frame = inspect.currentframe().f_back.f_back.f_back.f_back
        file_name = frame.f_code.co_filename
        file_name = os.path.split(file_name)[1]
        file_name = os.path.splitext(file_name)[0]
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        return f'{file_name}:{func_name}#{line_no}'

    def _header(self, text, level, spinner):
        if spinner:
            sign = next(self._spinner)
        else:
            sign = self._signs[self._levels.index(level)]
        if self.frame:
            sign = self._frame_info()
        return f'{self.colored(sign, level)} {text}'

    def log(self, text, level='info', update=False, spinner=True, once=None):
        # pylint: disable=import-outside-toplevel
        if once is not None:
            if once in self._once:
                return
            self._once.append(once)
        num_level = self._levels.index(level)
        if self._level > num_level:
            return
        if update:
            begin = '\r'
            end = ''
            header_len = 4
            width = self.width - header_len
            text += ' ' * width
            text = text[:width]
        else:
            begin = ''
            end = '\n'
        text = self._header(text, level, update and spinner)
        if not update and self._last_is_update:
            if self._last_use_spinner:
                tick = self.colored(self._spinner_done, self._last_level)
                begin = f'\r{tick}\n{begin}'
            else:
                begin = f'\n{begin}'
        print(begin + text, end=end, flush=True)
        self._last_is_update = update
        self._last_use_spinner = update and spinner
        self._last_level = level
        while num_level >= self._pause_level:
            r = input(
                'Continue [Return], Stack trace [t], '
                'Debugger [d], Abort [q], Raise [r]: ')
            if not r:
                break
            frame = inspect.currentframe().f_back.f_back
            if r == 'd':
                import ipdb
                ipdb.set_trace()
            elif r == 't':
                import traceback
                traceback.print_stack(frame)
            elif r == 'q':
                sys.exit(-1)
            elif r == 'r':
                tb = types.TracebackType(
                    None, frame, frame.f_lasti, frame.f_lineno)
                raise LoggerFail.with_traceback(tb)

    def debug(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'debug', update, spinner, once)

    def verbose(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'verbose', update, spinner, once)

    def info(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'info', update, spinner, once)

    def key(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'key', update, spinner, once)

    def warn(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'warn', update, spinner, once)

    def error(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'error', update, spinner, once)

    def fail(self, text, update=False, spinner=True, once=None):
        return self.log(text, 'fail', update, spinner, once)

    def fail_exit(self, fail_msg):
        with self.use_level('fail'):
            with self.use_pause_level('off'):
                self.fail(fail_msg)
        sys.exit(-1)

    def countdown(self, text, secs, level='info'):
        try:
            for i in range(secs):
                msg = f'{text} in {secs - i} seconds... (Abort: ctrl+c)'
                self.log(msg, level, update=True, spinner=False)
                time.sleep(1)
            return True
        except KeyboardInterrupt:
            log.verbose('We give up.')
            return False

    def exit(self):
        # emit an empty line, as last log has no carriage return
        if self._last_is_update:
            print()


log = Logger()
atexit.register(log.exit)
