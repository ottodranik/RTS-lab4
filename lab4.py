# Laba4. FFT

import sys
sys.path.append('../lab1')

import numpy as np
import matplotlib.pyplot as plt
import math

from random import random
from lab1 import generate_signal, draw, DrawOption, N, n, w, A, x


def fft(signal):
  def factor(pk, n):
    angle = -2 * math.pi / n * pk
    return complex(math.cos(angle), math.sin(angle))
  
  def inner_fft(signal, p, level_factor):
    n = len(signal)
    next_n = n // 2
    next_p = p % next_n
    if n > 2:
      signal_odd = np.array([signal[i] for i in range(1, n) if i % 2 == 1])
      signal_pair = np.array([signal[i] for i in range(n) if i % 2 == 0])
      next_factor = factor(next_p, next_n)
      f_odd = inner_fft(signal_odd, next_p, next_factor)
      f_pair = inner_fft(signal_pair, next_p, next_factor)
      return f_pair + level_factor * f_odd
    
    w_odd = -1 if p % 2 else 1
    return signal[0] + signal[1] * w_odd
  
  length = len(signal)
  result = np.array([inner_fft(signal, p, factor(p, length)) for p in range(length)])
  real, image = np.array([i.real for i in result]), np.array([i.imag for i in result])
  return real, image

def main_fn():
  signal = np.array([generate_signal(i) for i in range(N)])
  real, imagine = fft(signal)
  options = [
    DrawOption("Signal", "plot"),
    DrawOption("FFT: Real", "bar"),
    DrawOption("FFT: Imaginary", "bar")
  ]
  draw([signal, real, imagine], options, "lab4.png")

if __name__ == '__main__':
  main_fn()