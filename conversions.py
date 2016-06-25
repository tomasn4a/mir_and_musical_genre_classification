# -*- coding: utf8 -*-

import subprocess
import os


###############################################################################
# Convert all mp3 to mono wav files recursively from a directory
def mass_convert(main_dir, solution="ffmpeg", rate="44100"):
  os.chdir(main_dir)

  for root, dirs, files in os.walk(main_dir):
      for file in files:
          if file.endswith(".mp3") or file.endswith(".m4a"):
              file_in = root+"/"+file
              #file_out = "".join(i for i in str(root+"/"+file[:-4]+".wav") if ord(i)<128)
              file_out = str(root + "/" + file[:-4] + ".wav")

              if solution == "ffmpeg":
                # -ar sets bitrate and -ac the number of output channels
                subprocess.call(['ffmpeg', '-i',
                  file_in, '-y', '-ar', rate, '-ac', '1', file_out])

              elif solution == "mpg123":
                # -m sets the output to mono
                subprocess.call(['mpg123', '--wav',
                  file_out, '--rate', rate, '-m', file_in])

              else:
                raise ValueError('must choose between ffmpeg or mpg123')

              mp3_file = os.path.join(root, file)
              tail, track = os.path.split(mp3_file)
              tail, dir1 = os.path.split(tail)
              tail, dir2 = os.path.split(tail)

              ######### ATTENTION ! #########################################

              # This line will DELETE your mp3 file. Don't uncomment unless
              # necessary
              # os.remove(file_in)


# ##############################################################################
# # Evaluate performance of ffmpeg and music123
#
# # Wrapper for timing function
# def wrapper(func, *args, **kwargs):
#   def wrapped():
#     return func(*args, **kwargs)
#   return wrapped
#
# # Timing and comparing ffmpeg and mpg123
# rootDir = '/Users/tomas/Music/Chris Thile/Bach_ Sonatas & Partitas, Vol. 1'
# sol = "ffmpeg"
# rate = "44100"
# wrapped = wrapper(mass_convert, rootDir, sol, rate)
# t = timeit.timeit(wrapped, number=1)
# print "\n {}".format(t)
#
# # ffmpeg: 159.209589005
# # mpg123: 76.0900411606
#
#
# ##############################################################################
# # Mass convert a directory and its contents
#
# mass_convert('/Users/tomas/Music/Genres')

