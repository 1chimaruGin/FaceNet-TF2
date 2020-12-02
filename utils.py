import os


def check_folder(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  return dir_name