import numpy as np
import matplotlib.pyplot as plt

import pkgutil

# Get a list of all available packages
packages = [name for _, name, _ in pkgutil.iter_modules()]

# Print the list of packages
print(packages)
"""
import subprocess

# Specify the package you want to install
package_name = "nengo"

# Use pip to install the package
subprocess.call(['pip', 'install', package_name])
"""