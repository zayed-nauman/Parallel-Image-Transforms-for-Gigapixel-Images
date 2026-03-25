The project targets parallel image transformation and filtering of gigapixel images using heterogeneous computing (CPU + GPU).

Group Members:
*Gehna Bhatia*
*Muhammad Zayed Nauman*
*Zainab Irfan*
*Muhammad Anis Imran*

Steps for compiling and executing the files:

cd "/Users/.../Project"
python3 -m venv venv
source venv/bin/activate
pip install numpy tifffile
python -c "import numpy as np; import tifffile as tf; a=tf.imread('test_2048x2048_rgb.tiff'); b=tf.imread('out.tiff'); print('PASS' if np.array_equal(a,b) else 'FAIL')"
