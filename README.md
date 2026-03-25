The project aims at parallel image transform and filtering of gigapixel images with heterogeneous computing (CPU + GPU), creating a system capable of processing images larger than available memory with high throughput. The core challenge is out-of-core processing, handling data larger than working memory by streaming it in chunks, never loading more than a small portion of the image at once.

**Group Members:**
**Gehna Bhatia**
**Muhammad Zayed Nauman**
**Zainab Irfan**
**Muhammad Anis Imran**

Steps for compiling and executing the files:

cd "/Users/.../Project"
python3 -m venv venv
source venv/bin/activate
pip install numpy tifffile
python -c "import numpy as np; import tifffile as tf; a=tf.imread('test_2048x2048_rgb.tiff'); b=tf.imread('out.tiff'); print('PASS' if np.array_equal(a,b) else 'FAIL')"
