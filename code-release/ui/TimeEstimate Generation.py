{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import time\n",
    "import json\n",
    "import random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "sys.path.append('../core/python')\n",
    "\n",
    "from commons import inc_inference_with_model, inc_inference, show_heatmap\n",
    "from imagenet_classes import class_names\n",
    "from vgg16 import VGG16\n",
    "from resnet18 import ResNet18\n",
    "from inception3 import Inception3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "image_file_path = \"../images/input/imagenet/dog.jpg\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python2.7/dist-packages/h5py/_hl/dataset.py:313: H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.\n",
      "  \"Use dataset[()] instead.\", H5pyDeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "#vgg_model = VGG16(beta=1.0, gpu=True, n_labels=1000).eval()\n",
    "#resnet_model = ResNet18(beta=1.0, gpu=True, n_labels=1000).eval()\n",
    "#inception size fix issue \n",
    "#inception_model = Inception3(beta=1.0, gpu=True, n_labels=1000).eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python2.7/dist-packages/h5py/_hl/dataset.py:313: H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.\n",
      "  \"Use dataset[()] instead.\", H5pyDeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "def calculateX(xsize, ysize, patch, stride):\n",
    "    return (xsize - patch) * (ysize - patch) * 1.0 / stride / stride\n",
    "\n",
    "'''\n",
    "    choose a list of different sizes, using linear regression to make the approach\n",
    "    in terms of the patch_size as well as stride_size, we choose use random choice\n",
    "'''\n",
    "\n",
    "xsize_list = [224, 128, 168, 112, 168, 56, 56]\n",
    "ysize_list = [224, 224, 224, 112, 112, 112, 56]\n",
    "\n",
    "patch_size = [16, 8, 4, 2, 1]\n",
    "stride_size = [16, 8, 4, 2, 1]\n",
    "\n",
    "len_list = len(xsize_list)\n",
    "\n",
    "vgg_paralist = []\n",
    "vgg_timelist = []\n",
    "resnet_paralist = []\n",
    "resnet_timelist = []\n",
    "\n",
    "'''\n",
    "    there will be illegal memory issue for running the same model multiple times?\n",
    "'''\n",
    "\n",
    "for i in range(len_list):\n",
    "    for j in range(2):\n",
    "        patch_sample = random.choice(patch_size)\n",
    "        stride_sample = random.choice(stride_size)\n",
    "        #print('patch_sample ' + str(patch_sample))\n",
    "        #print('stride_sample ' + str(stride_sample))\n",
    "        \n",
    "        x = calculateX(xsize_list[i], ysize_list[i], patch_sample, stride_sample)\n",
    "        start_time = time.time()\n",
    "        _,_,_ = inc_inference(VGG16, image_file_path, patch_size=patch_sample, stride=stride_sample, x_size=xsize_list[i], y_size=ysize_list[i], gpu=True)\n",
    "        #_,_,_ = inc_inference_with_model(vgg_model, image_file_path, patch_size=patch_sample, stride=stride_sample, x_size=xsize_list[i], y_size=ysize_list[i], gpu=True)\n",
    "        \n",
    "        end_time = time.time()\n",
    "        vgg_paralist.append(x)\n",
    "        vgg_timelist.append(end_time - start_time)\n",
    "        \n",
    "        start_time = time.time()\n",
    "        _,_,_ = inc_inference(ResNet18, image_file_path, patch_size=patch_sample, stride=stride_sample, x_size=xsize_list[i], y_size=ysize_list[i], gpu=True)\n",
    "        #_,_,_ = inc_inference_with_model(resnet_model, image_file_path, patch_size=patch_sample, stride=stride_sample, x_size=xsize_list[i], y_size=ysize_list[i], gpu=True)\n",
    "        \n",
    "        end_time = time.time()\n",
    "        resnet_paralist.append(x)\n",
    "        resnet_timelist.append(end_time - start_time)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "    we have to minimize the ||Ax - b|| --> need to find the optimal approach \n",
    "    Calculate the slope and intercept for a given list for parameters\n",
    "    on the time as well as constant, we'll use closed-form linear regression to derive the slope,\n",
    "    intercept for vgg, resnet. \n",
    "    \n",
    "    vgg_paralist 1 * 14 appends all ones --> to make it A 2 * 14\n",
    "    vgg_timelist 1 * 14\n",
    "'''\n",
    "\n",
    "vgg_para = np.asarray(vgg_paralist).reshape((14,1))\n",
    "vgg_time = np.asarray(vgg_timelist).reshape((14,1))\n",
    "resnet_para = np.asarray(resnet_paralist).reshape((14,1))\n",
    "resnet_time = np.asarray(resnet_timelist).reshape((14,1))\n",
    "ones = np.ones((len(vgg_paralist),1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "A_vgg = np.hstack((vgg_para, ones))\n",
    "A_resnet = np.hstack((resnet_para, ones))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_vgg = np.dot(np.dot(np.linalg.inv(np.dot(A_vgg.T, A_vgg)),A_vgg.T), vgg_time)\n",
    "X_resnet = np.dot(np.dot(np.linalg.inv(np.dot(A_resnet.T, A_resnet)),A_resnet.T), resnet_time)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "slope 0.0008626577254934978 intercept 1.5259596772652921\n"
     ]
    }
   ],
   "source": [
    "print('slope ' + str(X_vgg[0][0]) + ' intercept ' + str(X_vgg[1][0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def abline(slope, intercept, para, time):\n",
    "    \"\"\"Plot a line from slope and intercept\"\"\"\n",
    "    fig = plt.figure()\n",
    "    \n",
    "    axes = plt.gca()\n",
    "    x_vals = np.arange(1,20000,1)\n",
    "    y_vals = intercept + slope * x_vals\n",
    "    ax = fig.add_axes(axes)\n",
    "    ax.plot(x_vals, y_vals, '--')\n",
    "    ax.plot(para, time, 'ro')\n",
    "    \n",
    "    return fig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = abline(X_vgg[0][0], X_vgg[1][0], vgg_paralist, vgg_timelist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Format 'fig' is not supported (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-19-e7340e927e86>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mfig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msavefig\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"time approx.fig\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/usr/local/lib/python2.7/dist-packages/matplotlib/figure.pyc\u001b[0m in \u001b[0;36msavefig\u001b[0;34m(self, fname, **kwargs)\u001b[0m\n\u001b[1;32m   2060\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_frameon\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mframeon\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2061\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2062\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprint_figure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2063\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2064\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mframeon\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python2.7/dist-packages/matplotlib/backend_bases.pyc\u001b[0m in \u001b[0;36mprint_figure\u001b[0;34m(self, filename, dpi, facecolor, edgecolor, orientation, format, **kwargs)\u001b[0m\n\u001b[1;32m   2171\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2172\u001b[0m         \u001b[0;31m# get canvas object and print method for format\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2173\u001b[0;31m         \u001b[0mcanvas\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_output_canvas\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2174\u001b[0m         \u001b[0mprint_method\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'print_%s'\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0mformat\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2175\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python2.7/dist-packages/matplotlib/backend_bases.pyc\u001b[0m in \u001b[0;36m_get_output_canvas\u001b[0;34m(self, fmt)\u001b[0m\n\u001b[1;32m   2103\u001b[0m         raise ValueError(\n\u001b[1;32m   2104\u001b[0m             \u001b[0;34m\"Format {!r} is not supported (supported formats: {})\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2105\u001b[0;31m             .format(fmt, \", \".join(sorted(self.get_supported_filetypes()))))\n\u001b[0m\u001b[1;32m   2106\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2107\u001b[0m     def print_figure(self, filename, dpi=None, facecolor=None, edgecolor=None,\n",
      "\u001b[0;31mValueError\u001b[0m: Format 'fig' is not supported (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)"
     ]
    }
   ],
   "source": [
    "fig.savefig(\"time approx.fig\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {}\n",
    "data['vgg16'] = []\n",
    "data['resnet18'] = []\n",
    "\n",
    "data['vgg16'].append({\n",
    "    'slope': X_vgg[0][0],\n",
    "    'intercept': X_vgg[1][0],\n",
    "})    \n",
    "\n",
    "data['resnet18'].append({\n",
    "    'slope': X_resnet[0][0],\n",
    "    'intercept': X_resnet[1][0],\n",
    "})    \n",
    "\n",
    "with open('time-estimation.txt', 'w') as outfile:\n",
    "    json.dump(data, outfile)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.70122396])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_vgg[1][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "    In this auto-configuration, we're targeting on find the intercept & slope for each model based on current GPU\n",
    "    in order to repeat the real scene of the entire time from setting up a model from scratch \n",
    "    parameters affecting the running time are: stride, patch, model, x_size , y_size\n",
    "    However, in one GPU, number of calcualation for a batch size is certain, therefore, we can estimate time to compute \n",
    "    depends on the these input, basically number of computations is on proportional to \n",
    "    number of computations = (x_size - patch) * (y_size - patch) / stride / stride\n",
    "'''\n",
    "\n",
    "'''\n",
    "def makeEstimation(model_class, image_file_path):\n",
    "    slope, intercept = get_parameters(model_class, image_file_path)\n",
    "    return slope, intercept\n",
    "    \n",
    "    \n",
    "def get_parameters(model_class, image_file_path):\n",
    "\tprint('start configuration')\n",
    "\tbegin_time1 = time.time()\n",
    "\t_, _, _ = inc_inference(model_class, image_file_path, patch_size=16, stride=8, beta=1.0, x0=0, y0=0, x_size=224, y_size=224, gpu=True, c=0.0)\n",
    "\tend_time1 = time.time()\n",
    "\tpatch, stride = 16, 8\n",
    "\ty1 = end_time1 - begin_time1\n",
    "\tx1 = (224 - patch) * (224 - patch) * 1.0 / stride / stride\n",
    "\tprint('1 done')\n",
    "\tbegin_time2 = time.time()\n",
    "\t_, _, _ = inc_inference(model_class, image_file_path, patch_size=16, stride=2, beta=1.0, x0=0, y0=0, x_size=100, y_size=100, gpu=True, c=0.0)\n",
    "\tend_time2 = time.time()\n",
    "\tpatch, stride = 16, 2\n",
    "\tprint('2 done')\t\n",
    "\ty2 = end_time2 - begin_time2\n",
    "\tx2 = (100 - patch) * (100 - patch) * 1.0 / stride / stride\n",
    "\talpha = (y2 - y1) / (x2 - x1)\n",
    "\tbeta = y1 - alpha * x1\n",
    "\treturn alpha, beta\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# begin_time = time.time()\n",
    "# _,_,_ = inc_inference_with_model(vgg_model, image_file_path, patch_size=4, stride=4, beta=1.0, x0=0, y0=0, image_size=224, x_size=224,\n",
    "#                       y_size=224, version='v1', gpu=True, c=0.0)\n",
    "# end_time = time.time()\n",
    "# time_elapsed = end_time - begin_time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "    In this auto-configuration, we're targeting on find the intercept & slope for each model based on current GPU\n",
    "    we'll plug in 2 samples to calculate the slope & intercept and save them into json file. \n",
    "    In this way, we can save lots of time in calculating time-estimation.\n",
    "'''\n",
    "\n",
    "def makeEstimation(model, image_file_path):\n",
    "    slope, intercept = auto_configure(model, image_file_path)\n",
    "    return slope, intercept\n",
    "    \n",
    "    \n",
    "def auto_configure(model, image_file_path):\n",
    "\tprint('start configuration')\n",
    "\tbegin_time1 = time.time()\n",
    "\t_, _, _ = inc_inference_with_model(model, image_file_path, patch_size=16, stride=8, beta=1.0, x0=0, y0=0, x_size=224, y_size=224, gpu=True, c=0.0)\n",
    "\tend_time1 = time.time()\n",
    "\tpatch, stride = 16, 8\n",
    "\ty1 = end_time1 - begin_time1\n",
    "\tx1 = (224 - patch) * (224 - patch) * 1.0 / stride / stride\n",
    "\tprint('1 done')\n",
    "\tbegin_time2 = time.time()\n",
    "\t_, _, _ = inc_inference_with_model(model, image_file_path, patch_size=16, stride=2, beta=1.0, x0=0, y0=0, x_size=100, y_size=100, gpu=True, c=0.0)\n",
    "\tend_time2 = time.time()\n",
    "\tpatch, stride = 16, 2\n",
    "\tprint('2 done')\t\n",
    "\ty2 = end_time2 - begin_time2\n",
    "\tx2 = (100 - patch) * (100 - patch) * 1.0 / stride / stride\n",
    "\talpha = (y2 - y1) / (x2 - x1)\n",
    "\tbeta = y1 - alpha * x1\n",
    "\treturn alpha, beta\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pt: 1\n",
      "st: 1\n",
      "x: 1\n",
      "x: 2\n",
      "x: 3\n",
      "x: 4\n",
      "x: 5\n",
      "x: 6\n",
      "x: 7\n",
      "x: 8\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "start the experimeng with vgg16\n",
    "iterate all time-possibilities of vgg16\n",
    "store all the patch,stride,height,width possibilities into .json \n",
    "as the backend for later estimate time cost\n",
    "of running Krypton\n",
    "'''\n",
    "patch_dict = [1,2,4,8,16]\n",
    "stride_dict = [1,2,4,8,16]\n",
    "\n",
    "data = {}\n",
    "data['vgg16'] = []\n",
    "\n",
    "for pt in patch_dict:\n",
    "    print('pt: ' + str(pt))\n",
    "    for st in stride_dict:\n",
    "        print('st: ' + str(st))\n",
    "        for x in range(1,225):\n",
    "            print('x: ' + str(x))\n",
    "            for y in range(1,225):\n",
    "                #print('y: ' + str(y))\n",
    "                begin_time = time.time()\n",
    "                _,_,_ = inc_inference_with_model(vgg_model, image_file_path, patch_size=pt, stride=st, beta=1.0, x0=0, y0=0, image_size=224, x_size=x,\n",
    "                                      y_size=y, version='v1', gpu=True, c=0.0)\n",
    "                end_time = time.time()\n",
    "                time_elapsed = end_time - begin_time\n",
    "                data['vgg16'].append({\n",
    "                    'patch': pt,\n",
    "                    'stride': st,\n",
    "                    'width': x,\n",
    "                    'height': y,\n",
    "                    'time': time_elapsed\n",
    "                })\n",
    "        \n",
    "        \n",
    "with open('time-estimation.txt', 'w') as outfile:\n",
    "    json.dump(data, outfile)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
