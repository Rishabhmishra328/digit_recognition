# Digit recognition
Digit recognition using linear regression on MNIST digit dataset.

## Tensorflow installation
*See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions on how to install our release binaries or how to build from source.*

**Individual whl files**
* Linux CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp27-none-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/)) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp34-cp34m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/)) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp35-cp35m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/))
* Linux GPU: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/42/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp27-none-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/)) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp34-cp34m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/)) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp35-cp35m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/))
* Mac CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-py2-none-any.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/)) / [Python 3](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-py3-none-any.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/))
* Windows CPU-only: [Python 3.5 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly-1.head-cp35-cp35m-win_amd64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly-1.head-cp36-cp36m-win_amd64.whl) ([build history](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=36/))
* Windows GPU: [Python 3.5 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly_gpu-1.head-cp35-cp35m-win_amd64.whl) ([build history](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly_gpu-1.head-cp36-cp36m-win_amd64.whl) ([build history](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=36/))
* Android: [demo APK](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk), [native libs](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/)
([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-android/))

#### *Try your first TensorFlow program*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```


## Requirements

-Tensorflow

## To run
To run the CNN network, run the python compiler in the source directory and type the following:
```python
>>> import train as t
>>> model = t.Train(model = 'cnn', learning_rate = 0.001, epochs = 5, batch =20, display_step = 1)
>>> model.train()
```
Similarly, for a linear regression model, run the following:
```python
>>> import train as t
>>> model = t.Train(model = 'linear_regression', learning_rate = 0.001, epochs = 500, batch =100, display_step = 1)
>>> model.train()
 -```
